import argparse

import time
from pathlib import Path
import os
import yaml
import logging
import shutil
import numpy as np
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm
from template import imagenet_templates
import fast_stylenet
from sampler import InfiniteSamplerWrapper
import clip
from template import imagenet_templates
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms.functional import adjust_contrast
from runners.image_editing import Diffusion
cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None 
ImageFile.LOAD_TRUNCATED_IMAGES = True
print(torch.cuda.is_available())
def train_transform(crop_size=256):
    transform_list = [
        #transforms.RandomCrop(crop_size),
        transforms.Resize(crop_size),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)
def test_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def hr_transform():
    transform_list = [
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def img_normalize(image):
    mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
    std=torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

def load_image(img_path, img_size=None):
    
    image = Image.open(img_path)
    if img_size is not None:
        image = image.resize((img_size, img_size))  
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        ])   
    image = transform(image)[:3, :, :].unsqueeze(0)

    return image


        
def clip_normalize(image):
    image = F.interpolate(image,size=224,mode='bicubic')
    mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

def reverse_normalize(image):
    mean=torch.tensor([-0.485/0.229, -0.456/0.224, -0.406/0.225]).to(device)
    std=torch.tensor([1./0.229, 1./0.224, 1./0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image
def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    
    return loss_var_l2

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
device = torch.device('cuda')
#=================================================================================================================================================================#

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--sample', action='store_true', help='Whether to produce samples from the model')
    parser.add_argument('-i', '--image_folder', type=str, default='./outputs', help="The folder name of samples")
    parser.add_argument('--save_mod', type=str, default='./mod', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--npy_name', type=str, required=True)
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=500, help='Sampling noise scale')

    #loss
    parser.add_argument('--id_lambda', default=0.1, type=float, help='ID loss multiplier factor')
    parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
    parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
    parser.add_argument('--res_lambda', default=0.1, type=float, help='L2 loss multiplier factor')
    parser.add_argument('--board_interval', default=10, type=int,
                        help='Interval for logging metrics to tensorboard')
    parser.add_argument('--lpips_type', default='alex', type=str, help='LPIPS backbone')

    # Basic options
    parser.add_argument('--content_dir', type=str, default ='./datasets/chu_train')
    parser.add_argument('--test_dir', type=str, default ='./test_set')
    parser.add_argument('--hr_dir', type=str)
    parser.add_argument('--img_dir', type=str, default ='./test_set')
    parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')

    # training options
    parser.add_argument('--save_dir', default='./model_fast',
                        help='Directory to save the model')

    parser.add_argument('--text', default='Fire',
                        help='text condition')
    parser.add_argument('--name', default='none',
                        help='name')

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)

    parser.add_argument('--max_iter', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--clip_weight', type=float, default=10.0)
    parser.add_argument('--tv_weight', type=float, default=1e-4)
    parser.add_argument('--glob_weight', type=float, default=1.0)
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--num_test', type=int, default=16)
    parser.add_argument('--save_model_interval', type=int, default=200)
    parser.add_argument('--save_img_interval', type=int, default=100)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--thresh', type=float, default=0.7)
    parser.add_argument('--decoder', type=str, default='./model_fast/clip_decoder_acrylic.pth.tar')
    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(os.path.join(args.exp, 'image_samples'), exist_ok=True)
    args.image_folder = os.path.join(args.exp, 'image_samples', args.image_folder)
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    else:
        overwrite = False
        if args.ni:
            overwrite = True
        else:
            response = input("Image folder already exists. Overwrite? (Y/N)")
            if response.upper() == 'Y':
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


#======================================================================================================================================================

#=================================================================================================================================================================#

# def adjust_learning_rate(optimizer, iteration_count):
#     """Imitating the original implementation"""
#     #args, config = parse_args_and_config()
#     lr = args.lr / (1.0 + args.lr_decay * iteration_count)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
def main():
    args, config = parse_args_and_config()

    print(">" * 80)
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print("<" * 80)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)


    #弥散模型
    runner = Diffusion(args, config)
    model, ckpt_id, n = runner.image_editing_sample()
    #content_images=runner.sampling(model, ckpt_id, n)
    #decoder 模型
    decoder = fast_stylenet.decoder
    vgg = fast_stylenet.vgg
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    decoder.load_state_dict(torch.load(args.decoder))

    network = fast_stylenet.Net(vgg, decoder)
    network.train()
    network.to(device)
    clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)

    def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
        return [template.format(text) for template in templates]

    source = "a Photo"

    with torch.no_grad():
        #目标文本
        prompt = args.text
        template_text = compose_text_with_templates(prompt, imagenet_templates)
        tokens = clip.tokenize(template_text).to(device)
        #目标文本特征
        text_features = clip_model.encode_text(tokens).detach()
        text_features = text_features.mean(axis=0, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_features_four_dimention =text_features[:,:,None,None]
        #原文本特征
        template_source = compose_text_with_templates(source, imagenet_templates)
        tokens_source = clip.tokenize(template_source).to(device)
        text_source = clip_model.encode_text(tokens_source).detach()
        text_source = text_source.mean(axis=0, keepdim=True)
        text_source /= text_source.norm(dim=-1, keepdim=True)

        text_source_four_dimention=text_source[:,:,None,None]
        #原文本和目标文本的差值
        delta_z=text_features
    content_tf = train_transform(args.crop_size)
    hr_tf = hr_transform()
    test_tf = test_transform()


    content_dataset = FlatFolderDataset(args.content_dir, content_tf)
    test_dataset = FlatFolderDataset(args.test_dir, test_tf)


    augment_trans = transforms.Compose([
        transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
        transforms.Resize(224)
    ])
    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))
    test_iter = iter(data.DataLoader(
        test_dataset, batch_size=args.num_test,
        num_workers=args.n_threads))
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    test_images1 = next(test_iter)
    test_images1 = test_images1.cuda()

    if args.hr_dir is not None:
        hr_dataset = FlatFolderDataset(args.hr_dir, hr_tf)

        hr_iter = iter(data.DataLoader(
            hr_dataset, batch_size=1,
            num_workers=args.n_threads))
        hr_images = next(hr_iter)
        hr_images = hr_images.cuda()

    #content_images_origin_1 = next(content_iter).to(device)
    print("Start finetuning")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler_ft = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    for i in tqdm(range(args.max_iter)):
        #adjust_learning_rate(optimizer, iteration_count=i)

        # lr = args.lr / (1.0 + args.lr_decay * i)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        #print(len(content_iter))
        print("epoch",i)
        content_images_origin_1 = next(content_iter).to(device)
        loss_c, out_img = network(content_images_origin_1)
        content_images,loss_all, encoder_loss_dict = runner.sampling(model, ckpt_id, n, out_img, delta_z, content_images_origin_1)
        #state = {‘model’:model.state_dict(), ‘optimizer’:optimizer.state_dict(), ‘epoch’:epoch}
        if i%99==0:
            torch.save(model, os.path.join(args.save_mod,
                                           f'water_purple_{i}.pth'))
        optimizer.zero_grad()
        loss_all.backward()
        scheduler_ft.step()
        # optimizer.zero_grad()
        # loss_all.backward()
        # optimizer.step()

        # if (i+1)%10==0:
        #     print('loss_content:' + str(loss_c.item()),'loss_patch:' + str(loss_patch.item()),'loss_dir:' + str(loss_glob.item()),'loss_tv:' + str(reg_tv.item()))

        # if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        #     torch.save(model.state_dict(), save_dir /
        #                'net_params_{:d}.pth'.format(i + 1))

        # if (i + 1) % args.save_img_interval ==0 :
        #     with torch.no_grad():
        #         _, test_out1 = network( test_images1)
        #         test_out1 = adjust_contrast(test_out1,1.5)
        #         output_test = torch.cat([test_images1,test_out1],dim=0)
        #
        #         output_name = './output_fast/test1_'+ args.text +'_'+ str(i+1)+'.png'
        #         save_image(output_test, str(output_name),nrow=test_out1.size(0),normalize=True,scale_each=True)
        #
        #         if args.hr_dir is not None:
        #             _, test_out = network(hr_images)
        #             test_out = adjust_contrast(test_out,1.5)
        #             output_name = './output_fast/hr_'+ args.name+'_'+ args.text +'_'+ str(i+1)+'.png'
        #             save_image(test_out, str(output_name),nrow=test_out.size(0),normalize=True,scale_each=True)
            
if __name__ == '__main__':
    start_time=int(time.perf_counter())
    main()
    print("END")
    end_time = int(time.perf_counter())
    print("mints",(end_time-start_time)/60)