import os
import numpy as np
from tqdm import tqdm

import torch
import torchvision.utils as tvu
from criteria.lpips.lpips import LPIPS
from models.diffusion import Model
from functions.process_data import *
from criteria import id_loss, moco_loss
import torch.nn.functional as F
def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


def image_editing_denoising_step_flexible_mask(x, t, *,delta_T,
                                               model,
                                               logvar,
                                               betas):
    """
    Sample from p(x_{t-1} | x_t)
    """
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)

    model_output = model(x, t,delta_T)
    weighted_score = betas / torch.sqrt(1 - alphas_cumprod)
    # print("1 / torch.sqrt(alphas)",1 / torch.sqrt(alphas))
    # print("="*20)
    # print("t",t)
    # print("=" * 20)
    # print("x.shape",x.shape)
    # print("=" * 20)
    # print("extract(1 / torch.sqrt(alphas), t, x.shape)",extract(1 / torch.sqrt(alphas), t, x.shape))
    # print("=" * 20)
    # print("extract(weighted_score, t, x.shape)",extract(weighted_score, t, x.shape))
    mean = extract(1 / torch.sqrt(alphas), t, x.shape) * (x - extract(weighted_score, t, x.shape) * model_output)

    logvar = extract(logvar, t, x.shape)
    #加燥
    noise = torch.randn_like(x)
    # delta_T=torch.sigmoid(delta_T)
    # noise=torch.flatten(noise)
    # noise=noise.reshape(-1,512)

    # noise=noise.reshape(-1,3,256,256)

    mask = 1 - (t == 0).float()
    mask = mask.reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))
    sample = mean + mask * torch.exp(0.5 * logvar) * noise
    sample = sample.float()
    return sample


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        self.global_step = 0
        self.count=0
        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]
        # Initialize loss
        if self.args.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type=self.args.lpips_type).to(self.device).eval()
        self.id_loss = id_loss.IDLoss().to(self.device).eval()

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
            (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

    def image_editing_sample(self):
        print("Loading model")
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        else:
            raise ValueError

        model = Model(self.config)
        # print(model)
        # state_dict = torch.load('/home/lmy/Mr_ZHAO/SDE_RAI/mod/save.pt', map_location=lambda storage, loc: storage).module.state_dict()
        # #state_dict = torch.load('/home/lmy/Mr_ZHAO/SDE_RAI/mod/save.pt')
        # model.load_state_dict(state_dict, strict=False)
        ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
        #model.load_state_dict(ckpt)
        model.load_state_dict(ckpt,
                              strict=False)
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        print("Model loaded")
        ckpt_id = 0

        #download_process_data(path="colab_demo")
        n = self.config.sampling.batch_size
        model.eval()
        print("Start sampling")
        return model, ckpt_id, n
    #===================================================================================================================================

    def print_metrics(self, metrics_dict, prefix):
        list1=[]
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            list1.append(key)
            list1.append(value)
            #print('\t{} = '.format(key), value)
        print(list1)

    # ===================================================================================================================================

    def calc_loss(self, x, y, y_hat, res_delta):
        loss_dict = {}
        loss = 0.0
        id_logs = None

        if self.args.id_lambda > 0:
            loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, x)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_id * self.args.id_lambda
        if self.args.l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, y)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.args.l2_lambda
        if self.args.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat, y)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.args.lpips_lambda

        if self.args.res_lambda > 0:
            target = torch.zeros_like(res_delta)
            loss_res = F.l1_loss(res_delta, target)
            loss_dict['loss_res'] = float(loss_res)
            loss += loss_res * self.args.res_lambda

        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs

    #================================================================================================================================
    def sampling(self,model,ckpt_id,n,content_images_origin,delta,content_images_origin_1):
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        # scheduler_ft = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        delta_T = content_images_origin-content_images_origin_1
        loss_dict = {}
        #with torch.no_grad():
        name = self.args.npy_name
        img = content_images_origin.to(self.config.device)
        x0 = img
        if self.count % 1 == 0:
            tvu.save_image(x0, os.path.join(self.args.image_folder,
                                                       f'original_style_{self.count}.png'))
            tvu.save_image(content_images_origin_1, os.path.join(self.args.image_folder,
                                            f'original_{self.count}.png'))
        x0 = (x0 - 0.5) * 2.
        res_images=x0
        #tvu.save_image(x0, os.path.join(self.args.image_folder, f'original_input1.png'))
        for it in range(self.args.sample_step):
            model.train()
            #optimizer.zero_grad()
            e = torch.randn_like(x0)
            tvu.save_image(e, os.path.join(self.args.image_folder, f'random_noise_{ckpt_id}.png'))
            total_noise_levels = self.args.t
            a = (1 - self.betas).cumprod(dim=0)
            # if it>=1:
            #     x = res_images * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
            # else:
            #     x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
            x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder, f'init_{ckpt_id}.png'))
            with torch.no_grad():
                with tqdm(total=total_noise_levels, desc="Iteration {}".format(it)) as progress_bar:
                    #for i in reversed(range(total_noise_levels)):
                    for i in range(total_noise_levels):
                        t = (torch.ones(n) * i).to(self.device)
                        x_ = image_editing_denoising_step_flexible_mask(x, t=t, delta_T=delta_T,model=model,
                                                                        logvar=self.logvar,
                                                                        betas=self.betas)
                        # x = x0 * a[i].sqrt() + e * (1.0 - a[i]).sqrt()
                        # x[:, (mask != 1.)] = x_[:, (mask != 1.)]
                        x=x_
                        #added intermediate step vis
                        if (i - 99) % 100 == 0 or i==0:
                            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                       f'noise_t_{i}_{self.count}.png'))
                        #progress_bar.update(1)
            #
            # #x0[:, (mask != 1.)] = x[:, (mask != 1.)]
            if self.count % 1 == 0:
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                       f'samples_{self.count}.png'))
            # if self.count%100==0:
            #     torch.save(x, os.path.join(self.args.image_folder,
            #                                f'samples_{self.count}.pth'))
            res_gt = (content_images_origin_1 - x).detach()
        #res_gt = (content_images_origin_1 - x).detach()
            loss, encoder_loss_dict, id_logs = self.calc_loss(content_images_origin_1, content_images_origin_1, x, res_gt)
            loss_dict = {**loss_dict, **encoder_loss_dict}
            loss.requires_grad_(True)
            # # optimizer.zero_grad()
            # loss.backward()
            # scheduler_ft.step()
            # if self.global_step % self.args.board_interval == 0:
            self.print_metrics(loss_dict, prefix='train')
            self.count+=1

        return x, loss, encoder_loss_dict