# SDE-RAE
IMAGE REC AND EDIT

Code and dataset coming soon

[**Project**](https://github.com/haizhu12/SDE-RAE) | [**Paper**](coming soon)

# SDE-RAE: Image reconstruction and edit Networks for Stochastic Differential Equation Optimization
PyTorch implementation of SDE-RAE 
 

![IVC_first_1](https://github.com/haizhu12/SDE-RAE/assets/93024130/1f40f089-bba2-437e-b4a3-ffa1086f147f)


##  Quickstart 

Follow the instructions below to download and run SDE-RAE on your own local. These instructions have been tested on a GPU with >18GB VRAM. If you don't have a GPU, you may need to change the default configuration.

### Set up a conda environment, and download a pretrained model:
Pytorch 1.9.1, Python 3.8
```
conda env create -n sde_rae
conda activate sde_rae
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install matplotlib numpy PyYAML tensorboard tqdm
```
[pytorch](https://pytorch.org/get-started/previous-versions/)

## Getting Started
The code will automatically download pretrained SDE (VP) PyTorch models on
[CelebA-HQ](https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt),
[LSUN bedroom](https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt)

### Re-training the model
Here is the [PyTorch implementation](https://github.com/ermongroup/ddim) for training the model.

### dataset
Here is the dataset [celeba-HQ](https://paperswithcode.com/dataset/celeba-hq) And [lsun](https://www.yf.io/p/lsun)

### pretrained
download pretrained [pretrained](https://pan.baidu.com/s/1MSegzm-7SprYBKG0Bg1whA?pwd=k8gb) 提取码：k8gb

unzip to "./pretrained"
### TRAINING:
Download clip-encoder, unzip it to model_fast [clip-encoder](https://pan.baidu.com/s/1U17dgOoH5HiFImwsI0E0fg?pwd=7gyz 
)提取码：7gyz
```
python train_fast.py --content_dir ./datasets/celeba_train --npy_name celeba --num_test 16 --decoder ./model_fast/clip_decoder_pencil.pth.tar
```
### TESTING:

```
python test_fast.py --content_dir ./datasets/celeba_test --npy_name celeba --config celeba.yml --max_iter 10000 --batch_size 4
```

### method:
...
REC
...
![train](https://github.com/haizhu12/SDE-RAE/assets/93024130/b3e621c8-2da1-49cd-9b44-5585bfe2fa75)
...
EDIT
...
![L_SENH](https://github.com/haizhu12/SDE-RAE/assets/93024130/a90352d0-6926-408f-bed7-2d02b797fdf0)

Acknowledgements

### Score-Based Generative Modeling through Stochastic Differential Equations [sde](https://github.com/yang-song/score_sde_pytorch)
### SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations  [SDEITE](https://github.com/ermongroup/SDEdit)
### High-Fidelity GAN Inversion for Image Attribute Editing [HFGI](https://tengfeiwang.github.io/HFGI/)
### Learning Transferable Visual Models From Natural Language Supervision [cliP](https://github.com/OpenAI/CLIP)
### StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery (ICCV 2021 Oral)[StyleCLIP](https://github.com/orpatashnik/StyleCLIP)
### StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators[stylegan-nada](https://stylegan-nada.github.io/)
### One-Shot Adaptation of GAN in Just One CLIP (TPAMI)[OneshotCLIP](https://github.com/anon96652/OneshotCLIP)
### VQGAN-CLIP[vqgan-clip](https://github.com/EleutherAI/vqgan-clip#vqgan-clip)
