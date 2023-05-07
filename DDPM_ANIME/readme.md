## Denoising Diffusion Probabilistic Model for Generating Anime face

#### 1. Introduction
- Here we will train a diffusion model to generate anime face 
- The dataset can be downloaded from [kaggle anime face dataset](https://www.kaggle.com/datasets/splcher/animefacedataset), download the dataset to `dataset` directory and put all the images under directory `anime/raw/images`, when you finish, the dataset looks like this:
```text
dataset
├── anime
│   └── raw
│   │   └── images
│   │       ├── 46651_2014.jpg
│   │       ├── 4665_2003.jpg
│   │       ├── ...
```
- Then we have to process these raw images, we've already done it, you can check this step following [VAE_ANIME](../VAE_ANIME), then your directory looks like this:
```text
dataset
├── anime
│   ├── processed
│   │   └── images
│   │       ├── 46651_2014.jpg
│   │       ├── 4665_2003.jpg
│   │       ├── ...
│   └── raw
│   │   └── images
│   │       ├── 46651_2014.jpg
│   │       ├── 4665_2003.jpg
│   │       ├── ...
```

#### 2. Load dataset, Build model, Train model
- For this task we follow the original paper's model, but it's slightly different, if you want to use the cifar10 model in paper, change the following parameters
```shell
ch = 128
ch_mult = [1, 1, 2, 2]
attn = [1]  # only in 16 * 16 resolution we use attention
```
This model takes up a lot of video memory
- Note that this model cost lots of cuda memory(~20GB), if you use the cifar10 model setting, it will cost about 24GB cuda memory, make sure you have enough memory or you have to reduce batch size
- Here I just use a NVIDIA GeForce RTX 3090 to train, each epoch will cost about 3min30s
- If you want to train from scratch, you don't have to modify anything. If you finish training and want to generate anime picture, modify `mode`, simply run program and wait for your generated anime faces
```shell
python run.py
```
- Of course, you can modify the model architecture or try some other hyper-parameters, do anything you want

#### 3. Check the quality of generated image
- I train for 500 epochs, but I find the effect is pretty good even only train for 100 epochs, so if you want to save time you can stop training after 100 epochs iteration
- Then we will use random Gaussian Noise to sample images. In the DDPM paper, there are two posterior variance, so here we also test these two settings

- First, we set $\sigma_{t}^2 = \beta_{t}$, below are 256 examples

![sample anime faces](gen/sample1.png)

- Second, we set $\sigma_{t}^2 = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}} \beta_{t}$, below are 256 examples

![sample anime faces](gen/sample2.png)

- Let's check the diffusion process, here we show first six diffusion process using first variance setting
![sample anime faces](gen/process1.png)

- I think the quality is good cause the total parameters of our generator is only 6.3M. The architecture used here is not exactly the same as the original DCGAN which is larger, you can try that architecture
- Compare to [VAE](../VAE_ANIME), GAN produces sharper lines while VAE produces slightly blurred images, this is a advantage of GAN

#### 4. Some references
- [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585.pdf)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
- [Diffusion Models Tutorial(English Blog)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#forward-diffusion-process)
- [Diffusion Models Tutorial(Chinese Blog)](https://zhuanlan.zhihu.com/p/525106459)
- [Diffusion Models Tutorail(Chinese Video)](https://www.bilibili.com/video/BV1b541197HX)
- [Diffusion Models implementation from scratch in PyTorch(English Videl)](https://www.youtube.com/watch?v=a4Yfz2FxXiY)
- [Unofficial PyTorch implementation of Denoising Diffusion Probabilistic Models](https://github.com/w86763777/pytorch-ddpm)