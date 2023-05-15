## Denoising Diffusion Implicit Model for Generating Anime face

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
- For this task the code is nearly the same to [DDPM_ANIME](../DDPM_ANIME), we just modify the `model.py` and `run.py` to support DDIM generation
- You don't have to train it again, just use the checkpoints in your [DDPM_ANIME](../DDPM_ANIME)
- Here I would like to run the program by shell, this will make sure that every time we run the program, random noise is the same
```shell
sh sample.sh
```

#### 3. Check the quality of generated image
- First, let's see the quality of generated image after 10, 100, 1000 steps sampling respectively

<div align=center>DDIM sample_steps=10</div>

![sample anime faces step 10](gen/sample_steps=10_eta=0.0.png)

<div align=center>DDIM sample_steps=100</div>

![sample anime faces step 100](gen/sample_steps=100_eta=0.0.png)

<div align=center>DDIM sample_steps=1000</div>

![sample anime faces step 1000](gen/sample_steps=1000_eta=0.0.png)

- When we set $eta$ to other value, the results are as follows:

<div align=center>sample_steps=100, eta=0.2</div>

![sample anime faces step 100](gen/sample_steps=100_eta=0.2.png)

<div align=center>sample_steps=100, eta=0.5</div>

![sample anime faces step 100](gen/sample_steps=100_eta=0.5.png)

<div align=center>DDPM sample_steps=100, eta=1.0</div>

![sample anime faces step 1000](gen/sample_steps=100_eta=1.0.png)

- I also do another experiment, I add noise to the original image(forward process), then use the noisy image to generate image to see whether it can recover the original image. For the forward process and backward process I set t equals to 100, below are the results(first column is original image, second column is noisy image which we add t steps' noise to original image, third column is generated image using DDIM)

![recover t=100](gen/recover_t=100.png)

- We can see generated images' quality are good using DDIM even for 10 steps, which is much faster than DDPM. And DDIM's generative processes are deterministic so the image's high-level feature remains the same while DDPMs are not. But for recovering image, DDIM is not able to deal with that but I think the result is better than DDPM
- For more details, you can clone the project to local and run by yourself

#### 4. Some references
- [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585.pdf)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
- [Denoising Diffusion Implicit Models](https://arxiv.org/pdf/2010.02502.pdf)
- [DDIM Tutorail(Chinese Video)](https://www.bilibili.com/video/BV1JY4y1N7dn)
- [OpenAI implementation of Improved Diffusion](https://github.com/openai/improved-diffusion/tree/main)
