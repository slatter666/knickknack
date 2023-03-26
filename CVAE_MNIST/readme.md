## Conditional Variational Auto Encoder for Generating Handwriting Numbers

#### 1. Introduction
- In the previous [VAE_MINIST](../VAE_MNIST) we can generate handwriting numbers randomly, for this project we wanna generate numbers conditionally(for example we wanna generate 7, then generated pictures should contain number 7). We will train a CVAE to do this
- The dataset is MNIST, it will be download under the folder `dataset` using torchvision, the dataset folder structure looks like this:
```text
dataset
├── mnist
│   └── MNIST
│   │   └── raw
│   │       ├── t10k-images-idx3-ubyte
│   │       ├── t10k-images-idx3-ubyte.gz
│   │       ├── t10k-labels-idx1-ubyte
│   │       ├── t10k-labels-idx1-ubyte.gz
│   │       ├── train-images-idx3-ubyte
│   │       ├── train-images-idx3-ubyte.gz
│   │       ├── train-labels-idx1-ubyte
│   │       └── train-labels-idx1-ubyte.gz
```

#### 2. Load dataset, Build model, Train model
- We don't have to modify too much, only part of the model and input should be modified, watch the code to check this
- Here I use a NVIDIA GeForce RTX 3090 to train, each epoch will cost about 3 seconds
- If you want to train from scratch, you don't have to modify any thing. If you finish training and want to generate number picture, modify `mode`, simply run program and wait for your generated numbers
```shell
python run.py
```
- Of course, you can modify the model architecture or try some other hyper parameters, do anything you want

#### 3. Check the quality of generated image
- Note that I also try to add label information only in decoder, but the effect is not good as adding to both encoder and decoder, I guess it's just because supervision is stronger
- First of all, we will use random Gaussian Noise and label information(for line i, we put label i, i from 0 to 9) to sample some images, here are 200 examples

![sample anime faces](gen/sample.png)

- Then we can see the reconstruct numbers

![](gen/reconstruct.png)

- I think the quality is good because we just use such a simple model, and it does generate image according to our label

#### 4. Some references
- [Tutorial on Variational Autoencoders(English)](https://arxiv.org/pdf/1606.05908.pdf)
- [Tutorial on Variational Autoencoders(Chinese)](https://zhuanlan.zhihu.com/p/348498294)
- [During training, it's reasonable that reconstruct loss decrease, but why KL divergence increase](https://www.cnblogs.com/BlueBlueSea/p/13149464.html)
- [CVAE(more specific)](https://zhuanlan.zhihu.com/p/611498730)