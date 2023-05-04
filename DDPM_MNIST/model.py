"""
  * FileName: model.py
  * Author:   Slatter
  * Date:     2023/5/3 18:21
  * Description:
"""
import math
from typing import List
from matplotlib import pyplot as plt

import pytorch_lightning as pl
from torch import nn, optim
from torchvision.utils import make_grid

from utils import *


class PositionalEncoding(nn.Module):
    def __init__(self, num_steps: int, embed_dim: int, dropout: float = 0):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        half_dim = embed_dim // 2
        den = torch.exp(- torch.arange(half_dim) * math.log(10000) / (half_dim - 1))  # (embed_dim)
        pos = torch.arange(0, num_steps).view(num_steps, 1)  # (num_steps, 1)
        pos_embedding = torch.zeros((num_steps, half_dim * 2))  # (num_steps, embed_dim * 2)
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, t: torch.Tensor):
        """
        :param t: t moment  (batch)
        :return: positional embedding according to t moment   (batch, embed_size)
        """
        return self.pos_embedding[t]


class Block(nn.Module):
    def __init__(self, channel_in, channel_out, time_embed_dim, up=False):
        super(Block, self).__init__()
        self.time_mlp = nn.Linear(time_embed_dim, channel_out)
        self.up = up

        if up:  # up sample
            # residual connection
            self.conv1 = nn.Conv2d(2 * channel_in, channel_out, kernel_size=3, stride=1, padding=1)
            self.transform = nn.ConvTranspose2d(channel_out, channel_out, kernel_size=4, stride=2, padding=1)
        else:  # down sample
            self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=1, padding=1)
            self.transform = nn.Conv2d(channel_out, channel_out, kernel_size=4, stride=2, padding=1)

        self.conv2 = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1)
        self.bnorm1 = nn.BatchNorm2d(channel_out)
        self.bnorm2 = nn.BatchNorm2d(channel_out)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        """
        one up sample or down sample process
        :param x: (batch, channel_in, m, m)
        :param t: (batch, time_embed_dim)
        :return:
            h2: (batch, channel_in, m, m) (up sample will be used for residual connection)
            out: (batch, channel_out, m / 2, m / 2) for down sample, (batch, channel_out, m * 2, m * 2) for up sample
        """
        # first convolution
        h1 = self.bnorm1(self.relu(self.conv1(x)))  # (batch, channel_out, m, m)
        # time embedding
        time_embed = self.relu(self.time_mlp(t))  # (batch, channel_out)
        time_embed = time_embed.unsqueeze(dim=-1).unsqueeze(dim=-1)  # (batch, channel_out, 1, 1)

        # add time embed
        h1 = time_embed + h1
        # second convolution
        h2 = self.bnorm2(self.relu(self.conv2(h1)))  # (batch, channel_out, m, m)

        # third convolution   up sample or down sample
        out = self.transform(h2)

        return h2, out


class SimpleUnet(nn.Module):
    def __init__(self, down_channels: List[int], up_channels: List[int], time_embed_dim: int, num_steps: int):
        super(SimpleUnet, self).__init__()
        self.down_channels = down_channels
        self.up_channels = up_channels
        self.time_embed_dim = time_embed_dim

        self.position_embedding = PositionalEncoding(num_steps, time_embed_dim, dropout=0.1)

        # down sample
        self.downs = nn.ModuleList(
            [Block(down_channels[i], down_channels[i + 1], time_embed_dim) for i in range(len(down_channels) - 1)])

        # bottom layer
        self.bottom = nn.Sequential(
            nn.Conv2d(down_channels[-1], down_channels[-1] * 2, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(down_channels[-1] * 2),

            nn.Conv2d(down_channels[-1] * 2, down_channels[-1] * 2, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(down_channels[-1] * 2),

            nn.ConvTranspose2d(down_channels[-1] * 2, down_channels[-1], 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(down_channels[-1])
        )

        # up sample
        self.ups = nn.ModuleList(
            [Block(up_channels[i], up_channels[i + 1], time_embed_dim, up=True) for i in range(len(up_channels) - 1)])

    def forward(self, x, t):
        """
        :param x: (batch, channel_in, image_size, image_size)
        :param time_embed: (batch)
        :return: (batch, channel_in, image_size, image_size)
        """
        # embed time
        time_embed = self.position_embedding(t)  # (batch, embed_dim)

        # down sample
        residual_inputs = []
        for down in self.downs:
            h, x = down(x, time_embed)
            residual_inputs.append(h)

        # bottom layer
        x = self.bottom(x)

        # up sample
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # add residual x as additional channels
            x = torch.cat([x, residual_x], dim=1)
            h, x = up(x, time_embed)

        out = h
        return out


class DiffusionModel(pl.LightningModule):
    def __init__(self, image_size, down_channels: List[int], up_channels: List[int], time_embed_dim: int, num_steps,
                 lr=1e-4):
        super(DiffusionModel, self).__init__()
        self.save_hyperparameters()

        self.image_size = image_size
        self.time_embed_dim = time_embed_dim
        self.num_steps = num_steps

        self.unet = SimpleUnet(down_channels, up_channels, time_embed_dim, num_steps)

        self.lr = lr
        self.criterion = nn.MSELoss()

        # here are some constant parameters
        self.betas = torch.linspace(1e-4, 2e-2, num_steps)  # beta_{t}
        self.alphas = 1 - self.betas  # alpha_{t}
        self.alphas_sqrt = torch.sqrt(self.alphas)  # sqrt alpha_{t}

        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)  # alpha_{t}_cumprod
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.]), self.alphas_cumprod[:-1]],
                                             dim=0)  # alpha_{t-1}_cumprod
        self.alphas_cumprod_sqrt = torch.sqrt(self.alphas_cumprod)  # sqrt alpha_{t}_cumprod
        self.one_minus_alphas_cumprod_sqrt = torch.sqrt(1 - self.alphas_cumprod)  # sqrt 1 - alpha_{t}_cumprod

        # used for test
        self.check_every = 20  # check the quality of generated image every 20 epochs
        self.test_to_device = False
        self.test_noise = torch.randn(256, *self.image_size)

    def move_to_device(self, x):
        """
        move all tensors to device
        """
        self.betas = self.betas.type_as(x)
        self.alphas = self.alphas.type_as(x)
        self.alphas_sqrt = self.alphas_sqrt.type_as(x)

        self.alphas_cumprod = self.alphas_cumprod.type_as(x)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.type_as(x)
        self.alphas_cumprod_sqrt = self.alphas_cumprod_sqrt.type_as(x)
        self.one_minus_alphas_cumprod_sqrt = self.one_minus_alphas_cumprod_sqrt.type_as(x)

        self.test_noise = self.test_noise.type_as(x)
        self.test_to_device = True

    def forward_process(self, x_0, t):
        """
        forward process
        :param x_0: (batch, image_size)
        :param t: t moment (batch)
        :return x_t: (batch, image_size)
        """
        gaussian_noise = torch.randn_like(x_0)  # (batch, 3, 64, 64)

        mean = self.alphas_cumprod_sqrt[t]  # (batch)
        mean = mean[:, None, None, None]  # (batch, 1, 1, 1)
        std = self.one_minus_alphas_cumprod_sqrt[t]  # (batch)
        std = std[:, None, None, None]  # (batch, 1, 1, 1)
        x_t = mean * x_0 + gaussian_noise * std  # (batch, 3, 64, 64)
        return gaussian_noise, x_t

    def reverse_step(self, x_t, t):
        """
        one reverse process step
        :param x_t: (batch, image_size)
        :param t: t moment  a scalar
        :return x_{t-1}: (batch, image_size)
        """
        t = torch.tensor([t]).type_as(x_t).long()  # (1)

        denoise = self(x_t, t)  # (batch, 3, 64, 64)
        coef = (1 - self.alphas[t]) / self.one_minus_alphas_cumprod_sqrt[t]  # (1)
        mean = (1 / self.alphas_sqrt[t]) * (x_t - coef * denoise)  # (batch, 3, 64, 64)

        z = torch.randn_like(x_t)  # (batch, 3, 64, 64)
        sigma_t = self.betas[t].sqrt()  # (1)
        std = sigma_t * z  # (batch, 3, 64, 64)
        next_step = mean + std  # (batch, 3, 64, 64)
        return next_step

    def reverse_process(self, x_t, show_process=False, img_nums=1, process_nums=10, save_path=None):
        """
        full reverse process step
        :param x_t: random guassian noise   (batch, image_size):
        :param show_process: show process or not
        :param show_nums: how many images to show in the diffusion process
        :param process_nums: how many process to show in the diffusion process
        :param save_path: path to save diffusion process visualization
        :return: x_0 (batch, image_size)
        """
        if not self.test_to_device:
            self.move_to_device(x_t)

        step_size = math.ceil(self.num_steps / process_nums)

        self.eval()
        with torch.no_grad():
            cur_x = x_t
            for i in range(self.num_steps - 1, -1, -1):
                cur_x = self.reverse_step(cur_x, i)

                if show_process and i % step_size == 0:
                    col = process_nums - i // step_size
                    images = cur_x[:img_nums].detach().cpu()  # (img_nums, h, w, c)
                    for i in range(images.shape[0]):
                        pos = i * process_nums + col
                        plt.subplot(img_nums, process_nums, pos)
                        plt.imshow(tensor_image_to_PIL(images[i]), cmap='gray')

        if show_process:
            plt.savefig(save_path)
        return cur_x

    def compute_loss(self, x_0):
        """
        :param x_0: (batch, image_size)
        :return: loss
        """
        bsz = x_0.size(0)
        t = torch.randint(low=0, high=self.num_steps, size=(bsz,)).type_as(x_0).long()  # (batch)
        noise, x_t = self.forward_process(x_0, t)  # forward process  (batch, image_size)

        output = self(x_t, t)
        loss = self.criterion(noise, output)
        return loss

    def forward(self, x, t):
        """
        :param x: (batch, image_size)
        :param t: t moment  (batch)
        :return: generated noise (batch, image_size)
        """
        return self.unet(x, t)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, _ = batch

        if not self.test_to_device:  # move to device
            self.move_to_device(x)

        loss = self.compute_loss(x_0=x)
        self.log('loss', loss)
        return loss

    def training_epoch_end(self, step_output):
        loss = []
        for x in step_output:
            loss.append(x['loss'])
        avg_loss = sum(loss) / len(step_output)
        print('Loss: {:.2f}'.format(avg_loss))

        if self.current_epoch % self.check_every == 0:
            # log sampled images, it's a sequence of images, we use x_0
            test_imgs = convert_image_to_natural(self.reverse_process(self.test_noise))
            grid = make_grid(test_imgs, nrow=16)
            self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
