"""
  * FileName: model.py
  * Author:   Slatter
  * Date:     2023/3/25 12:12
  * Description:  
"""

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torchvision.utils import make_grid


class MNISTGenerator(nn.Module):
    def __init__(self, image_size, latent_dim, h_dim):
        super(MNISTGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, h_dim),
            nn.LeakyReLU(),

            nn.Linear(h_dim, h_dim * 2),
            nn.LeakyReLU(),

            nn.Linear(h_dim * 2, h_dim * 4),
            nn.LeakyReLU(),

            nn.Linear(h_dim * 4, np.prod(image_size)),
            nn.Tanh()
        )

    def forward(self, z):
        """
        :param z: (batch, latent_dim)
        :return: (batch, 784)
        """
        out = self.model(z)
        return out


class MNISTDiscriminator(nn.Module):
    def __init__(self, image_size, h_dim):
        super(MNISTDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(image_size), h_dim * 4),
            nn.LeakyReLU(),

            nn.Linear(h_dim * 4, h_dim * 2),
            nn.LeakyReLU(),

            nn.Linear(h_dim * 2, h_dim),
            nn.LeakyReLU(),

            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param x: (batch, 1, 28, 28)
        :return: (batch)
        """
        out = self.model(x).squeeze(dim=1)
        return out


class MNISTGAN(pl.LightningModule):
    def __init__(self, image_size, latent_dim, h_dim, lr):
        super(MNISTGAN, self).__init__()
        self.save_hyperparameters()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.h_dim = h_dim

        self.generator = MNISTGenerator(image_size, latent_dim, h_dim)
        self.discriminator = MNISTDiscriminator(image_size, h_dim)

        self.lr = lr
        self.criterion = nn.BCELoss()

        self.check_every = 20  # check the quality of generated image every 20 epochs
        self.test_to_device = False
        self.test_noise = torch.randn((256, latent_dim))

    def forward(self, z):
        """
        Generates an image using the generator
        given input noise z
        """
        return self.generator(z)

    def configure_optimizers(self):
        gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        dis_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        return gen_opt, dis_opt

    def generator_step(self, x):
        """
        Training step for generator
        1. sample random noise to generate images
        2. classify generated images using the discriminator
        3. compute loss

        :param x: (batch_size, 784)
        """
        batch_size = x.size(0)
        real_label = torch.ones(batch_size).type_as(x)

        # sample noise
        z = torch.randn(batch_size, self.latent_dim).type_as(x)

        # generate images and classify them
        fake_imgs = self(z)
        d_output = self.discriminator(fake_imgs)

        # compute loss
        g_loss = self.criterion(d_output, real_label)
        return g_loss

    def discriminator_step(self, x):
        """
        Training step for discriminator
        1. get actual images then predict probabilities of actual images
        2. get fake images from generator then predict probabilities of fake images
        3. compute loss

        :param x: (batch_size, 784)
        """
        batch_size = x.size(0)
        real_label = torch.ones(batch_size).type_as(x)
        fake_label = torch.zeros(batch_size).type_as(x)

        # real images
        d_output = self.discriminator(x)
        loss_real = self.criterion(d_output, real_label)

        # fake images
        z = torch.randn(batch_size, self.latent_dim).type_as(x)
        fake_imgs = self(z).detach()
        d_output = torch.squeeze(self.discriminator(fake_imgs))
        loss_fake = self.criterion(d_output, fake_label)

        # compute total loss
        d_loss = loss_real + loss_fake
        return d_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)

        if not self.test_to_device:
            self.test_noise = self.test_noise.type_as(x)

        # train generator
        if optimizer_idx == 0:
            loss = self.generator_step(x)

        # train discriminator
        if optimizer_idx == 1:
            loss = self.discriminator_step(x)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        g_loss = outputs[0]['loss']
        d_loss = outputs[1]['loss']
        loss_dict = {'g_loss': g_loss, 'd_loss': d_loss}
        self.log('loss', loss_dict)

    def training_epoch_end(self, outputs):
        if self.current_epoch % self.check_every == 0:
            # log sampled images
            test_imgs = self(self.test_noise)
            test_imgs = test_imgs.view(test_imgs.size(0), *self.image_size)
            grid = make_grid(test_imgs, nrow=16)
            self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
