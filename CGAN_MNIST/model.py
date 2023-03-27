"""
  * FileName: model.py
  * Author:   Slatter
  * Date:     2023/3/27 17:52
  * Description:  
"""

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import make_grid


class MNISTGenerator(nn.Module):
    def __init__(self, image_size, latent_dim, h_dim, num_classes):
        super(MNISTGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, h_dim),
            nn.LeakyReLU(),

            nn.Linear(h_dim, h_dim * 2),
            nn.LeakyReLU(),

            nn.Linear(h_dim * 2, h_dim * 4),
            nn.LeakyReLU(),

            nn.Linear(h_dim * 4, np.prod(image_size)),
            nn.Tanh()
        )

    def forward(self, z, y):
        """
        :param z: (batch, latent_dim)
        :param y: (batch, num_classes)
        :return: (batch, 784)
        """
        input_x = torch.concat([z, y], dim=1)
        out = self.model(input_x)
        return out


class MNISTDiscriminator(nn.Module):
    def __init__(self, image_size, h_dim, num_classes):
        super(MNISTDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(image_size) + num_classes, h_dim * 4),
            nn.LeakyReLU(),

            nn.Linear(h_dim * 4, h_dim * 2),
            nn.LeakyReLU(),

            nn.Linear(h_dim * 2, h_dim),
            nn.LeakyReLU(),

            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        """
        :param x: (batch, 1, 28, 28)
        :param y: (batch, num_classes)
        :return: (batch)
        """
        input_x = torch.cat([x, y], dim=1)
        out = self.model(input_x).squeeze(dim=1)
        return out


class MNISTGAN(pl.LightningModule):
    def __init__(self, image_size, latent_dim, h_dim, num_classes, lr):
        super(MNISTGAN, self).__init__()
        self.save_hyperparameters()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.h_dim = h_dim
        self.num_classes = num_classes

        self.generator = MNISTGenerator(image_size, latent_dim, h_dim, num_classes)
        self.discriminator = MNISTDiscriminator(image_size, h_dim, num_classes)

        self.lr = lr
        self.criterion = nn.BCELoss()

        self.check_every = 20  # check the quality of generated image every 20 epochs
        self.test_to_device = False
        self.test_noise = torch.randn(200, latent_dim)
        self.test_label = F.one_hot(torch.matmul(torch.arange(10).view(-1, 1), torch.ones(1, 20, dtype=torch.long)).flatten(), num_classes)

    def forward(self, z, y):
        """
        :param z: (batch_size, latent_dim)
        :param y: (batch_size, num_classes)
        """
        return self.generator(z, y)

    def configure_optimizers(self):
        gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.lr)
        dis_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        return gen_opt, dis_opt

    def generator_step(self, x, y):
        """
        Training step for generator
        1. sample random noise to generate images
        2. classify generated images using the discriminator
        3. compute loss

        :param x: (batch_size, 784)
        :param y: (batch_size, num_classes)
        """
        batch_size = x.size(0)
        real_label = torch.ones(batch_size).type_as(x)

        # sample noise
        z = torch.randn(batch_size, self.latent_dim).type_as(x)

        # generate images and classify them
        fake_imgs = self(z, y)
        d_output = self.discriminator(fake_imgs, y)

        # compute loss
        g_loss = self.criterion(d_output, real_label)
        return g_loss

    def discriminator_step(self, x, y):
        """
        Training step for discriminator
        1. get actual images then predict probabilities of actual images
        2. get fake images from generator then predict probabilities of fake images
        3. compute loss

        :param x: (batch_size, 784)
        :param y: (batch_size, num_classes)
        """
        batch_size = x.size(0)
        real_label = torch.ones(batch_size).type_as(x)
        fake_label = torch.zeros(batch_size).type_as(x)

        # real images
        d_output = self.discriminator(x, y)
        loss_real = self.criterion(d_output, real_label)

        # fake images
        z = torch.randn(batch_size, self.latent_dim).type_as(x)
        fake_imgs = self(z, y).detach()
        d_output = torch.squeeze(self.discriminator(fake_imgs, y))
        loss_fake = self.criterion(d_output, fake_label)

        # compute total loss
        d_loss = loss_real + loss_fake
        return d_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, labels = batch
        x = x.view(x.size(0), -1)
        y = F.one_hot(labels, num_classes=self.num_classes)

        if not self.test_to_device:
            self.test_noise = self.test_noise.type_as(x)
            self.test_label = self.test_label.type_as(x)
            self.test_to_device = True

        # train generator
        if optimizer_idx == 0:
            loss = self.generator_step(x, y)

        # train discriminator
        if optimizer_idx == 1:
            loss = self.discriminator_step(x, y)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        g_loss = outputs[0]['loss']
        d_loss = outputs[1]['loss']
        loss_dict = {'g_loss': g_loss, 'd_loss': d_loss}
        self.log('loss', loss_dict)

    def training_epoch_end(self, outputs):
        if self.current_epoch % self.check_every == 0:
            # log sampled images
            test_imgs = self(self.test_noise, self.test_label)
            test_imgs = test_imgs.view(test_imgs.size(0), *self.image_size)
            grid = make_grid(test_imgs, nrow=int(200 / self.num_classes))
            self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
