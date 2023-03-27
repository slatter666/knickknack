"""
  * FileName: model.py
  * Author:   Slatter
  * Date:     2023/3/27 12:36
  * Description:  
"""
import torch
from torch import nn
import pytorch_lightning as pl
from torchvision.utils import make_grid


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AnimeGenerator(nn.Module):
    def __init__(self, in_planes, out_planes, depth):
        super(AnimeGenerator, self).__init__()
        self.net = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=in_planes, out_channels=depth * 8, kernel_size=4, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(depth * 8),
            nn.ReLU(inplace=True),

            # (depth * 8, 4, 4)
            nn.ConvTranspose2d(in_channels=depth * 8, out_channels=depth * 4, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(depth * 4),
            nn.ReLU(inplace=True),

            # (depth * 4, 8, 8)
            nn.ConvTranspose2d(in_channels=depth * 4, out_channels=depth * 2, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(depth * 2),
            nn.ReLU(inplace=True),

            # (depth * 2, 16, 16)
            nn.ConvTranspose2d(in_channels=depth * 2, out_channels=depth, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(depth),
            nn.ReLU(inplace=True),

            # (depth, 32, 32)
            nn.ConvTranspose2d(in_channels=depth, out_channels=out_planes, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.Tanh()
            # (out_planes, 64, 64)
        )

        self.net.apply(weights_init)

    def forward(self, z):
        """
        :param z: (batch, in_planes, 1, 1)
        :return: (batch, out_planes, 64, 64)
        """
        return self.net(z)


class AnimeDiscriminator(nn.Module):
    def __init__(self, in_planes, depth):
        super(AnimeDiscriminator, self).__init__()
        self.net = nn.Sequential(
            # input size  (3, 64, 64)
            nn.Conv2d(in_channels=in_planes, out_channels=depth, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (depth, 32, 32)
            nn.Conv2d(in_channels=depth, out_channels=depth * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(depth * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (depth * 2, 16, 16)
            nn.Conv2d(in_channels=depth * 2, out_channels=depth * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(depth * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (depth * 4, 8, 8)
            nn.Conv2d(in_channels=depth * 4, out_channels=depth * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(depth * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (depth * 8, 4, 4)
            nn.Conv2d(in_channels=depth * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.net.apply(weights_init)

    def forward(self, x):
        """
        :param x: (batch, 3, 64, 64)
        :return: (batch)
        """
        return self.net(x).view(-1)


class AnimeGAN(pl.LightningModule):
    def __init__(self, image_size, latent_size, depth, lr):
        super(AnimeGAN, self).__init__()
        self.save_hyperparameters()
        self.image_size = image_size
        self.latent_size = latent_size
        self.depth = depth

        self.generator = AnimeGenerator(latent_size[0], image_size[0], depth)
        self.discriminator = AnimeDiscriminator(image_size[0], depth)

        self.lr = lr
        self.criterion = nn.BCELoss()

        self.check_every = 20  # check the quality of generated image every 20 epochs
        self.test_to_device = False
        self.test_noise = torch.randn(256, *latent_size)

    def forward(self, z):
        """
        :param z: (batch, gen_in, 1, 1)
        :return: (batch, gen_out, 64, 64)
        """
        return self.generator(z)

    def configure_optimizers(self):
        gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        dis_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return gen_opt, dis_opt

    def generator_step(self, x):
        """
        Training step for generator
        1. sample random noise to generate images
        2. classify generated images using the discriminator
        3. compute loss

        :param x: (batch_size, 3, 64, 64)
        """
        batch_size = x.size(0)
        real_label = torch.ones(batch_size).type_as(x)

        # sample noise
        z = torch.randn(batch_size, *self.latent_size).type_as(x)

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

        :param x: (batch_size, 3, 64, 64)
        """
        batch_size = x.size(0)
        real_label = torch.ones(batch_size).type_as(x)
        fake_label = torch.zeros(batch_size).type_as(x)

        # real images
        d_output_real = self.discriminator(x)
        loss_real = self.criterion(d_output_real, real_label)

        # fake images
        z = torch.randn(batch_size, *self.latent_size).type_as(x)
        fake_imgs = self(z).detach()
        d_output_fake = torch.squeeze(self.discriminator(fake_imgs))
        loss_fake = self.criterion(d_output_fake, fake_label)

        # compute total loss
        d_loss = loss_real + loss_fake
        return d_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch

        if not self.test_to_device:
            self.test_noise = self.test_noise.type_as(x)
            self.test_to_device = True

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
