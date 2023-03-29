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


class AnimeEncoder(nn.Module):
    def __init__(self, in_planes, out_planes, depth):
        super(AnimeEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # (channel_in, 64, 64) -> (depth, 32, 32)
            nn.BatchNorm2d(in_planes),
            nn.ReLU(),
            nn.Conv2d(in_planes, depth, kernel_size=3, stride=2, padding=1, bias=False),

            # (depth, 32, 32) -> (depth * 2, 16, 16)
            nn.BatchNorm2d(depth),
            nn.ReLU(),
            nn.Conv2d(depth, depth * 2, kernel_size=3, stride=2, padding=1, bias=False),

            # (c_hid * 2, 16, 16) -> (depth * 4, 8, 8)
            nn.BatchNorm2d(depth * 2),
            nn.ReLU(),
            nn.Conv2d(depth * 2, depth * 4, kernel_size=3, stride=2, padding=1, bias=False),

            # (depth * 4, 8, 8) -> (depth * 8, 4, 4)
            nn.BatchNorm2d(depth * 4),
            nn.ReLU(),
            nn.Conv2d(depth * 4, depth * 8, kernel_size=3, stride=2, padding=1, bias=False),

            # (depth * 8, 4, 4) -> (depth * 16, 2, 2)
            nn.BatchNorm2d(depth * 8),
            nn.ReLU(),
            nn.Conv2d(depth * 8, depth * 16, kernel_size=3, stride=2, padding=1, bias=False)
        )

        self.mean = nn.Sequential(
            # (depth * 16, 2, 2) -> (out_planes, 1, 1)
            nn.Conv2d(depth * 16, out_planes, kernel_size=2, stride=1, padding=0, bias=False)
        )

        self.var = nn.Sequential(
            # (depth * 16, 2, 2) -> (out_planes, 1, 1)
            nn.Conv2d(depth * 16, out_planes, kernel_size=2, stride=1, padding=0, bias=False)
        )

    def encode(self, x):
        """
        encoding, return mean and variance
        :param x: (batch, in_planes, 64, 64)
        :return mean: (batch, c_hid * 16, 2, 2)  log_var: (batch, c_hid * 16, 2, 2)
        """
        enc_out = self.encoder(x)  # (batch, c_hid * 8, 4, 4)
        return self.mean(enc_out), self.var(enc_out)

    def reparameterize(self, mu, log_var):
        """
        :param mu: (batch, out_planes, 1, 1)
        :param log_var: (batch, out_planes, 1, 1)
        :return: latent vector: (batch, out_planes, 1, 1)
        """
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        :param x: (batch, in_planes, 64, 64)
        :return: enc_z, mu, log_var
        """
        enc_out = self.encoder(x)  # (batch, depth *16, 2, 2)
        mu, log_var = self.mean(enc_out), self.var(enc_out)  # (batch, out_planes, 1, 1)
        enc_z = self.reparameterize(mu, log_var)  # (batch, out_planes, 1, 1)
        return enc_z, mu, log_var


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
        )
        self.last_layer = nn.Sequential(
            # (depth * 8, 4, 4)
            nn.Conv2d(in_channels=depth * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.net.apply(weights_init)
        self.last_layer.apply(weights_init)

    def forward(self, x):
        """
        :param x: (batch, 3, 64, 64)
        :return: last_hidden, out
        """
        last_hidden = self.net(x)  # (batch, depth * 8, 4, 4) this will be used to compute reconstruction loss
        out = self.last_layer(last_hidden).view(-1)  # (batch)
        return last_hidden, out


class AnimeGAN(pl.LightningModule):
    def __init__(self, image_size, latent_size, depth, lr):
        super(AnimeGAN, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.image_size = image_size
        self.latent_size = latent_size
        self.depth = depth

        self.encoder = AnimeEncoder(image_size[0], latent_size[0], depth)
        self.generator = AnimeGenerator(latent_size[0], image_size[0], depth)
        self.discriminator = AnimeDiscriminator(image_size[0], depth)

        self.lr = lr
        self.criterion1 = nn.BCELoss()
        self.criterion2 = nn.MSELoss()

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
        enc_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        dis_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return enc_opt, gen_opt, dis_opt

    def update_step(self, x, enc_opt, gen_opt, dis_opt):
        """
        :param x: (batch_size, 3, 64, 64)
        """
        batch_size = x.size(0)
        real_label = torch.ones(batch_size).type_as(x)
        fake_label = torch.zeros(batch_size).type_as(x)

        """
        First we update discriminator
        1. predict probabilities of real images
        2. encode the input image to latent representation, then use generator to reconstruct image, predict probabilities of fake images
        3. sample noise then use generator to generate image, then predict probabilities of fake images
        4. compute loss for discriminator
        """
        _, prob_real = self.discriminator(x)

        enc_z, mu, log_var = self.encoder(x)
        x_hat = self.generator(enc_z)
        _, prob_xhat = self.discriminator(x_hat)

        zp = torch.randn(batch_size, *self.latent_size).type_as(x)
        xp = self.generator(zp)
        _, prob_xp = self.discriminator(xp)

        loss_dis = self.criterion1(prob_real, real_label) + self.criterion1(prob_xhat, fake_label) + self.criterion1(
            prob_xp, fake_label)
        dis_opt.zero_grad()
        loss_dis.backward()
        dis_opt.step()

        """
        Second we update encoder and generator
        1. encode input image to latent representation(z), then use generator to reconstruct image(x_hat)
        2. use discriminator to get hidden representation of x and x_hat
        3. sample noise then use generator to generate image, then predict probabilities of fake images
        4. compute loss_prior, loss_reconst, loss_gan
        """
        enc_z, mu, log_var = self.encoder(x)
        x_hat = self.generator(enc_z)

        dis_x, out_x = self.discriminator(x)
        dis_xhat, out_xhat = self.discriminator(x_hat)

        zp = torch.randn(batch_size, *self.latent_size).type_as(x)
        xp = self.generator(zp)
        dis_xp, out_xp = self.discriminator(xp)

        loss_prior = (- 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())) / batch_size
        loss_reconst = self.criterion2(dis_xhat, dis_x)
        loss_gan = self.criterion1(out_xp, real_label)

        enc_opt.zero_grad()
        gen_opt.zero_grad()
        # here I set gamma=0.1, when back propagate loss_enc and loss_dec, loss_enc =  loss_prior + loss_reconst
        loss_enc = loss_prior + 0.9 * loss_reconst
        loss_dec = 0.1 * loss_reconst + loss_gan
        loss_enc.backward(retain_graph=True)
        loss_dec.backward()
        enc_opt.step()
        gen_opt.step()

        loss_dict = {'loss_prior': loss_prior, 'loss_reconst': loss_reconst, 'loss_dis': loss_dis, 'loss_dec': loss_dec}
        self.log('loss', loss_dict)

    def training_step(self, batch, batch_idx):
        x, _ = batch

        if not self.test_to_device:
            self.test_noise = self.test_noise.type_as(x)
            self.test_to_device = True

        enc_opt, gen_opt, dis_opt = self.optimizers()
        self.update_step(x, enc_opt, gen_opt, dis_opt)

    def training_epoch_end(self, outputs):
        if self.current_epoch % self.check_every == 0:
            # log sampled images
            test_imgs = self(self.test_noise)
            test_imgs = test_imgs.view(test_imgs.size(0), *self.image_size)
            grid = make_grid(test_imgs, nrow=16)
            self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
