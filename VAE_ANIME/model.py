"""
  * FileName: model.py
  * Author:   Slatter
  * Date:     2023/3/16 23:47
  * Description:  this VAE is different from MinistVAE, here we will use convolution
"""
import pytorch_lightning as pl
import torch
from torch import nn, optim


class AnimeVAE(pl.LightningModule):
    def __init__(self, channel_in=3, c_hid=64, lr=1e-3):
        super(AnimeVAE, self).__init__()
        self.encoder = nn.Sequential(
            # (channel_in, 64, 64) -> (c_hid, 32, 32)
            nn.BatchNorm2d(channel_in),
            nn.ReLU(),
            nn.Conv2d(channel_in, c_hid, kernel_size=3, stride=2, padding=1, bias=False),

            # (c_hid, 32, 32) -> (c_hid * 2, 16, 16)
            nn.BatchNorm2d(c_hid),
            nn.ReLU(),
            nn.Conv2d(c_hid, c_hid * 2, kernel_size=3, stride=2, padding=1, bias=False),

            # (c_hid * 2, 16, 16) -> (c_hid * 4, 8, 8)
            nn.BatchNorm2d(c_hid * 2),
            nn.ReLU(),
            nn.Conv2d(c_hid * 2, c_hid * 4, kernel_size=3, stride=2, padding=1, bias=False),

            # (c_hid * 4, 8, 8) -> (c_hid * 8, 4, 4)
            nn.BatchNorm2d(c_hid * 4),
            nn.ReLU(),
            nn.Conv2d(c_hid * 4, c_hid * 8, kernel_size=3, stride=2, padding=1, bias=False)
        )

        self.mean = nn.Sequential(
            # (c_hid * 8, 4, 4) -> (c_hid * 16, 2, 2)
            nn.Conv2d(c_hid * 8, c_hid * 16, kernel_size=3, stride=2, padding=1, bias=False)
        )

        self.var = nn.Sequential(
            # (c_hid * 8, 4, 4) -> (c_hid * 16, 2, 2)
            nn.Conv2d(c_hid * 8, c_hid * 16, kernel_size=3, stride=2, padding=1, bias=False)
        )

        self.latent = (c_hid * 16, 2, 2)

        self.decoder = nn.Sequential(
            # (c_hid * 16, 2, 2) -> (c_hid * 8, 4, 4)
            nn.BatchNorm2d(c_hid * 16),
            nn.ReLU(),
            nn.ConvTranspose2d(c_hid * 16, c_hid * 8, kernel_size=4, stride=2, padding=1, bias=False),

            # (c_hid * 8, 4, 4) -> (c_hid * 4, 8, 8)
            nn.BatchNorm2d(c_hid * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(c_hid * 8, c_hid * 4, kernel_size=4, stride=2, padding=1, bias=False),

            # (c_hid * 4, 8, 8) -> (c_hid * 2, 16, 16)
            nn.BatchNorm2d(c_hid * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(c_hid * 4, c_hid * 2, kernel_size=4, stride=2, padding=1, bias=False),

            # (c_hid * 2, 16, 16) -> (c_hid, 32, 32)
            nn.BatchNorm2d(c_hid * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(c_hid * 2, c_hid, kernel_size=4, stride=2, padding=1, bias=False),

            # (c_hid, 32, 32) -> (channel_in, 64, 64)
            nn.BatchNorm2d(c_hid),
            nn.ConvTranspose2d(c_hid, channel_in, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid()
        )

        self.criterion1 = nn.BCELoss(reduction='sum')
        self.criterion2 = nn.KLDivLoss(reduction='sum')

        self.lr = lr

    def encode(self, x):
        """
        encoding, return mean and variance
        :param x: (batch, 3, 64, 64)
        :return mean: (batch, 128, 8, 8)  log_var: (batch, 128, 8, 8)
        """
        enc_out = self.encoder(x)  # (batch, 64, 16, 16)
        return self.mean(enc_out), self.var(enc_out)

    def reparameterize(self, mu, log_var):
        """
        :param mu: (batch, 128, 8, 8)
        :param log_var: (batch, 128, 8, 8)
        :return: latent vector: (batch, 128, 8, 8)
        """
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        decoding, return generated image
        :param z: (batch, 128, 8, 8)
        :return out: (batch, 3, 64, 64)
        """
        out = self.decoder(z)
        return out

    def forward(self, x):
        """
        :param x: (batch, image_size)
        :return: generated x: (batch, image_size)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, _ = batch

        x_reconst, mu, log_var = self.forward(x)
        reconst_loss = self.criterion1(x_reconst, x)
        # kl_loss = self.criterion2()
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        train_loss = reconst_loss + kl_div
        self.log('reconstruct loss', reconst_loss)
        self.log('kl divergence', kl_div)
        self.log('train loss', train_loss)
        return {'reconstruct': reconst_loss, 'kl': kl_div, 'loss': train_loss}

    def training_epoch_end(self, step_output):
        reconst, kl, loss = [], [], []
        for x in step_output:
            reconst.append(x['reconstruct'])
            kl.append(x['kl'])
            loss.append(x['loss'])
        reconst_loss = sum(reconst) / len(step_output)
        kl_loss = sum(kl) / len(step_output)
        avg_loss = sum(loss) / len(step_output)
        print('Reconstruct loss: {:.2f}, KL divergence: {:.2f}, loss: {:.2f}'.format(reconst_loss, kl_loss, avg_loss))
