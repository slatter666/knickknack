"""
  * FileName: model.py
  * Author:   Slatter
  * Date:     2023/3/14 21:56
  * Description:  
"""
import pytorch_lightning as pl
import torch
from torch import nn, optim


class VAE(pl.LightningModule):
    def __init__(self, image_size=784, h_dim=400, z_dim=20, lr=1e-3):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(image_size, h_dim),
            nn.ReLU()
        )

        self.mean = nn.Linear(h_dim, z_dim)  # mean
        self.var = nn.Linear(h_dim, z_dim)  # variance

        self.latent = [z_dim]

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),

            nn.Linear(h_dim, image_size),
            nn.Sigmoid()
        )

        self.lr = lr
        self.criterion1 = nn.BCELoss(reduction='sum')

    def encode(self, x):
        """
        encoding, return mean and variance
        :param x: (batch, image_size)
        :return mean: (batch, z_dim)  log_var: (batch, z_dim)
        """
        enc_out = self.encoder(x)
        return self.mean(enc_out), self.var(enc_out)

    def reparameterize(self, mu, log_var):
        """
        :param mu: (batch, z_dim)
        :param log_var: (batch, z_dim)
        :return: latent vector: (batch, z_dim)
        """
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        decoding, return generated image
        :param z: (batch, z_dim)
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
        inputs, labels = batch
        x = inputs.view(-1, 784)

        x_reconst, mu, log_var = self.forward(x)
        reconst_loss = self.criterion1(x_reconst, x)
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
