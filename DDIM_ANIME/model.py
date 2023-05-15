"""
  * FileName: model.py
  * Author:   Slatter
  * Date:     2023/5/15 17:00
  * Description:
"""

import pytorch_lightning as pl
from matplotlib import pyplot as plt
from torch import optim
from torchvision.utils import make_grid

from unet import *
from utils import *


class DiffusionModel(pl.LightningModule):
    def __init__(self, image_size, num_steps: int, ch: int, ch_mult: List[int], attn: List[int], num_res_blocks: int,
                 dropout: float, lr=1e-3):
        super(DiffusionModel, self).__init__()
        self.save_hyperparameters()

        self.image_size = image_size
        self.num_steps = num_steps

        self.unet = UNet(num_steps, ch, ch_mult, attn, num_res_blocks, dropout)

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

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (
                    1. - self.alphas_cumprod)  # second posterior variance setting

        # this will be used when we use DDIM generate process
        self.time_step_sequence = None

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
        self.posterior_variance = self.posterior_variance.type_as(x)

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

    def ddpm_reverse_step(self, x_t, t, posterior_choice=0):
        """
        one DDPM reverse process step
        :param x_t: (batch, image_size)
        :param t: t moment  a scalar
        :param posterior_choice: 0 for self.betas and 1 for self.posterior_variance
        :return x_{t-1}: (batch, image_size)
        """
        t = torch.tensor([t]).type_as(x_t).long()  # (1)

        denoise = self(x_t, t)  # (batch, 3, 64, 64)
        coef = (1 - self.alphas[t]) / self.one_minus_alphas_cumprod_sqrt[t]  # (1)
        mean = (1 / self.alphas_sqrt[t]) * (x_t - coef * denoise)  # (batch, 3, 64, 64)

        z = torch.randn_like(x_t)  # (batch, 3, 64, 64)
        if posterior_choice == 0:
            sigma_t = self.betas[t].sqrt()  # (1)
        else:
            sigma_t = self.posterior_variance[t].sqrt()  # (1)

        std = sigma_t * z  # (batch, 3, 64, 64)
        next_step = mean + std  # (batch, 3, 64, 64)
        return next_step

    def ddpm_reverse_process(self, x_t, show_process=False, img_nums=1, process_nums=10, save_path=None,
                             posterior_choice=0):
        """
        full DDPM reverse process step
        :param x_t: random guassian noise   (batch, image_size):
        :param show_process: show process or not
        :param show_nums: how many images to show in the diffusion process
        :param process_nums: how many process to show in the diffusion process
        :param save_path: path to save diffusion process visualization
        :param posterior_choice: 0 for self.betas and 1 for self.posterior_variance
        :return: x_0 (batch, image_size)
        """
        if not self.test_to_device:
            self.move_to_device(x_t)

        step_size = math.ceil(self.num_steps / process_nums)

        self.eval()
        with torch.no_grad():
            cur_x = x_t
            for i in range(self.num_steps - 1, -1, -1):
                cur_x = self.ddpm_reverse_step(cur_x, i, posterior_choice)

                if show_process and i % step_size == 0:
                    col = process_nums - i // step_size
                    images = cur_x[:img_nums].detach().cpu()  # (img_nums, h, w, c)
                    for j in range(images.shape[0]):
                        pos = j * process_nums + col
                        plt.subplot(img_nums, process_nums, pos)
                        plt.imshow(tensor_image_to_PIL(images[j]))
                        plt.axis('off')

        if show_process:
            plt.savefig(save_path)
        return cur_x

    def ddim_reverse_step(self, x_t, t, t_prev, eta):
        """
        one DDIM reverse process step
        :param x_t: (batch, image_size)
        :param t: t moment  a scalar
        :param t_prev: previous t moment a scalar
        :return x_{t-1}: (batch, image_size)
        """
        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_t_prev = self.alphas_cumprod[t_prev]

        # denoise
        t = torch.tensor([t]).type_as(x_t).long()  # (1)
        denoise = self(x_t, t)  # (batch, 3, 64, 64)

        # predicted x0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod_t) * denoise) / torch.sqrt(alpha_cumprod_t)  # (3, 64, 64)

        # compute posterior variance
        sigma_t = eta * torch.sqrt(
            (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))  # (1)

        # direction pointing to xt
        pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t ** 2) * denoise  # (3, 64, 64)

        # random gaussian noise
        z = torch.randn_like(x_t)  # (batch, 3, 64, 64)

        next_step = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigma_t * z  # (batch, 3, 64, 64)
        return next_step

    def ddim_reverse_process(self, x_t, time_steps=100, eta=0.0, discr_method='linear', show_process=False, img_nums=1,
                             process_nums=10, save_path=None):
        """
        full DDIM reverse process step
        :param x_t: random guassian noise   (batch, image_size)
        :param time_steps: reverse steps
        :param eta: a hyper parameter in DDIM, it's original DDPM generative process when eta=1 and DDIM when eta=0
        :param discr_method: reverse process sub-sequence selection: linear or quadratic
        :param show_process: show process or not
        :param show_nums: how many images to show in the diffusion process
        :param process_nums: how many process to show in the diffusion process
        :param save_path: path to save diffusion process visualization
        :return: x_0 (batch, image_size)
        """
        if discr_method == 'linear':
            time_step_seq = torch.linspace(0, self.num_steps - 2, time_steps, dtype=torch.long)
        elif discr_method == 'quadratic':
            time_step_seq = (torch.linspace(0, int(math.sqrt(self.num_steps - 2)), time_steps) ** 2).long()
        else:
            raise NotImplementedError(f"There is no DDIM discretization method called '{discr_method}'")

        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        time_step_seq = time_step_seq + 1
        # previous sequence
        time_step_seq_prev = torch.cat([torch.tensor([0]), time_step_seq[:-1]])
        time_step_seq.to(x_t.device)
        time_step_seq_prev.to(x_t.device)

        if not self.test_to_device:
            self.move_to_device(x_t)

        step_size = math.ceil(time_steps / process_nums)

        self.eval()
        with torch.no_grad():
            cur_x = x_t
            for i in range(time_steps - 1, -1, -1):
                t, t_prev = time_step_seq[i], time_step_seq_prev[i]
                cur_x = self.ddim_reverse_step(cur_x, t, t_prev, eta)

                if show_process and i % step_size == 0:
                    col = process_nums - i // step_size
                    images = cur_x[:img_nums].detach().cpu()  # (img_nums, h, w, c)
                    for j in range(images.shape[0]):
                        pos = j * process_nums + col
                        plt.subplot(img_nums, process_nums, pos)
                        plt.imshow(tensor_image_to_PIL(images[j]))
                        plt.axis('off')

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
            test_imgs = convert_image_to_natural(self.ddpm_reverse_process(self.test_noise))
            grid = make_grid(test_imgs, nrow=16)
            self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
