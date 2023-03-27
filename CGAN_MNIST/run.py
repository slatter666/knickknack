"""
  * FileName: run.py
  * Author:   Slatter
  * Date:     2023/3/27 20:30
  * Description:  
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import MNISTLoader
from model import MNISTGAN
from torchvision.utils import make_grid, save_image

manualSeed = 999
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.set_float32_matmul_precision('medium')

# hyper parameters
h_dim = 128
latent_dim = 100
num_classes = 10
batch_size = 128
workers = 8
epochs = 200
lr = 2e-4
save_dir = 'checkpoints'


def train():
    loader = MNISTLoader('../dataset/mnist', batch_size, workers)

    model = MNISTGAN(loader.dims, latent_dim, h_dim, num_classes, lr)

    # training
    checkpoint_path = os.path.join(save_dir, f'lr={lr}-batch={batch_size}')
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        save_weights_only=True,
        save_last=True
    )

    tb_logger = pl.loggers.TensorBoardLogger(save_dir=os.path.join(checkpoint_path, 'logs'))
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=epochs, callbacks=[checkpoint_callback],
                         logger=tb_logger)
    trainer.fit(model, loader)


def sample(ck_path, s_num: int=200):
    device = torch.device('cuda')
    model = MNISTGAN.load_from_checkpoint(ck_path).to(device)
    model.eval()

    size = (s_num, latent_dim)
    z_noise = torch.randn(size).to(device)
    z_label = F.one_hot(torch.matmul(torch.arange(10).view(-1, 1), torch.ones(1, 20, dtype=torch.long)).flatten(), num_classes).to(device)

    out = model(z_noise, z_label).view(200, 1, 28, 28)
    out = make_grid(out, nrow=int(s_num / num_classes))
    save_image(out, os.path.join('gen', 'sample.png'))


if __name__ == '__main__':
    # if you want to sample, please change the mode to 'test'
    mode = 'test'
    if mode == 'train':
        train()
    else:
        # set your ckpt here
        ck_path = f'checkpoints/lr={lr}-batch={batch_size}/last.ckpt'
        if not os.path.exists('./gen'):  # create a directory 'gen' to store generated image
            os.system('mkdir gen')
        sample(ck_path)
