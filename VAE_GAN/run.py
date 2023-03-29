"""
  * FileName: run.py
  * Author:   Slatter
  * Date:     2023/3/27 13:15
  * Description:
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import AnimeLoader
from model import AnimeGAN
from torchvision.utils import make_grid, save_image

manualSeed = 999
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.set_float32_matmul_precision('medium')

# hyper parameters
image_size = (3, 64, 64)
latent_size = (100, 1, 1)
depth = 64
batch_size = 128
workers = 8
epochs = 200
lr = 2e-4
save_dir = 'checkpoints'


def train():
    loader = AnimeLoader('../dataset/anime/processed', image_size[-1], batch_size, workers)

    model = AnimeGAN(image_size, latent_size, depth, lr)

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


def check_real_image():
    loader = AnimeLoader('../dataset/anime/processed', image_size[-1], 256, workers)
    loader.setup()
    test_loader = loader.test_dataloader()
    batch, _ = next(iter(test_loader))

    out = make_grid(batch, nrow=16)
    save_image(out, os.path.join('gen', 'real.png'))


def sample(ck_path):
    model = AnimeGAN.load_from_checkpoint(ck_path).to(torch.device('cuda'))
    model.eval()

    size = (256, *latent_size)
    z = torch.randn(size).to(torch.device('cuda'))

    out = model(z)
    out = make_grid(out, nrow=16)
    save_image(out, os.path.join('gen', 'sample.png'))


if __name__ == '__main__':
    # if you want to sample, please change the mode to 'test'
    mode = 'train'
    if mode == 'train':
        train()
    else:
        # set your ckpt here
        ck_path = f'checkpoints/lr={lr}-batch={batch_size}/last.ckpt'
        if not os.path.exists('./gen'):  # create a directory 'gen' to store generated image
            os.system('mkdir gen')
        check_real_image()
        sample(ck_path)
