"""
  * FileName: run.py
  * Author:   Slatter
  * Date:     2023/3/14 21:45
  * Description:  
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import Loader
from model import VAE
from torchvision.utils import save_image, make_grid

manualSeed = 999
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.set_float32_matmul_precision('medium')

# hyper parameters
image_size = 784
h_dim = 400
z_dim = 20
batch_size = 128
workers = 8
epochs = 100
lr = 1e-3
save_dir = 'checkpoints'


def train():
    dataloader = Loader(batch_size, workers)
    train_loader = dataloader.get_loader()

    model = VAE(image_size, h_dim, z_dim, lr)

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
    trainer.fit(model, train_loader)


def sample(ck_path):
    model = VAE.load_from_checkpoint(ck_path).to(torch.device('cuda'))
    model.eval()

    size = model.latent
    size.insert(0, 256)
    z = torch.randn(size).to(torch.device('cuda'))

    out = model.decode(z).view(-1, 1, 28, 28)
    out = make_grid(out, nrow=16)
    save_image(out, os.path.join('gen', 'sample.png'))


def reconstruct(ck_path):
    dataloader = Loader(batch_size, workers)
    train_loader = dataloader.get_loader()
    x, label = next(iter(train_loader))
    x = x.to(torch.device('cuda'))

    model = VAE.load_from_checkpoint(ck_path).to(torch.device('cuda'))
    model.eval()

    out, _, _ = model(x.view(-1, image_size))
    out = out.view(-1, 1, 28, 28)
    x_concat = torch.cat([x, out], dim=3)  # left is x, right is generated x
    save_image(x_concat, os.path.join('gen', 'reconstruct.png'))


if __name__ == '__main__':
    # if you want to sample, please change the mode to 'test'
    mode = 'train'
    if mode == 'train':
        train()
    else:
        # set your ckpt here
        ck_path = 'checkpoints/lr=0.001-batch=128/last.ckpt'
        if not os.path.exists('./gen'):  # create a directory 'gen' to store generated image
            os.system('mkdir gen')
        sample(ck_path)
        reconstruct(ck_path)
