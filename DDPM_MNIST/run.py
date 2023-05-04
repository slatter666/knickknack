"""
  * FileName: run.py
  * Author:   Slatter
  * Date:     2023/5/3 18:21
  * Description:
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import MNISTLoader
from model import DiffusionModel
from torchvision.utils import make_grid, save_image
from utils import *

manualSeed = 999
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.set_float32_matmul_precision('medium')

# hyper parameters
down_channels = [1, 64, 128]
up_channels = [128, 64, 1]
time_embed_dim = 200

image_size = 28
batch_size = 128
workers = 8

num_steps = 300
epochs = 300
lr = 1e-4
save_dir = 'checkpoints'


def train():
    loader = MNISTLoader('../dataset/mnist', batch_size, workers)

    model = DiffusionModel(loader.dims, down_channels, up_channels, time_embed_dim, num_steps, lr)

    # training
    checkpoint_path = os.path.join(save_dir, f'lr={lr}-batch={batch_size}-steps={num_steps}')
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        save_weights_only=True,
        save_last=True
    )

    tb_logger = pl.loggers.TensorBoardLogger(save_dir=os.path.join(checkpoint_path, 'logs'))
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=epochs, callbacks=[checkpoint_callback],
                         logger=tb_logger)
    trainer.fit(model, loader)


def sample(ck_path):
    image_save_path = 'gen/sample.png'
    process_save_path = 'gen/process.png'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffusionModel.load_from_checkpoint(ck_path).to(device)
    model.eval()

    size = (256, 1, 28, 28)
    z = torch.randn(size).to(device)

    out = convert_image_to_natural(model.reverse_process(z, show_process=True, img_nums=6, process_nums=8, save_path=process_save_path))
    out = make_grid(out, nrow=16)
    save_image(out, image_save_path)


if __name__ == '__main__':
    # if you want to sample, please change the mode to 'test'
    mode = 'test'
    if mode == 'train':
        train()
    else:
        # set your ckpt here
        ck_path = f'checkpoints/lr={lr}-batch={batch_size}-steps={num_steps}/last.ckpt'
        if not os.path.exists('./gen'):  # create a directory 'gen' to store generated image
            os.system('mkdir gen')
        sample(ck_path)