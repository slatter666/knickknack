"""
  * FileName: run.py
  * Author:   Slatter
  * Date:     2023/5/15 18:00
  * Description:  
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import AnimeLoader
from model import DiffusionModel
from torchvision.utils import make_grid, save_image
from utils import *

manualSeed = 999
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.set_float32_matmul_precision('medium')

# hyper parameters
image_size = 64
batch_size = 128
workers = 8

num_steps = 1000
ch = 64
ch_mult = [1, 2, 4, 4]
attn = [2]  # only in 16 * 16 use attention
num_res_blocks = 2
dropout = 0.1

epochs = 500
lr = 1e-4
save_dir = 'checkpoints'


def train():
    loader = AnimeLoader('../dataset/anime/processed', image_size, batch_size, workers)

    model = DiffusionModel(loader.dims, num_steps, ch, ch_mult, attn, num_res_blocks, dropout, lr)

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


def sample(ck_path, sample_steps=100, eta=0, discr_method='linear'):
    """
    :param ck_path: checkpoint path
    :param sample_steps: reverse steps
    :param eta: 0.0 for LLAMA and 1.0 for DDPM
    :param discr_method: linear or quadratic
    """
    image_save_path = f'gen/sample_steps={sample_steps}_eta={eta}.png'
    process_save_path = f'gen/process_steps={sample_steps}_eta={eta}.png'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffusionModel.load_from_checkpoint(ck_path).to(device)
    model.eval()

    size = (256, 3, 64, 64)
    z = torch.randn(size).to(device)

    # we show 15 diffusion process
    out = convert_image_to_natural(
        model.ddim_reverse_process(z, time_steps=sample_steps, eta=eta, discr_method=discr_method, show_process=True, img_nums=6, process_nums=15,
                                   save_path=process_save_path))
    out = make_grid(out, nrow=16)
    save_image(out, image_save_path)


def recover(ck_path, t, show_num=10):
    image_save_path = f'gen/recover_t={t}.png'

    # select `show_num` real images
    loader = AnimeLoader('../dataset/anime/processed', image_size, batch_size, workers)
    loader.setup()
    test_loader = loader.test_dataloader()
    batch, _ = next(iter(test_loader))
    sample = batch[:show_num]

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffusionModel.load_from_checkpoint(ck_path).to(device)
    model.eval()

    # forward process
    sample = sample.to(device)  # (show_num, 3, 64, 64)
    time = torch.tensor([t - 1] * sample.size(0))  # (show_num)
    model.move_to_device(sample)
    _, forward_result = model.forward_process(sample, time)  # (show_num, 3, 64, 64)

    # backward process
    backward_result = convert_image_to_natural(
        model.ddim_reverse_process(forward_result, time_steps=t, eta=0.0))  # (show_num, 3, 64, 64)

    all = []
    for i in range(show_num):
        all.append(sample[i])
        all.append(forward_result[i])
        all.append(backward_result[i])

    all = torch.stack(all)
    out = make_grid(all, nrow=3)
    save_image(out, image_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="ResNet Running Program")
    parser.add_argument('--mode', default='test', type=str, choices=['train', 'test'], help='choose train or test')
    parser.add_argument('--steps', type=int, default=100, help='choose how many sample steps')
    parser.add_argument('--eta', default=0.0, type=float, help='set eta, 0.0 for LLAMA and 1.0 for DDPM')

    args = parser.parse_args()
    if args.mode == 'train':
        train()
    else:
        # set your ckpt here
        ck_path = f'checkpoints/lr={lr}-batch={batch_size}-steps={num_steps}/last.ckpt'
        if not os.path.exists('./gen'):  # create a directory 'gen' to store generated image
            os.system('mkdir gen')

        # sample(ck_path, sample_steps=args.steps, eta=args.eta)

        recover(ck_path, t=100)
