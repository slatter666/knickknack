"""
  * FileName: run.py
  * Author:   Slatter
  * Date:     2023/5/27 18:11
  * Description:
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import *
from model import PoetryBert

import argparse

parser = argparse.ArgumentParser(prog="Bert Poetry Program")
parser.add_argument('--batch-size', default=64, type=int, help='set batch size')
parser.add_argument('--epochs', default=10, type=int, help='set training epochs')
parser.add_argument('--lr', default=2e-5, type=float, help='set learning rate')
parser.add_argument('--save-dir', default='checkpoints', type=str, help='set folder path to save checkpoint')
parser.add_argument('--mode', default='test', type=str, choices=['train', 'test'], help='set mode')
parser.add_argument('--temperature', default=0.8, type=float, help='set temperature, must be in range of [0, 1]')
parser.add_argument('--top-p', default=0.9, type=float, help='set top p sample rate, must be in range of [0, 1]')
parser.add_argument('--head', default='', type=str, help='set initial head of poem')
parser.add_argument('--category', default='normal', type=str, choices=['normal', 'acrostic'],
                    help='which kind of poem to generate')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file_path = '../dataset/poetry/poetry.txt'
tokenizer_path = '/data2/daijincheng/pretrain/bert-base-chinese'
model_pretrain_path = '/data2/daijincheng/pretrain/bert-base-chinese'


def train():
    loader = PoemLoader(file_path, tokenizer_path, batch_size=args.batch_size, device=device)

    model = PoetryBert(model_pretrain_path, loader.pad_token_id).to(device)

    # training
    checkpoint_path = os.path.join(args.save_dir, f'lr={args.lr}-batch={args.batch_size}')
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        save_weights_only=True,
        save_last=True
    )

    tb_logger = pl.loggers.TensorBoardLogger(save_dir=os.path.join(checkpoint_path, 'logs'))
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=args.epochs, callbacks=[checkpoint_callback],
                         logger=tb_logger)
    trainer.fit(model, loader)


def generate(ck_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = PoetryBert.load_from_checkpoint(ck_path).to(device)

    if args.category == 'normal':
        poem = model.generate(tokenizer, head=args.head, temperature=args.temperature, top_p=args.top_p)
    else:
        assert len(args.head) != 0, "Please give your head to generate acrostic poem"
        poem = model.generate(tokenizer, head=list(args.head), temperature=args.temperature, top_p=args.top_p)

    return poem


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        # set your ckpt here
        ck_path = f'checkpoints/lr={args.lr}-batch={args.batch_size}/last.ckpt'
        poem = generate(ck_path)
        print(poem)
