"""
  * FileName: run.py
  * Author:   Slatter
  * Date:     2023/6/22 19:16
  * Description:
"""
import argparse
import os
from typing import List, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import *
from model import GPT2Mini
from utils import *

manualSeed = 999
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.set_float32_matmul_precision('medium')

parser = argparse.ArgumentParser(prog="GPT2Mini")
parser.add_argument("--batch", type=int, default=128, help="set batch size")
parser.add_argument("--num-workers", type=int, default=8, help="set number of workers")
parser.add_argument("--embed-size", type=int, default=512, help="set embedding size")
parser.add_argument("--ffn-hid-size", type=int, default=2048, help="feed-forward hidden size")
parser.add_argument("--nhead", type=int, default=8, help="num of head")
parser.add_argument("--num-layer", type=int, default=6, help="num of transformer encoder layer")
parser.add_argument("--dropout", type=float, default=0.1, help="set dropout rate")
parser.add_argument("--norm-first", type=bool, default=False, help="False for post-norm and True for pre-norm")
parser.add_argument("--max-len", type=int, default=512, help="set generated translation's max length")
parser.add_argument("--lr", type=float, default=2e-4, help="set learning rate")
parser.add_argument("--activation", type=str, default='relu', help='set activation function')
parser.add_argument("--epochs", type=int, default=30, help="set epoch of training")
parser.add_argument("--warmup", type=int, default=0, help="set warm up steps")
parser.add_argument("--save-dir", type=str, default="checkpoints", help="set where to save checkpoints")
parser.add_argument("--num-gpus", type=int, default=1, help="number of available GPUs")
parser.add_argument("--mode", type=str, choices=['train', 'test'], default='test', help="choose train or test model")
parser.add_argument("--temperature", type=float, default=1.0, help="set temperature, used for generation")
parser.add_argument("--top-p", type=float, default=0.85, help="set top_p sample rate, used for generation")

args = parser.parse_args()

# set hyper parameters
batch_size = args.batch
num_workers = args.num_workers
epochs = args.epochs
lr = args.lr
d_model = args.embed_size
ffn_hid_size = args.ffn_hid_size
n_head = args.nhead
num_layer = args.num_layer
dropout = args.dropout
norm_first = args.norm_first
activation = args.activation
warmup = args.warmup
max_len = args.max_len

save_dir = args.save_dir
gpus = args.num_gpus

file_path = '../dataset/corpus/processed2.json'

tokenizer_path = '/data2/daijincheng/pretrain/bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)


def train():
    full_set = GPTDataset.load_from_file(file_path)

    loader = GPTLoader(full_set, tokenizer, max_len, bsz=batch_size, num_workers=num_workers)
    train_loader = loader.train_dataloader()
    valid_loader = loader.val_dataloader()

    model = GPT2Mini(
        tokenizer=tokenizer,
        d_model=d_model,
        nhead=n_head,
        dim_feedforward=ffn_hid_size,
        num_layers=num_layer,
        dropout=dropout,
        max_len=max_len,
        activation=activation,
        norm_first=norm_first,
        lr=lr,
        warmup_steps=warmup
    )

    # training
    checkpoint_path = os.path.join(save_dir, f'lr={lr}-batch={batch_size}')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_ppl',
        dirpath=checkpoint_path,
        filename='{epoch}-{val_loss:.3f}-{val_ppl:.3f}',
        save_weights_only=True,
        save_top_k=5,
        mode='min'
    )

    tb_logger = pl.loggers.TensorBoardLogger(save_dir=os.path.join(checkpoint_path, 'logs'))
    trainer = pl.Trainer(accelerator='gpu', devices=gpus, max_epochs=epochs, precision=16,
                         strategy='ddp_find_unused_parameters_false', callbacks=[checkpoint_callback],
                         logger=tb_logger)
    trainer.fit(model, train_loader, valid_loader)


def generate(ck_path: str, prompts: Union[str, List[str]], temperature: float = 0, top_p: float = 0.9):
    if isinstance(prompts, list):
        pass
    elif isinstance(prompts, str):
        prompts = [prompts]
    else:
        raise AssertionError('Input must be string or a list of string')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT2Mini.load_from_checkpoint(ck_path).to(device)

    res = model.generate(prompts, temperature, top_p, device)
    for x in res:
        print(x.replace(' ', ''))
        print()


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        ck_path = 'checkpoints/lr=0.0002-batch=32/epoch=29-val_loss=2.643-val_ppl=15.407.ckpt'
        temperature = 0
        top_p = 0.9

        prompts = [
            "在这个阳光明媚的日子里",
            "怀揣着梦想",
            "校园生活是美好的",
            "起风了",
            "每一个小孩子",
            "光阴似箭",
            "秋日午后",
            "母亲"
        ]
        generate(ck_path, prompts, temperature=1.0, top_p=0.85)
