"""
  * FileName: run.py
  * Author:   Slatter
  * Date:     2023/6/8 16:01
  * Description:  
"""
import argparse
import os
from typing import Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import *
from model import NMT
from utils import *

manualSeed = 999
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.set_float32_matmul_precision('medium')

parser = argparse.ArgumentParser(prog="Seq2Seq Machine Translation")
parser.add_argument("--batch", type=int, default=128, help="set batch size")
parser.add_argument("--num-workers", type=int, default=8, help="set number of workers")
parser.add_argument("--embed-size", type=int, default=512, help="set embedding size")
parser.add_argument("--ffn-hid-size", type=int, default=512, help="feed-forward hidden size")
parser.add_argument("--nhead", type=int, default=8, help="num of head")
parser.add_argument("--encoder-layer", type=int, default=3, help="num of transformer encoder layer")
parser.add_argument("--decoder-layer", type=int, default=3, help="num of transformer decoder layer")
parser.add_argument("--dropout", type=float, default=0.1, help="set dropout rate")
parser.add_argument("--norm-first", type=bool, default=False, help="False for post-norm and True for pre-norm")
parser.add_argument("--max-len", type=int, default=128, help="set generated translation's max length")
parser.add_argument("--lr", type=float, default=2e-4, help="set learning rate")
parser.add_argument("--activation", type=str, default='relu', help='set activation function')
parser.add_argument("--epochs", type=int, default=30, help="set epoch of training")
parser.add_argument("--warmup", type=int, default=0, help="set warm up steps")
parser.add_argument("--save-dir", type=str, default="checkpoints", help="set where to save checkpoints")
parser.add_argument("--num-gpus", type=int, default=1, help="number of available GPUs")
parser.add_argument("--mode", type=str, choices=['train', 'test'], default='test', help="choose train or test model")

args = parser.parse_args()

# set hyper parameters
batch_size = args.batch
num_workers = args.num_workers
epochs = args.epochs
lr = args.lr
d_model = args.embed_size
ffn_hid_size = args.ffn_hid_size
n_head = args.nhead
encoder_layer = args.encoder_layer
decoder_layer = args.decoder_layer
dropout = args.dropout
norm_first = args.norm_first
activation = args.activation
warmup = args.warmup
max_len = args.max_len

save_dir = args.save_dir
gpus = args.num_gpus

train_file_path = '../dataset/nmt/en-zh/translation2019zh_train.json'
valid_file_path = '../dataset/nmt/en-zh/translation2019zh_valid.json'

en_tokenizer = SelfTokenizer.load_from_file("../dataset/nmt/en-zh/en-tokenizer.json", max_len)
zh_tokenizer = SelfTokenizer.load_from_file("../dataset/nmt/en-zh/zh-tokenizer.json", max_len)


def train():
    train_set = NMTDataset.load_from_file(train_file_path)
    valid_set = NMTDataset.load_from_file(valid_file_path)

    loader = NMTLoader(train_set, valid_set, en_tokenizer, zh_tokenizer, max_len, bsz=batch_size,
                       num_workers=num_workers)
    train_loader = loader.train_dataloader()
    valid_loader = loader.val_dataloader()

    model = NMT(
        src_tokenizer=en_tokenizer,
        tgt_tokenizer=zh_tokenizer,
        d_model=d_model,
        nhead=n_head,
        num_encoder_layers=encoder_layer,
        num_decoder_layers=decoder_layer,
        dim_feedforward=ffn_hid_size,
        dropout=dropout,
        activation=activation,
        norm_first=norm_first,
        lr=lr,
        warmup_steps=warmup
    )

    # training
    checkpoint_path = os.path.join(save_dir, f'lr={lr}-batch={batch_size}')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_path,
        filename='{epoch}-{val_loss:.3f}',
        save_weights_only=True,
        save_top_k=5,
        mode='min'
    )

    tb_logger = pl.loggers.TensorBoardLogger(save_dir=os.path.join(checkpoint_path, 'logs'))
    trainer = pl.Trainer(accelerator='gpu', devices=gpus, max_epochs=epochs, precision=16,
                         strategy='ddp_find_unused_parameters_false', callbacks=[checkpoint_callback],
                         logger=tb_logger)
    trainer.fit(model, train_loader, valid_loader)


def translate(ck_path: str, en: Union[List[str], str]):
    if isinstance(en, list):
        pass
    elif isinstance(en, str):
        en = [en]
    else:
        raise AssertionError('Input must be string or a list of string')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NMT.load_from_checkpoint(ck_path).to(device)

    res = model.greedy_decode(en, device=device)
    print(res)

    beam_res = model.beam_search(en, device=device)
    print(beam_res)


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        ck_path = 'checkpoints/lr=0.0002-batch=150/epoch=29-val_loss=2.584.ckpt'

        en = ["I want to train a neural machine translation model",
              "print this paper and send it to Bob",
              "have you ever used my machine translation model?",
              "It's sunny today, let's go to the beach to have a happy holiday!"]
        translate(ck_path, en)
