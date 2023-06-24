"""
  * FileName: dataset.py
  * Author:   Slatter
  * Date:     2023/6/18 21:15
  * Description:
"""

import json

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer

from utils import *


class GPTDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    @classmethod
    def load_from_file(cls, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = json.load(f)
            lines = [piece['text'] for piece in lines]

        return cls(lines)

    def __getitem__(self, idx):
        return self.texts[idx]

    def __len__(self):
        return len(self.texts)


class GPTLoader(LightningDataModule):
    def __init__(self, full_set: GPTDataset, tokenizer: BertTokenizer, max_len: int = 1024, train_size: float=0.97,
                 bsz: int = 128, num_workers: int = 8):
        self.train_dataset, self.valid_dataset = self.split_dataset(full_set, train_size)
        self.tokenizer = tokenizer

        self.max_len = max_len
        self.bsz = bsz
        self.num_workers = num_workers

    @staticmethod
    def split_dataset(dataset: GPTDataset, train_rate: float):
        train_size = int(len(dataset) * train_rate)
        valid_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(999)
        train_set, valid_set = random_split(dataset, [train_size, valid_size], generator=generator)
        return train_set, valid_set

    def collate(self, batch):
        enc_batch = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_len, return_tensors='pt')

        input_ids, padding_mask = enc_batch['input_ids'], enc_batch['attention_mask']
        attention_mask = create_mask(padding_mask)
        return input_ids, attention_mask

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.bsz, shuffle=False, collate_fn=self.collate,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.bsz, shuffle=False, collate_fn=self.collate,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        raise NotImplementedError


if __name__ == '__main__':
    file_path = '../dataset/corpus/processed.json'
    full_set = GPTDataset.load_from_file(file_path)

    max_len = 1024
    tokenizer_path = '/data2/daijincheng/pretrain/bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    loader = GPTLoader(full_set, tokenizer, max_len, bsz=5)
    train_dataloader = loader.train_dataloader()
    valid_dataloader = loader.val_dataloader()
    batch = next(iter(train_dataloader))
