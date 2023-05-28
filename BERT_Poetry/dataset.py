"""
  * FileName: dataset.py
  * Author:   Slatter
  * Date:     2023/5/26 19:28
  * Description:  
"""
import torch

from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from functools import partial


class PoemDataset(Dataset):
    def __init__(self, poems):
        super(PoemDataset, self).__init__()
        self.poems = poems

    @classmethod
    def load_from_file(cls, file_path: str):
        poems = []
        with open(file_path, 'r') as f:
            for poem in f:
                poems.append(poem.strip())

        return cls(poems)

    def __getitem__(self, idx):
        return self.poems[idx]

    def __len__(self):
        return len(self.poems)


class PoemLoader(pl.LightningDataModule):
    def __init__(self, poem_file: str, tokenizer_path: str, batch_size=8, max_len=128, device=torch.device('cpu')):
        super(PoemLoader, self).__init__()
        self.dataset = PoemDataset.load_from_file(poem_file)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_len = max_len
        self.pad_token_id = self.tokenizer.pad_token_id

        self.batch_size = batch_size
        self.device = device

    def collate_fn(self, batch):
        enc_batch = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_len, return_tensors='pt').to(self.device)
        return enc_batch

    def train_dataloader(self):
        print(f'Training samples: {len(self.dataset)}')
        collate = partial(self.collate_fn)
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate)


if __name__ == '__main__':
    file_path = '../dataset/poetry/poetry.txt'
    tokenizer_path = '/data2/daijincheng/pretrain/bert-base-chinese'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = PoemLoader(file_path, tokenizer_path, batch_size=16, device=device)
    train_loader = loader.train_dataloader()
    patch = next(iter(train_loader))
    print(patch)
