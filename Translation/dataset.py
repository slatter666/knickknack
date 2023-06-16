"""
  * FileName: dataset.py
  * Author:   Slatter
  * Date:     2023/6/7 20:09
  * Description:
"""
import json

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from utils import *


class NMTDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    @classmethod
    def load_from_file(cls, file_path: str):
        pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                pairs.append([line['english'], line['chinese']])

        return cls(pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

    def __len__(self):
        return len(self.pairs)


class NMTLoader(LightningDataModule):
    def __init__(self, train_set: NMTDataset, valid_set: NMTDataset, src_tokenizer: SelfTokenizer,
                 tgt_tokenizer: SelfTokenizer, max_len: int = 128, test_set: NMTDataset = None,
                 bsz: int = 128, num_workers: int = 8):
        self.train_dataset = train_set
        self.valid_dataset = valid_set
        self.test_dataset = test_set
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.max_len = max_len
        self.bsz = bsz
        self.num_workers = num_workers

    def collate(self, batch):
        en = [x[0] for x in batch]
        zh = [x[1] for x in batch]

        encode_src = self.src_tokenizer.encode_batch(en)
        src, src_padding_mask = encode_src['input_ids'], encode_src['attention_mask']
        src, src_padding_mask = torch.tensor(src, dtype=torch.long), torch.tensor(src_padding_mask)

        encode_tgt = self.tgt_tokenizer.encode_batch(zh)
        tgt, tgt_padding_mask = encode_tgt['input_ids'], encode_tgt['attention_mask']
        tgt, tgt_padding_mask = torch.tensor(tgt, dtype=torch.long), torch.tensor(tgt_padding_mask)

        src_mask, tgt_mask = create_mask(src_padding_mask, tgt_padding_mask)
        return src, src_mask, tgt, tgt_mask

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.bsz, shuffle=False, collate_fn=self.collate,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.bsz, shuffle=False, collate_fn=self.collate,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        raise NotImplementedError


if __name__ == '__main__':
    train_file_path = '../dataset/nmt/en-zh/translation2019zh_train.json'
    valid_file_path = '../dataset/nmt/en-zh/translation2019zh_valid.json'
    #
    # train_set = NMTDataset.load_from_file(train_file_path)
    valid_set = NMTDataset.load_from_file(valid_file_path)

    max_len = 128
    en_tokenizer = SelfTokenizer.load_from_file("../dataset/nmt/en-zh/en-tokenizer.json", max_len)
    zh_tokenizer = SelfTokenizer.load_from_file("../dataset/nmt/en-zh/zh-tokenizer.json", max_len)

    loader = NMTLoader(None, valid_set, en_tokenizer, zh_tokenizer, max_len, bsz=3)
    valid_dataloader = loader.val_dataloader()
    batch = next(iter(valid_dataloader))
