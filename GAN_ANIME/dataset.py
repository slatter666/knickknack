"""
  * FileName: dataset.py
  * Author:   Slatter
  * Date:     2023/3/27 11:58
  * Description:  
"""
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, datasets
import pytorch_lightning as pl


class AnimeLoader(pl.LightningDataModule):
    def __init__(self, data_dir, img_size, bsz, workers):
        super(AnimeLoader, self).__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.bsz = bsz
        self.workers = workers

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def setup(self, stage: str = None):
        full = datasets.ImageFolder(self.data_dir, transform=self.transform)
        train, val = int(len(full) * 0.90), int(len(full) * 0.05)
        test = len(full) - train - val
        self.anime_train, self.anime_val, self.anime_test = random_split(full, [train, val, test], torch.Generator().manual_seed(999))

    def train_dataloader(self):
        print(f'Train samples: {len(self.anime_train)}')
        return DataLoader(self.anime_train, batch_size=self.bsz, shuffle=True, num_workers=self.workers)

    def val_dataloader(self):
        print(f'Valid samples: {len(self.anime_val)}')
        return DataLoader(self.anime_val, batch_size=self.bsz, shuffle=False, num_workers=self.workers)

    def test_dataloader(self):
        print(f'Test samples: {len(self.anime_test)}')
        return DataLoader(self.anime_test, batch_size=self.bsz, shuffle=False, num_workers=self.workers)


if __name__ == '__main__':
    loader = AnimeLoader('../dataset/anime/processed', 64, 128, 8)
    loader.setup()
    train_loader = loader.train_dataloader()
    val_loader = loader.val_dataloader()
    test_loader = loader.test_dataloader()
