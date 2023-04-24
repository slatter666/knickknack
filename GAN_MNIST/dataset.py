"""
  * FileName: dataset.py
  * Author:   Slatter
  * Date:     2023/3/25 11:52
  * Description:  
"""
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, datasets
import pytorch_lightning as pl


class MNISTLoader(pl.LightningDataModule):
    def __init__(self, data_dir, bsz, workers):
        super(MNISTLoader, self).__init__()
        self.data_dir = data_dir
        self.bsz = bsz
        self.workers = workers

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        self.dims = (1, 28, 28)

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == 'fit' or stage is None:
            mnist_full = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == 'test' or stage is None:
            self.mnist_test = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        print(f'Train samples: {len(self.mnist_train)}')
        return DataLoader(self.mnist_train, batch_size=self.bsz, shuffle=True, num_workers=self.workers)

    def val_dataloader(self):
        print(f'Valid samples: {len(self.mnist_val)}')
        return DataLoader(self.mnist_val, batch_size=self.bsz, shuffle=False, num_workers=self.workers)

    def test_dataloader(self):
        print(f'Test samples: {len(self.mnist_test)}')
        return DataLoader(self.mnist_test, batch_size=self.bsz, shuffle=False, num_workers=self.workers)


if __name__ == '__main__':
    loader = MNISTLoader('../dataset/mnist', 128, 8)
    loader.setup(None)
    train_loader = loader.train_dataloader()
    val_loader = loader.val_dataloader()
    test_loader = loader.test_dataloader()
