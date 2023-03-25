"""
  * FileName: dataset.py
  * Author:   Slatter
  * Date:     2023/3/14 21:45
  * Description:  
"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Loader:
    def __init__(self, bsz, workers):
        self.bsz = bsz
        self.workers = workers

    def get_loader(self):
        dataset = datasets.MNIST(root='../dataset/mnist', train=True, transform=transforms.ToTensor(), download=True)
        print(f'Total samples: {len(dataset)}')

        loader = DataLoader(dataset, batch_size=self.bsz, shuffle=True, num_workers=self.workers)
        return loader


if __name__ == '__main__':
    loader = Loader(64, 2)
    train_loader = loader.get_loader()
    batch, label = next(iter(train_loader))
    print(batch.shape)
