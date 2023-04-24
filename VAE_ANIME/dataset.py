"""
  * FileName: dataset.py
  * Author:   Slatter
  * Date:     2023/3/16 22:45
  * Description:  
"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class AnimeLoader(DataLoader):
    def __init__(self, data_dir, img_size, bsz, workers):
        self.data_dir = data_dir
        self.img_size = img_size
        self.bsz = bsz
        self.workers = workers

    def get_loader(self):
        dataset = datasets.ImageFolder(self.data_dir,
                                       transform=transforms.Compose([
                                           transforms.Resize(self.img_size),
                                           transforms.CenterCrop(self.img_size),
                                           transforms.ToTensor(),
                                       ]))

        print('Total samples: {}'.format(len(dataset)))
        dataloader = DataLoader(dataset, batch_size=self.bsz, shuffle=True, num_workers=self.workers)
        return dataloader


if __name__ == '__main__':
    anime = AnimeLoader('../dataset/anime/processed', 64, 64, 8)
    loader = anime.get_loader()
    batch, _ = next(iter(loader))
    print(batch.size())
