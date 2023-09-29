from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import BaseDataset


class R2DataLoader(DataLoader):
    def __init__(self, args, split, shuffle):
        self.args = args
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.split = split

        # self.transform = transforms.Compose([
        #     transforms.ToTensor()])

        self.dataset = BaseDataset('data.json', split)
        # print(self.dataset.size())
        
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': 4,
            'shuffle': self.shuffle,
            'num_workers': 1
        }
        super().__init__(**self.init_kwargs)

