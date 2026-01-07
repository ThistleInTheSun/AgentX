# data_module.py
import pytorch_lightning as pl
from config import config
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = config.batch_size
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),  # 转换为张量，范围[0,1]
            transforms.Normalize([0.5], [0.5]),  # 归一化到[-1,1]
        ])
    
    def setup(self, stage=None):
        # 加载完整数据集
        self.train_dataset = datasets.MNIST(
            root="./data", 
            train=False, 
            download=True, 
            transform=self.transform
        )
        self.val_dataset = datasets.MNIST(
            root="./data", 
            train=False, 
            download=True, 
            transform=self.transform
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=4
        )