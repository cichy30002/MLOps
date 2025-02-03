import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from src.model.Dataset import BodyFatDataset


class BodyFatDataModule(pl.LightningDataModule):
    def __init__(self, data, batch_size=32):
        super().__init__()
        self.data = data
        self.batch_size = batch_size

    def setup(self, stage=None):
        dataset = BodyFatDataset(self.data)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
