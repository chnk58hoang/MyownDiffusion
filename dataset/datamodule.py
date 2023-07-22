import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 train_dataset,
                 valid_dataset,
                 batch_size,
                 num_workers=8):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
