import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
import model
import numpy as np
import glob, os


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.unet = model.UNet(out_channels=1)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x_hat = self.unet(x)
        loss = F.mse_loss(x_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class NumpyDataset(Dataset):
    def __init__(self, sample_dir, target_dir):
        self.sample_dir = sample_dir
        self.target_dir = target_dir

        self.sample_files = []
        for fname in glob.iglob(self.sample_dir + "/**", recursive=True):
            if os.path.isfile(fname) and fname.endswith(".npy"):
                self.sample_files.append(fname)

        self.target_files = []
        for fname in glob.iglob(self.target_dir + "/**", recursive=True):
            if os.path.isfile(fname) and fname.endswith(".npy"):
                self.target_files.append(fname)

        assert len(self.sample_files) == len(self.target_files)

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        sample = np.load(self.sample_files[idx]).astype(np.float32)
        target = np.load(self.target_files[idx]).astype(np.float32)
        return sample, target


npydataset = NumpyDataset("/tmp/samples/", "/tmp/targets")
dataloader = DataLoader(npydataset, batch_size=1)

model = LitAutoEncoder()
trainer = pl.Trainer()
trainer.fit(model=model, train_dataloaders=dataloader)
