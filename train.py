import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
import model
import numpy as np
import glob, os
import logging
from rich.logging import RichHandler
from rich.progress import track

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET",
    # format="Rank: " + str(rank) + "/" + str(size) + ": %(asctime)s - %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
log = logging.getLogger("rich")


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, n_blocks: int = 4, start_filters: int = 32, learning_rate: float = 1e-3):
        super().__init__()
        self.unet = model.UNet(
            out_channels=1, n_blocks=n_blocks, start_filters=start_filters
        )
        self.learning_rate = learning_rate

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
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

        assert len(self.sample_files) > 0
        assert len(self.sample_files) == len(self.target_files)

        npvec = np.load(self.sample_files[0])
        ncells = len(npvec)
        nc_sqrt = int(np.sqrt(ncells))
        ncols = 2 ** (int(np.floor(np.log2(nc_sqrt))))
        q, r = np.divmod(ncells, ncols)

        if r > 0:
            nrowsi = q + 1
        else:
            nrowsi = q

        # UNet rows and cols must be at least divisible by 8.
        r8 = np.remainder(nrowsi, 8)
        if r8 > 0:
            nrows = nrowsi + 8 - r8
        else:
            nrows = nrowsi

        npad = nrows * ncols - ncells
        left_pad = 0
        if npad > 0:
            npmed = np.median(npvec)
            npnew = npmed * np.ones((ncells + npad), dtype=float)
            min_pad, rem_val = np.divmod(npad, 2)
            if rem_val == 0:
                left_pad = min_pad
            else:
                left_pad = min_pad + 1

        self.ncells = ncells
        self.nrows = nrows
        self.ncols = ncols
        self.left_pad = left_pad
        self.npad = npad
        log.info("Input shape = %d, Output shape = %d x %d\n" % (ncells, nrows, ncols))

    def __len__(self):
        return len(self.sample_files)

    def _reshape_1d_to_2d(self, array: np.ndarray):
        if array.ndim != 1:
            raise ValueError("Input array must be 1D.")

        npmed = np.median(array)
        npnew = npmed * np.ones((self.ncells + self.npad), dtype=np.float32)
        npnew[self.left_pad : self.ncells + self.left_pad] = array
        out_array = npnew.reshape(1, self.nrows, self.ncols)

        return out_array

    def __getitem__(self, idx):
        sample = self._reshape_1d_to_2d(
            np.load(self.sample_files[idx]).astype(np.float32)
        )
        target = self._reshape_1d_to_2d(
            np.load(self.target_files[idx]).astype(np.float32)
        )
        return sample, target


# n_blocks, start_filters, learning_rate, batch size, epochs, sample folder, target folder, training logs folder

# npydataset = NumpyDataset("/tmp/samples/", "/tmp/targets")
# dataloader = DataLoader(npydataset, batch_size=1)

# model = LitAutoEncoder()
# trainer = pl.Trainer()
# trainer.fit(model=model, train_dataloaders=dataloader)
