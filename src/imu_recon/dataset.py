from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from src.imu_recon.utils import bandpass_filter
import re


class EEGIMUDataset(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, csv_path: Path, window: int, stride: int, bandpass: Tuple[int,int] = (5,30), imu_mean: Tensor = None, imu_std: Tensor = None):
        df = pd.read_csv(csv_path)
        eeg_cols = [c for c in df.columns if "EEG" in c and "AUX" not in c and "index" not in c]
        imu_cols = [c for c in df.columns if "linAcc" in c or "gyr" in c]

        self.channel_names = [re.sub(r"OpenBCI_EEG_", "", str(col)) for col in eeg_cols]

        self.eeg = bandpass_filter(df[eeg_cols].to_numpy(), low = bandpass[0], high = bandpass[1], fs = 125).copy()
        self.eeg = torch.tensor(self.eeg, dtype=torch.float32)

        self.mean_eeg = self.eeg.mean(dim=0)

        self.imu = torch.tensor(df[imu_cols].to_numpy(), dtype=torch.float32)

        if imu_mean is not None and imu_std is not None:
            self.imu_std = imu_std
            self.imu_mean = imu_mean
        else:
            self.imu_std = self.imu.std(dim=0, keepdim=True).clamp(min=1e-6)
            self.imu_mean = self.imu.mean(dim=0, keepdim=True)

        # global standardization for IMU
        self.imu = (self.imu - self.imu_mean) / self.imu_std
        self.window, self.stride = window, stride

    def __len__(self) -> int:
        return max(0, (len(self.eeg) - self.window) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window
        eeg_window = self.eeg[start:end]
        imu_window = self.imu[start:end]

        # Standardize EEG window: (T, C)
        eeg_window = (eeg_window - eeg_window.mean(dim=0)) / eeg_window.std(dim=0).clamp(min=1e-6)

        return eeg_window, imu_window

    def destandardize_imu(self, x: Tensor) -> Tensor:
        """
        Reverse normalization of IMU signal.

        Args:
            x (Tensor): shape [..., C], standardized IMU

        Returns:
            Tensor: de-standardized IMU
        """
        return x * self.imu_std + self.imu_mean

    def train_test_val_split(self, bs):

        generator1 = torch.Generator().manual_seed(42)
        n = len(self)
        train_size = int(0.7 * n)
        val_size = int(0.15 * n)
        test_size = n - train_size - val_size
        train_ds, val_ds, test_ds = random_split(self, [train_size, val_size, test_size], generator=generator1)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

        return train_loader, val_loader, test_loader
