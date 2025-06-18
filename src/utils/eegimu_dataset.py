from pathlib import Path
import re
from typing import Tuple

import pandas as pd
import torch
from torch import Tensor

from src.utils.utils import bandpass_filter
from src.utils.base_dataset import BaseDataset


class EEGIMUDataset(BaseDataset):
    """
    Load EEG and IMU data from CSV, apply windowing and normalization.

    Attributes:
        channel_names: List of EEG channel labels.
        eeg: Full EEG tensor (T × C).
        imu: Full standardized IMU tensor (T × D).
        eeg_mean: Mean per-channel over full EEG.
        imu_mean: Mean per-dimension over full IMU.
        imu_std: Std dev per-dimension over full IMU.
    """

    def __init__(
        self,
        csv_path: Path,
        window: int,
        stride: int,
        bandpass: Tuple[int, int] = (5, 30),
    ) -> None:
        """
        Read CSV, filter EEG, standardize IMU, compute stats.

        Args:
            csv_path: Path to data CSV file.
            window: Number of samples per window.
            stride: Step between windows.
            bandpass: Low/high cutoff (Hz) for EEG filter.
        """
        super().__init__(window, stride)
        # Read DataFrame
        df = pd.read_csv(csv_path)

        # Identify EEG/IMU columns
        eeg_cols = [c for c in df.columns if "EEG" in c and "AUX" not in c and "index" not in c]
        imu_cols = [c for c in df.columns if any(key in c for key in ("linAcc", "gyr"))]

        # Clean channel names
        self.channel_names = [re.sub(r"OpenBCI_EEG_", "", c) for c in eeg_cols]

        # Bandpass filter EEG and convert to tensor
        inp_np = bandpass_filter(df[eeg_cols].to_numpy(), low=bandpass[0], high=bandpass[1], fs=125)
        self.inp = torch.tensor(inp_np.copy(), dtype=torch.float32)
        self.in_mean = self.inp.mean(dim=0)

        # Load and standardize IMU
        out_tensor = torch.tensor(df[imu_cols].to_numpy(), dtype=torch.float32)

        # Prevent division by zero
        self.out_std = out_tensor.std(dim=0, keepdim=True).clamp(min=1e-6)
        self.out_mean = out_tensor.mean(dim=0, keepdim=True)
        self.out = (out_tensor - self.out_mean) / self.out_std
        self.inp_dim = len(eeg_cols)
        self.out_dim = len(imu_cols)

    def __len__(self) -> int:
        """
        Count of available windows: floor((T - window) / stride) + 1.
        """
        total = self.inp.shape[0]
        return max(0, (total - self.window) // self.stride + 1)

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Return normalized EEG and IMU windows.

        Args:
            idx: Index of window segment.

        Returns:
            Tuple of (eeg_window, imu_window), shape (window × channels).
        """
        start = idx * self.stride
        end = start + self.window
        inp_win = self.inp[start:end]
        out_win = self.out[start:end]

        # Normalize EEG per window
        inp_win = (inp_win - inp_win.mean(dim=0)) / inp_win.std(dim=0).clamp(min=1e-6)
        return inp_win, out_win

    def destandardize(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Convert standardized IMU back to original units.

        Args:
            x: Standardized IMU tensor (... × D).

        Returns:
            De-standardized IMU tensor.
        """
        return x * self.out_std + self.out_mean
