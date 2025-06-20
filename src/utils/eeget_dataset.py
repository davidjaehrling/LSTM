from pathlib import Path
import re
from typing import Tuple

import pandas as pd
import torch
from torch import Tensor

from src.utils.utils import bandpass_filter
from src.utils.base_dataset import BaseDataset


class EEGETDataset(BaseDataset):
    """
    Load EEG and Eye-Tracker (ET) data from CSV, apply windowing and normalization.

    Attributes:
        channel_names: List of EEG channel labels.
        inp: Full EEG tensor (T × C).
        out: Full standardized ET tensor (T × D).
        inp_mean: Mean per-channel over full EEG.
        out_mean: Mean per-dimension over full ET.
        out_std: Std dev per-dimension over full ET.
    """

    def __init__(
        self,
        csv_path: Path,
        window: int,
        stride: int,
        bandpass: Tuple[int, int] = (5, 30),
    ) -> None:
        """
        Read CSV, filter EEG, standardize ET, compute stats.

        Args:
            csv_path: Path to data CSV file.
            window: Number of samples per window.
            stride: Step between windows.
            bandpass: Low/high cutoff (Hz) for EEG filter.
        """
        super().__init__(window, stride)
        # Read DataFrame
        df = pd.read_csv(csv_path)
        self.fs = 1/df["time"][1]

        # Identify EEG/ET columns
        et_cols = ["X", "Y", "Z"]
        eeg_cols = [c for c in df.columns if c not in ["time"] + et_cols]

        # Clean channel names
        self.channel_names = eeg_cols

        # Bandpass filter EEG and convert to tensor
        inp_np = bandpass_filter(df[eeg_cols].to_numpy(), low=bandpass[0], high=bandpass[1], fs=self.fs)
        self.inp = torch.tensor(inp_np.copy(), dtype=torch.float32)
        self.in_mean = self.inp.mean(dim=0)

        # Bandpass on target eyetracking coordinates and convert to tensor
        out = bandpass_filter(
            df[et_cols].to_numpy(), low=1, high=30, fs=self.fs
        )
        out_tensor = torch.tensor(out.copy(), dtype=torch.float32)

        # Standardize target
        self.out_std = out_tensor.std(dim=0, keepdim=True).clamp(min=1e-6)
        self.out_mean = out_tensor.mean(dim=0, keepdim=True)
        self.out = (out_tensor - self.out_mean) / self.out_std
        self.inp_dim = len(eeg_cols)
        self.out_dim = len(et_cols)
