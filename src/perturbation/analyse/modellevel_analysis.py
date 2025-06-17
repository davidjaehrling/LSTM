import torch
from torch import Tensor
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients, DeepLift
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from pandas import DataFrame

from src.imu_recon.utils import reconstruct_signal
from src.perturbation.analyse.base_analyser import BaseAnalyser


class LSTMWrapper(torch.nn.Module):
    """
    Wrapper to extract a single IMU channel at a given timestep for attribution.
    """
    def __init__(self, model: torch.nn.Module, imu_channel: int = 0, timestep: int = -1):
        super().__init__()
        self.model = model
        self.imu_channel = imu_channel
        self.timestep = timestep

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, C]
        out = self.model(x)  # [B, T, D]
        # select the target timestep and channel
        # return out[:, self.timestep, self.imu_channel]
        return out[:, :, self.imu_channel].mean(dim=1)


class ModelLevelAnalysis(BaseAnalyser):
    """
    Perform model-level explainability using Integrated Gradients and Input Gradients
    on the entire EEG sequence, then visualize heatmaps aligned with IMU reconstruction.
    """
    def __init__(self, model: torch.nn.Module, dataset, loader: DataLoader, config: dict, vis_window: tuple):
        super().__init__(model, dataset, loader, config, vis_window)
        # use full EEG sequence for analysis
        self.eeg_full  = dataset.eeg.clone().detach()

    def get_baseline(self, x: torch.Tensor, mode="zero", noise_std=0.0) -> torch.Tensor:
        """
        Returns a baseline input of same shape as `x`.

        Args:
            x: shape [1, T, C]
            mode: 'zero', 'mean', or 'mean+noise'
            noise_std: standard deviation for Gaussian noise

        Returns:
            baseline: Tensor of shape [1, T, C]
        """
        if mode == "zero":
            return torch.zeros_like(x)
        elif mode == "mean":
            base = self.dataset.mean_eeg[None, None, :]  # shape [1, 1, C]
            return base.expand_as(x)
        elif mode == "mean+noise":
            base = self.dataset.mean_eeg[None, None, :].expand_as(x)
            noise = torch.randn_like(x) * noise_std
            return base + noise
        else:
            raise ValueError(f"Unknown baseline mode: {mode}")

    def compute_integrated_gradients(self,
                                     imu_channel: int = 0,
                                     timestep: int = -1,) -> pd.DataFrame:
        """
        Computes integrated gradients attribution for each time × channel
        on the full EEG sequence, and pads the result into full time length.
        Returns a DataFrame indexed by time, columns=channel names.
        """
        wrapper = LSTMWrapper(self.model, imu_channel, timestep)
        ig = IntegratedGradients(wrapper)
        dl = DeepLift(wrapper)

        # prepare full input: [1, T_partial, C]
        sub_eeg = self.eeg_full[self.start:self.stop]
        x = sub_eeg.unsqueeze(0).requires_grad_(True)
        baseline = self.get_baseline(x, mode="zero", noise_std=0.05)

        print("Compute attributions...")
        # attributions = dl.attribute(x, baseline)  # shape: [1, T_partial, C]
        attributions = ig.attribute(
            x,
            baseline,
            n_steps=20,
            internal_batch_size=5,
            return_convergence_delta=False,
        )

        attr = attributions.squeeze(0).detach().numpy()  # shape: [T_partial, C]

        # Create full-length DataFrame with NaNs
        T_full = self.eeg_full.shape[0]
        df_full = pd.DataFrame(
            data=np.nan,
            index=np.arange(T_full),
            columns=self.dataset.channel_names
        )
        df_partial = pd.DataFrame(attr, columns=self.dataset.channel_names)
        df_full.iloc[self.start:self.stop] = df_partial.values

        return df_full
