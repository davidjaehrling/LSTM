import torch
from pandas import DataFrame
from torch.utils.data import DataLoader
from src.imu_recon.utils import reconstruct_signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec

from src.perturbation.analyse.base_analyser import BaseAnalyser


class ChannelAnalyzer(BaseAnalyser):
    # ----------------------------------------------------------------------
    #   UTILITIES
    # ----------------------------------------------------------------------

    def perturb_channel(self, channel_idx):
        """Creates a DataLoader with the specified channel perturbed."""
        perturbed_samples = []

        for eeg_batch, imu_batch in self.loader:
            for eeg, imu in zip(eeg_batch, imu_batch):  # Each eeg: [T, C], imu: [T, C]
                eeg_perturbed = eeg.clone()
                eeg_perturbed[:, channel_idx] = 0
                perturbed_samples.append((eeg_perturbed, imu))

        return DataLoader(
            perturbed_samples, batch_size=self.loader.batch_size, shuffle=False
        )

    # ----------------------------------------------------------------------
    #   CHANNEL IMPORTANCE
    # ----------------------------------------------------------------------

    def analyze_channel_importance(self):
        channel_importance = []
        baseline_mse = torch.nn.functional.mse_loss(self.baseline_pred, self.baseline_true).item()

        num_channels = self.loader.dataset[0][0].shape[1]

        for ch_idx in range(num_channels):
            print(f"Perturbing channel {self.dataset.channel_names[ch_idx]}")
            perturbed_loader = self.perturb_channel(ch_idx)
            perturbed_pred, perturbed_true = reconstruct_signal(self.model, perturbed_loader, self.dataset)
            perturbed_mse = torch.nn.functional.mse_loss(perturbed_pred, perturbed_true).item()

            importance = perturbed_mse - baseline_mse
            channel_importance.append(importance)

        self.plot_channel_importance(channel_importance)
        return channel_importance

    def plot_channel_importance(self, channel_importance, scale: str = "linear"):
        plt.figure(figsize=(10, 5))
        channels = self.dataset.channel_names
        plt.bar(channels, channel_importance)
        plt.xlabel('EEG Channel Index')
        plt.ylabel('Increase in MSE after Perturbation')
        plt.yscale(scale)
        plt.title('EEG Channel Importance')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(self.plotpath + f"channel_importance.png")
        plt.close()

    # ----------------------------------------------------------------------
    #   CHANNEL IMPORTANCE OVER TIME
    # ----------------------------------------------------------------------
    def compute_temporal_importance(self):
        """
        Computes temporal importance for all channels over the full test set.
        Stores result in a pandas DataFrame with time index and channel columns.
        """
        # reconstruct baseline pred already computed
        baseline_pred = self.baseline_pred
        baseline_true = self.baseline_true

        # compute baseline MSE per timestep
        mse_baseline = (
            torch.nn.functional.mse_loss(baseline_pred, baseline_true, reduction="none")
            .mean(dim=1)
            .detach()
            .numpy()
        )

        # collect perturbed predictions
        channel_imp = {}
        num_channels = len(self.dataset.channel_names)
        for ch in range(num_channels):
            print(f"Computing temporal importance for channel {self.dataset.channel_names[ch]}...")
            # create perturbed loader for this channel
            perturbed_loader = self.perturb_channel(ch)
            perturbed_pred, _ = reconstruct_signal(
                self.model, perturbed_loader, self.dataset
            )
            mse_perturbed = (
                torch.nn.functional.mse_loss(
                    perturbed_pred, baseline_true, reduction="none"
                )
                .mean(dim=1)
                .detach()
                .numpy()
            )
            # delta MSE
            channel_imp[ch] = mse_perturbed - mse_baseline

        # create DataFrame: index=time steps, columns=channel indices or names
        df = pd.DataFrame(channel_imp)
        if hasattr(self.dataset, "channel_names"):
            df.columns = self.dataset.channel_names
        return df
