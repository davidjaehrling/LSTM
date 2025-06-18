from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from src.imu_recon.utils import reconstruct_signal
from src.perturbation.analyse.base_analyser import BaseAnalyser


class ChannelAnalyzer(BaseAnalyser):
    """
    Analyze EEG channel importance for IMU signal reconstruction.

    Methods:
        perturb_channel: generate a DataLoader with channel perturbed signals.
        analyze_channel_importance: compute global MSE impact for all channels.
        compute_temporal_importance: compute MSE impact over time per channel.
    """

    def perturb_channel(self, channel_idx: int) -> DataLoader[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create a DataLoader with the specified EEG channel zeroed out.

        Args:
            channel_idx: Index of EEG channel to perturb.

        Returns:
            DataLoader yielding perturbed (eeg, imu) pairs.
        """
        perturbed_samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # Iterate through original loader, zeroing out the chosen channel
        for eeg_batch, imu_batch in self.loader:
            for eeg, imu in zip(eeg_batch, imu_batch):
                eeg_mod = eeg.clone()
                eeg_mod[:, channel_idx] = 0
                perturbed_samples.append((eeg_mod, imu))

        # Return new DataLoader with perturbed channel
        return DataLoader(perturbed_samples, batch_size=self.loader.batch_size, shuffle=False)

    def analyze_channel_importance(self) -> List[float]:
        """
        Compute per-channel increase in MSE when perturbed compared to baseline.

        Returns:
            List of MSE increases for each EEG channel.
        """
        # Baseline MSE on unperturbed data
        baseline_mse = torch.nn.functional.mse_loss(
            self.baseline_pred, self.baseline_true
        ).item()

        # Number of EEG channels: [batch, time, channels]
        num_channels = len(self.dataset.channel_names)

        importances = []
        for idx in range(num_channels):
            print(f"Perturbing channel {self.dataset.channel_names[idx]}")
            # Reconstruct with channel idx zeroed
            loader_mod = self.perturb_channel(idx)
            pred_mod, true_mod = reconstruct_signal(self.model, loader_mod, self.dataset)
            mse_mod = torch.nn.functional.mse_loss(pred_mod, true_mod).item()
            importances.append(mse_mod - baseline_mse)

        return importances


    def compute_temporal_importance(self) -> pd.DataFrame:
        """
        Compute MSE impact per EEG channel over time.

        Returns:
            DataFrame indexed by time steps with channels as columns and ΔMSE values.
        """
        # Per-timestep baseline MSE (mean over batch)
        mse_baseline = (
            torch.nn.functional.mse_loss(
                self.baseline_pred, self.baseline_true, reduction="none"
            )
            .mean(dim=1)
            .detach()
            .numpy()
        )

        temporal_imp: dict[int, np.ndarray] = {}
        num_channels = len(self.dataset.channel_names)

        for idx in range(num_channels):
            print(f"Computing temporal importance for channel {self.dataset.channel_names[idx]}")
            loader_mod = self.perturb_channel(idx)
            pred_mod, _ = reconstruct_signal(self.model, loader_mod, self.dataset)
            mse_mod = (
                torch.nn.functional.mse_loss(
                    pred_mod, self.baseline_true, reduction="none"
                )
                .mean(dim=1)
                .detach()
                .numpy()
            )
            temporal_imp[idx] = mse_mod - mse_baseline

        # Build DataFrame with time index and channel columns
        df = pd.DataFrame.from_dict(temporal_imp, orient="columns")
        df.columns = self.dataset.channel_names
        return df
