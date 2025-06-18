from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.signal import butter, filtfilt

from src.utils.utils import reconstruct_signal
from src.perturbation.analyse.base_analyser import BaseAnalyser


class FrequencyAnalyzer(BaseAnalyser):
    """
    Analyze model sensitivity to perturbations in EEG frequency bands.

    Methods:
        _bandstop_filter: Apply Butterworth band-stop filter to data.
        perturb_band: Remove specific frequency band by applying _bandstop_filter.
        analyze_band_importance: Compute global ΔMSE per band.
        compute_temporal_importance: Time-resolved ΔMSE per band.
    """

    # Default EEG bands (Hz)
    DEFAULT_BANDS = {
        "low Delta (1-2Hz)": (1, 2),
        "high Delta (2-4Hz)": (2, 4),
        "low Theta (4-6-2Hz)": (4, 6),
        "high Theta (6-8Hz)": (6, 8),
        "low Alpha (8-10Hz)": (8, 10),
        "high Alpha (10-12Hz)": (10, 12),
        "Beta (12-30Hz)": (12, 30),
        "Gamma (30-50Hz)": (30, 50),
    }

    @staticmethod
    def _bandstop_filter(
        data: np.ndarray,
        low: float,
        high: float,
        fs: float,
        order: int = 5,
    ) -> np.ndarray:
        """
        Apply a Butterworth band-stop filter data.

        Args:
            data: Array of shape (T, C) where T samples, C channels.
            low: Lower cutoff frequency in Hz.
            high: Upper cutoff frequency in Hz.
            fs: Sampling frequency in Hz.
            order: Filter order.

        Returns:
            Filtered data array of same shape.
        """
        # Normalize cutoff frequencies
        nyquist = 0.5 * fs
        low_norm = low / nyquist
        high_norm = high / nyquist
        # Design band-stop filter
        b, a = butter(order, [low_norm, high_norm], btype="bandstop")
        # Apply filtering along time axis
        return filtfilt(b, a, data, axis=0)

    def perturb_band(
        self,
        band: str,
    ) -> DataLoader[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Zero out a specific EEG frequency band by applying band-stop filter.

        Args:
            band: Name of the band to remove (key in DEFAULT_BANDS).

        Returns:
            DataLoader yielding perturbed (eeg, target) pairs.
        """
        # Determine frequency range for band removal
        low, high = self.DEFAULT_BANDS[band]
        perturbed_samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # Iterate through dataset batches
        for inp_batch, out_batch in self.loader:
            # Convert batch to NumPy for filtering
            inp_np = inp_batch.numpy()  # shape: [B, T, C]
            for idx in range(inp_np.shape[0]):
                # Extract single trial data
                trial_data = inp_np[idx]  # shape: [T, C]
                # Remove band via band-stop filter
                filtered = self._bandstop_filter(trial_data, low, high, self.fs)
                # Convert back to tensor
                inp_tensor = torch.tensor(filtered.copy(), dtype=torch.float32)
                perturbed_samples.append((inp_tensor, out_batch[idx]))

        # Return new DataLoader with same batch size
        return DataLoader(
            perturbed_samples,
            batch_size=self.loader.batch_size,
            shuffle=False,
        )

    def analyze_band_importance(
        self,
    ) -> Dict[str, float]:
        """
        Compute increase in MSE for each band removal compared to baseline.

        Returns:
            Dictionary mapping band names to ΔMSE values.
        """
        bands = self.DEFAULT_BANDS
        # Baseline MSE on full EEG
        baseline_mse = torch.nn.functional.mse_loss(
            self.baseline_pred, self.baseline_true
        ).item()

        results: Dict[str, float] = {}
        for name, (low, high) in bands.items():
            print(f"Perturbing band {name}: {low}-{high} Hz")
            # Generate perturbed data loader
            pert_loader = self.perturb_band(name)
            # Reconstruct and compute MSE
            pert_pred, pert_true = reconstruct_signal(
                self.model, pert_loader, self.dataset
            )
            mse = torch.nn.functional.mse_loss(pert_pred, pert_true).item()
            results[name] = mse - baseline_mse

        return results

    def compute_temporal_importance(
        self,
    ) -> DataFrame:
        """
        Compute MSE impact per band removal over time.

        Returns:
            pandas DataFrame with time indices as rows and bands as columns.
        """
        bands = self.DEFAULT_BANDS
        # Baseline MSE per time step (averaged over samples)
        mse_base = (
            torch.nn.functional.mse_loss(
                self.baseline_pred,
                self.baseline_true,
                reduction="none",
            )
            .mean(dim=1)
            .detach()
            .cpu()
            .numpy()
        )

        temporal_data: Dict[str, np.ndarray] = {}
        for name in bands:
            print(f"Computing temporal importance for band {name}")
            loader_bs = self.perturb_band(name)
            pert_pred, _ = reconstruct_signal(
                self.model, loader_bs, self.dataset
            )
            # MSE per time for perturbed
            mse_pert = (
                torch.nn.functional.mse_loss(
                    pert_pred,
                    self.baseline_true,
                    reduction="none",
                )
                .mean(dim=1)
                .detach()
                .cpu()
                .numpy()
            )
            # ΔMSE time series
            temporal_data[name] = mse_pert - mse_base

        # Construct DataFrame: rows=time, columns=bands
        df = pd.DataFrame(temporal_data)
        return df
