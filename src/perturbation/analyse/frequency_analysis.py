import torch
from pandas import DataFrame
from torch.utils.data import DataLoader
from src.imu_recon.utils import reconstruct_signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt
from src.perturbation.analyse.base_analyser import BaseAnalyser


class FrequencyAnalyzer(BaseAnalyser):
    """
    Analyze model sensitivity to perturbations in EEG frequency bands.
    """
    # Canonical EEG bands (Hz)
    DEFAULT_BANDS = {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta':  (12, 30),
        'Gamma': (30, 50)
    }

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
    def _bandstop_filter(data: np.ndarray, low: float, high: float, fs: float, order: int = 5) -> np.ndarray:
        """
        Apply a Butterworth band-stop filter to the data.
        Args:
            data: array of shape [T, C]
            low: lower cutoff freq (Hz)
            high: higher cutoff freq (Hz)
            fs: sampling freq (Hz)
        Returns:
            filtered data same shape
        """
        nyq = 0.5 * fs
        low_norm = low / nyq
        high_norm = high / nyq
        b, a = butter(order, [low_norm, high_norm], btype='bandstop')
        return filtfilt(b, a, data, axis=0)

    def perturb_band(self, band: str):
        """
        Generate a perturbed dataset where the specified band is removed.
        """
        low, high = self.DEFAULT_BANDS[band]
        perturbed = []
        for eeg_batch, imu_batch in self.loader:
            # eeg_batch: [B, T, C]
            eeg_np = eeg_batch.numpy()
            for i in range(eeg_np.shape[0]):
                data = eeg_np[i]  # [T, C]
                filtered = self._bandstop_filter(data, low, high, self.fs).copy()
                eeg_tensor = torch.tensor(filtered, dtype=torch.float32)
                perturbed.append((eeg_tensor, imu_batch[i]))
        return DataLoader(perturbed, batch_size=self.loader.batch_size, shuffle=False)

    def analyze_band_importance(self, bands: dict = None):
        """
        Compute ΔMSE for each band removal, comparing to baseline.
        """
        bands = bands or self.DEFAULT_BANDS
        baseline_mse = torch.nn.functional.mse_loss(
            self.baseline_pred, self.baseline_true
        ).item()
        results = {}
        for name, (low, high) in bands.items():
            print(f"Perturbing band {name}: {low}-{high} Hz")
            pert_loader = self.perturb_band(name)
            pert_pred, pert_true = reconstruct_signal(self.model, pert_loader, self.dataset)
            mse = torch.nn.functional.mse_loss(pert_pred, pert_true).item()
            results[name] = mse - baseline_mse
        self.plot_band_importance(results)
        return results

    def plot_band_importance(self, results: dict, scale: str = "linear"):
        labels = list(results.keys())
        vals = [results[b] for b in labels]
        plt.figure(figsize=(8, 4))
        plt.bar(labels, vals)
        plt.xlabel('Frequency Band')
        plt.xticks(rotation=90)
        plt.ylabel('Increase in MSE after Band Removal')
        plt.yscale(scale)
        plt.title('Frequency Band Importance')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(self.plotpath + f'frequency_band_importance.png')
        plt.close()

    def compute_temporal_importance(self, bands: dict = None) -> DataFrame:
        """
        Compute temporal ΔMSE for each band removal over full test set.
        Returns a DataFrame time × band.
        """
        bands = bands or self.DEFAULT_BANDS
        # baseline MSE per timepoint
        mse_base = (
            torch.nn.functional.mse_loss(
                self.baseline_pred, self.baseline_true, reduction='none'
            )
            .mean(dim=1)
            .detach().numpy()
        )
        imp = {}
        for name in bands:
            print(f"Temporal importance for band {name}")
            loader_bs = self.perturb_band(name)
            pert_pred, _ = reconstruct_signal(self.model, loader_bs, self.dataset)
            mse_pert = (
                torch.nn.functional.mse_loss(
                    pert_pred, self.baseline_true, reduction='none'
                )
                .mean(dim=1)
                .detach().numpy()
            )
            imp[name] = mse_pert - mse_base
        df = pd.DataFrame(imp)
        return df
