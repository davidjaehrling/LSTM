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
    def __init__(self, model: torch.nn.Module, dataset, loader: DataLoader, config: dict):
        super().__init__(model, dataset, loader, config)
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
                                     timestep: int = -1,
                                     start: int = 0,
                                     stop: int = 5000) -> DataFrame:
        """
        Computes integrated gradients attribution for each time × channel
        on the full EEG sequence.
        Returns a DataFrame indexed by time, columns=channel names.
        """

        wrapper = LSTMWrapper(self.model, imu_channel, timestep)
        ig = IntegratedGradients(wrapper)
        dl = DeepLift(wrapper)

        # prepare full input: [1, T, C]
        sub_eeg = self.eeg_full[start:stop]
        x = sub_eeg.unsqueeze(0).requires_grad_(True)
        baseline = self.get_baseline(x, mode="zero", noise_std=0.05)
        print("Compute attributions...")
        attributions = dl.attribute(x, baseline)
        # attributions, _ = ig.attribute(
        #     x,
        #     baseline,
        #     n_steps=20,
        #     internal_batch_size=5,  # process 5 interpolation points at once
        #     return_convergence_delta=False,
        # )

        attr = attributions.squeeze(0).detach().numpy()  # [T, C]
        df = pd.DataFrame(attr, columns=self.dataset.channel_names)
        return df


    def plot_model_level_heatmap_and_reconstruction(
        self,
        df: DataFrame,
        start: int = 0,
        stop: int = None,
        imu_channel: int = 0,
        figsize=(15, 7),
        name: str = "_",
        title: str = "",
    ):
        """
        Plot a temporal heatmap of model-level attributions (IG or input gradients)
        alongside the reconstructed and true IMU signal and its error.
        """
        # clamp df
        df_seg = df.copy()

        # prepare IMU data
        imu_pred = self.baseline_pred.detach().cpu().numpy()[:, imu_channel]
        imu_true = self.baseline_true.detach().cpu().numpy()[:, imu_channel]
        mse = np.square(imu_pred - imu_true)[start:stop]

        x = np.arange(start, stop)

        # layout
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 0.1, 2], hspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax_cbar = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2], sharex=ax1)

        # heatmap
        data = df_seg.T.values
        vmin = np.max([data.min(), 1e-6])
        vmax = data.max()
        im = ax1.imshow(
            data,
            aspect="auto",
            extent=[start, stop, 0, df.shape[1]],
            origin="lower",
            cmap="viridis",
        )
        # separators
        for i in range(1, df.shape[1]):
            ax1.hlines(i, start, stop, colors="white", linestyles="dotted", linewidth=0.5)
        ax1.set_yticks(np.arange(0.5, df.shape[1] + 0.5))
        ax1.set_yticklabels(df.columns)
        ax1.set_ylabel("Channel")
        ax1.set_title(f"{title}")

        # colorbar
        cbar = fig.colorbar(im, cax=ax_cbar, orientation="horizontal")
        cbar.set_label("Attribution")

        # reconstruction & mse
        ax2.plot(x, imu_pred[start:stop], label="Reconstructed IMU", color="black")
        ax2.plot(x, imu_true[start:stop], label="True IMU", color="green")
        ax2.set_ylabel("IMU")
        ax2.set_xlabel("Time step")
        ax2.grid(True)

        ax3 = ax2.twinx()
        ax3.plot(x, mse, label="ΔError", color="red", linestyle=":")
        ax3.set_ylabel("MSE")

        # legend
        l1, lab1 = ax2.get_legend_handles_labels()
        l2, lab2 = ax3.get_legend_handles_labels()
        ax2.legend(l1 + l2, lab1 + lab2, loc="upper right")

        plt.tight_layout()
        plt.savefig(self.plotpath + f"model_level{name}.png")
        plt.show()
