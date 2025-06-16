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

    def plot_channel_importance(self, channel_importance, scale: str = "linear", name: str = "_"):
        plt.figure(figsize=(10, 5))
        channels = self.dataset.channel_names
        plt.bar(channels, channel_importance)
        plt.xlabel('EEG Channel Index')
        plt.ylabel('Increase in MSE after Perturbation')
        plt.yscale(scale)
        plt.title('EEG Channel Importance')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(self.plotpath + f"channel_importance{name}.png")
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

    def plot_temporal_heatmap_and_reconstruction(
        self,
        start: int = 0,
        stop: int = None,
        imu_channel: int = 0,
        figsize=(15, 7),
        df: DataFrame = None,
        name: str = "_",
    ):
        T = df.shape[0]
        stop = stop or T
        stop = min(stop, T)
        df_seg = df.iloc[start:stop]

        imu_pred = self.baseline_pred.detach().numpy()[:, imu_channel]
        imu_true = self.baseline_true.detach().numpy()[:, imu_channel]
        mse = np.square(imu_pred - imu_true)[start:stop]

        x = np.arange(start, stop)

        # GridSpec for fine control
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 0.1, 2], hspace=0.3)

        ax1 = fig.add_subplot(gs[0])
        ax_cbar = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2], sharex=ax1)

        # Heatmap
        data = df_seg.T.values
        vmin = np.max([data.min(), 1e-6])
        vmax = data.max()
        im = ax1.imshow(
            data,
            aspect="auto",
            extent=[start, stop, 0, df.shape[1]],
            origin="lower",
            cmap="viridis",
            #norm=LogNorm(vmin=vmin, vmax=vmax),
        )

        for i in range(1, df.shape[1]):
            ax1.hlines(
                i,
                xmin=start,
                xmax=stop,
                colors="white",
                linestyles="dotted",
                linewidth=0.5,
            )
        ax1.set_yticks(np.arange(0.5, df.shape[1] + 0.5))
        ax1.set_yticklabels(df.columns)
        ax1.set_ylabel("EEG Channels")
        ax1.set_title("Temporal Importance Heatmap")

        # Horizontal colorbar
        cbar = fig.colorbar(im, cax=ax_cbar, orientation="horizontal")
        cbar.set_label("ΔMSE")

        # Reconstruction and MSE
        ax2.plot(x, imu_pred[start:stop], label="Reconstructed IMU", color="black")
        ax2.plot(x, imu_true[start:stop], label="True IMU", color="green")
        ax2.set_ylabel("IMU Value")
        ax2.set_xlabel("Time Step")
        ax2.grid(True)

        ax3 = ax2.twinx()
        ax3.plot(x, mse, label="ΔMSE (IMU vs True)", color="red", alpha=0.7, linestyle=":")
        ax3.set_ylabel("ΔMSE")

        # Combined legend
        lines, labels = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper right")

        plt.savefig(self.plotpath + f"channel_temporal_importance{name}.png")
        plt.show()
