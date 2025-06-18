from src.perturbation.analyse.base_analyser import BaseAnalyser
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.signal import spectrogram, cwt, morlet2
from scipy.interpolate import interp1d
from pandas import DataFrame
import torch
from torch.utils.data import DataLoader


class Plotter(BaseAnalyser):
    def __init__(self, model: torch.nn.Module, dataset, loader: DataLoader, config: dict, fs: float = 125.0, vis_window: tuple = (0,5000)):
        super().__init__(model, dataset, loader, config, fs)
        self.start = vis_window[0]
        self.stop = vis_window[1]

    def _create_figure(self, height_ratios, figsize):
        """Helper to create figure and GridSpec axes."""
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            len(height_ratios), 1, height_ratios=height_ratios, hspace=0.3
        )
        axes = [fig.add_subplot(gs[i]) for i in range(len(height_ratios))]
        return fig, axes

    def _plot_heatmap(self, ax, cax, df_seg, ylabel, title, cmap="viridis"):
        data = df_seg.T.values
        im = ax.imshow(
            data,
            aspect="auto",
            extent=[self.start, self.stop, 0, df_seg.shape[1]],
            origin="lower",
            cmap=cmap,
        )
        # separators
        for i in range(1, df_seg.shape[1]):
            ax.hlines(
                i,
                self.start,
                self.stop,
                colors="white",
                linestyles="dotted",
                linewidth=0.5,
            )
        ax.set_yticks(np.arange(0.5, df_seg.shape[1] + 0.5))
        ax.set_yticklabels(df_seg.columns)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
        return im, cbar

    def _plot_bar_importance(
        self, ax, importance, labels, title, ylabel="ΔMSE", scale="linear"
    ):
        ax.bar(labels, importance, color="gray")
        ax.set_ylabel(ylabel)
        ax.set_yscale(scale)
        ax.set_title(title)
        ax.grid(axis="y")
        ax.tick_params(axis="x", rotation=45)

    def _plot_spectrogram(
        self, ax, cax, eeg_seg, spec_channel, method="morlet", log_scale=True
    ):

        # resolve channel index if name provided
        if eeg_seg.ndim == 2:
            if not isinstance(spec_channel, int):
                try:
                    spec_channel = self.dataset.channel_names.index(spec_channel)
                except ValueError:
                    raise ValueError(
                        f"spec_channel must be int or valid channel name, got {spec_channel}"
                    )
            data = eeg_seg[:, spec_channel]
        else:
            data = eeg_seg

        if method == "stft":
            nperseg = 128
            noverlap = 64
            f, t, Sxx = spectrogram(
                data, fs=self.fs, nperseg=nperseg, noverlap=noverlap
            )
            power = 10 * np.log10(Sxx + 1e-10)
            times = t * self.fs

            if log_scale:
                f_log = np.logspace(np.log10(f[1]), np.log10(f[-1]), len(f))

                interp_func = interp1d(
                    f, power, axis=0, bounds_error=False, fill_value="extrapolate"
                )
                power = interp_func(f_log)
                f = f_log
            im = ax.pcolormesh(times, f, power, shading="auto", cmap="magma")

        else:
            w = 5.0
            if log_scale:
                freqs = np.logspace(np.log10(1), np.log10(30), num=40)
            else:
                freqs = np.linspace(1, 30, num=40)
            scales = w / (2 * np.pi * freqs)
            cwtmatr = cwt(data, morlet2, scales, w=w)
            power = 10 * np.log10(np.abs(cwtmatr) ** 2 + 1e-10)
            times = np.arange(self.start, self.stop, 1)
            im = ax.pcolormesh(times, freqs, power, shading="auto", cmap="magma")

        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(
            f'Spectrogram ({method}, {"log" if log_scale else "linear"}) Channel ({self.dataset.channel_names[spec_channel]})'
        )
        cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label("Power (dB)")
        ax.grid(True)
        return im, cbar

    def _plot_timeseries(self, ax, eeg_seg, channel):
        """
        Plot the time series of a single EEG channel over the [start, stop] window.

        Args:
            ax: matplotlib axis to plot on
            eeg_seg: [T, C] array of EEG data (T = time, C = channels)
            channel: int or str, channel index or name
        """
        channel_names = [self.dataset.channel_names[i] for i in channel]

        # Extract time window and data
        times = np.arange(self.start, self.stop, 1) / self.fs
        # Plot
        for i, (idx, name) in reversed(list(enumerate(zip(channel, channel_names)))):
            data = eeg_seg[:, idx]
            linewidth = max(1, 2.5 - 0.4 * i)  # thinner for later channels
            gray_level = min(1, 0.2 + 0.15 * i)  # less bright for later channels
            color = (gray_level, gray_level, gray_level)
            ax.plot(times, data, label=name, linewidth=linewidth, color=color)
        ax.set_title(f"EEG Timeseries - Channel: {channel_names})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (µV)")
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=8)

    def _plot_reconstruction(self, ax, imu_channel):
        imu_pred = self.baseline_pred.detach().cpu().numpy()[:, imu_channel]
        imu_true = self.baseline_true.detach().cpu().numpy()[:, imu_channel]
        mse = np.square(imu_pred - imu_true)[self.start : self.stop]
        times = np.arange(self.start, self.stop) / self.fs
        ax.plot(
            times,
            imu_pred[self.start : self.stop],
            label="Reconstructed IMU",
            color="black",
        )
        ax.plot(
            times, imu_true[self.start : self.stop], label="True IMU", color="green"
        )
        ax.set_ylabel("IMU Value")
        ax.set_xlabel("Time Step")
        ax.grid(True)
        ax3 = ax.twinx()
        ax3.plot(times, mse, label="MSE", color="red", linestyle=":")
        ax3.set_ylabel("MSE")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right")
        return ax, ax3

    def plot_channel_importance(
        self, channel_importance, scale="linear", filename="channel_importance.png"
    ):
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = self.dataset.channel_names
        self._plot_bar_importance(
            ax, channel_importance, labels, title="EEG Channel Importance", scale=scale
        )
        plt.tight_layout()
        plt.savefig(self.plotpath + filename)
        plt.close()

    def plot_band_importance(
        self, band_importance, scale="linear", filename="band_importance.png"
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = list(band_importance.keys())
        values = list(band_importance.values())
        self._plot_bar_importance(
            ax, values, labels, title="EEG Band Importance", scale=scale
        )
        plt.tight_layout()
        plt.savefig(self.plotpath + filename)
        plt.close()

    def plot_heatmap_and_reconstruction(
        self,
        df: DataFrame,
        imu_channel: int = 0,
        ylabel: str = "Channel",
        colorbar_label: str = "ΔMSE",
        title: str = "Temporal Importance Heatmap",
        filename: str = "heatmap_plot.png",
        figsize=(15, 7),
    ):
        df_seg = df.iloc[self.start : self.stop]
        fig, axes = self._create_figure([3, 0.1, 2], figsize)
        # Heatmap
        self._plot_heatmap(axes[0], axes[1], df_seg, ylabel, title)
        # Reconstruction
        self._plot_reconstruction(axes[2], df_seg, imu_channel)
        plt.tight_layout()
        plt.savefig(self.plotpath + filename)
        plt.show()

    def plot_with_spectrogram(
        self,
        df: DataFrame,
        imu_channel: int = 0,
        spec_channel: int = 0,
        filename: str = "spectrogram_analysis.png",
        figsize=(15, 12),
    ):
        df_seg = df.iloc[self.start : self.stop]
        eeg_seg = self.dataset.eeg.detach().numpy()[self.start : self.stop]
        fig, axes = self._create_figure([2, 0.1, 2, 0.1, 2], figsize)
        # Heatmap
        self._plot_heatmap(
            axes[0], axes[1], df_seg, "Channel", "Temporal Importance Heatmap"
        )
        # Spectrogram
        self._plot_spectrogram(
            axes[2], axes[3], eeg_seg, spec_channel[0], method="morlet", log_scale=False
        )
        # Reconstruction
        self._plot_reconstruction(axes[4], imu_channel)
        plt.tight_layout()
        plt.savefig(self.plotpath + filename)
        plt.show()

    def plot_with_timeseries(
        self,
        df: DataFrame,
        imu_channel: int = 0,
        spec_channel: list = (0),
        filename: str = "Analysis_with_timeseries.png",
        figsize=(15, 12),
    ):
        df_seg = df.iloc[self.start : self.stop]
        eeg_seg = self.dataset.eeg.detach().numpy()[self.start : self.stop]
        fig, axes = self._create_figure([2, 0.1, 2, 2], figsize)
        # Heatmap
        self._plot_heatmap(
            axes[0], axes[1], df_seg, "Channel", "Temporal Importance Heatmap"
        )
        # Timeseries
        self._plot_timeseries(axes[2], eeg_seg, spec_channel)
        # Reconstruction
        self._plot_reconstruction(axes[3], imu_channel)
        plt.tight_layout()
        plt.savefig(self.plotpath + filename)
        plt.close()

    def plot_combined(
        self,
        df_ch_imp: DataFrame,
        df_freq_imp: DataFrame,
        df_ig: DataFrame,
        filename: str = "combined.png",
        spec_channel: list = (0),
        figsize=(15, 20),
    ):

        df_ch_imp_seg = df_ch_imp.iloc[self.start : self.stop]
        df_freq_imp_seg = df_freq_imp.iloc[self.start : self.stop]
        df_ig_seg = df_ig.iloc[self.start : self.stop]
        eeg_seg = self.dataset.eeg.detach().numpy()[self.start : self.stop]
        fig, axes = self._create_figure([2, 0.1, 2, 0.1, 2, 0.1, 2, 2], figsize)
        # Heatmap Channel Importance
        self._plot_heatmap(
            axes[0], axes[1], df_ch_imp_seg, "Channel", "Temporal Importance Heatmap"
        )
        # Heatmap Frequency Importance
        self._plot_heatmap(axes[2], axes[3], df_freq_imp_seg, "Channel", "")
        # Heatmap Integrated Gradients
        self._plot_heatmap(axes[4], axes[5], df_ig_seg, "Channel", "")
        # Timeseries
        self._plot_timeseries(axes[6], eeg_seg, spec_channel)
        # Reconstruction
        self._plot_reconstruction(axes[7], 0)
        plt.tight_layout()
        plt.savefig(self.plotpath + filename)
        plt.close()
