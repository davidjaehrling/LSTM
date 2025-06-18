import os
from typing import Tuple, List, Union, Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import spectrogram, cwt, morlet2
from scipy.interpolate import interp1d

from src.perturbation.analyse.base_analyser import BaseAnalyser


class Plotter(BaseAnalyser):
    """
    Plotting utilities for EEG→IMU analysis.

    Provides heatmap, bar, spectrogram, timeseries, and reconstruction plots
    integrating channel/frequency importance and model reconstructions.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Any,
        loader: DataLoader[Tuple[Tensor, Tensor]],
        config: dict[str, Any],
        fs: float = 125.0,
        vis_window: Tuple[int, int] = (0, 5000),
    ) -> None:
        super().__init__(model, dataset, loader, config, fs)
        self.start, self.stop = vis_window

    def _create_figure(
        self,
        height_ratios: List[float],
        figsize: Tuple[float, float],
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Initialize figure with vertical subplots using GridSpec.

        Args:
            height_ratios: Relative heights for each row.
            figsize: Figure size (width, height).

        Returns:
            Tuple of (figure, list of axes).
        """
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            nrows=len(height_ratios), ncols=1,
            height_ratios=height_ratios, hspace=0.3)
        axes = [fig.add_subplot(gs[i, 0]) for i in range(len(height_ratios))]
        return fig, axes

    def _plot_heatmap(
        self,
        ax: plt.Axes,
        cax: plt.Axes,
        df_seg: DataFrame,
        ylabel: str,
        title: str,
        cmap: str = "viridis",
    ) -> None:
        """
        Render heatmap of DataFrame with colorbar.

        Args:
            ax: Main axes for heatmap.
            cax: Axes for horizontal colorbar.
            df_seg: Segment of DataFrame (time × channels).
            ylabel: Label for y-axis.
            title: Plot title.
            cmap: Matplotlib colormap name.
        """
        data = df_seg.T.values
        im = ax.imshow(
            data,
            aspect="auto",
            extent=(self.start, self.stop, 0, data.shape[0]),
            origin="lower",
            cmap=cmap,
        )
        # Draw horizontal separators between channels
        for i in range(1, data.shape[0]):
            ax.hlines(i, self.start, self.stop,
                      colors="white", linestyles="dotted", linewidth=0.5)
        ax.set_yticks(np.arange(0.5, data.shape[0] + 0.5))
        ax.set_yticklabels(df_seg.columns)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.colorbar(im, cax=cax, orientation="horizontal")

    def _plot_bar_importance(
        self,
        ax: plt.Axes,
        importance: List[float],
        labels: List[str],
        title: str,
        ylabel: str = "ΔMSE",
        scale: str = "linear",
    ) -> None:
        """
        Draw bar plot for importance metrics.

        Args:
            ax: Axes for bar plot.
            importance: Numeric values per label.
            labels: Category labels.
            title: Plot title.
            ylabel: Label for y-axis.
            scale: Scale for y-axis ('linear' or 'log').
        """
        ax.bar(labels, importance, color="gray")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_yscale(scale)
        ax.grid(axis="y")
        ax.tick_params(axis="x", rotation=45)

    def _plot_spectrogram(
        self,
        ax: plt.Axes,
        cax: plt.Axes,
        eeg_seg: np.ndarray,
        spec_channel: Union[int, str],
        method: str = "morlet",
        log_scale: bool = True,
    ) -> None:
        """
        Plot time–frequency representation for a given EEG segment.

        Args:
            ax: Axes for spectrogram.
            cax: Axes for colorbar.
            eeg_seg: EEG data segment (time × channels).
            spec_channel: Index or name of channel to analyze.
            method: 'stft' or 'morlet'.
            log_scale: Whether to log-scale frequency axis.
        """
        # Select channel data
        if eeg_seg.ndim == 2:
            if isinstance(spec_channel, str):
                spec_channel = self.dataset.channel_names.index(spec_channel)
            data = eeg_seg[:, spec_channel]
        else:
            data = eeg_seg

        if method == "stft":
            # Compute short-time Fourier transform
            f, t, Sxx = spectrogram(
                data, fs=self.fs, nperseg=128, noverlap=64
            )
            power = 10 * np.log10(Sxx + 1e-10)
            # Optionally interpolate to log-spaced frequencies
            if log_scale:
                f_log = np.logspace(np.log10(f[1]), np.log10(f[-1]), len(f))
                power = interp1d(f, power, axis=0,
                                 fill_value="extrapolate")(f_log)
                f = f_log
            im = ax.pcolormesh(t * self.fs, f, power, shading="auto")
        else:
            # Continuous wavelet transform with Morlet
            w = 5.0
            freqs = np.logspace(0, np.log10(30), 40) if log_scale else np.linspace(1, 30, 40)
            scales = w / (2 * np.pi * freqs)
            cwt_mat = cwt(data, morlet2, scales, w=w)
            power = 10 * np.log10(np.abs(cwt_mat) ** 2 + 1e-10)
            times = np.arange(self.start, self.stop) / self.fs
            im = ax.pcolormesh(times, freqs, power, shading="auto")

        ax.set_title(f"Spectrogram ({method}, {'log' if log_scale else 'linear'})"
                     f" Channel: {self.dataset.channel_names[spec_channel]}")
        ax.set_ylabel("Frequency (Hz)")
        plt.colorbar(im, cax=cax, orientation="horizontal").set_label("Power (dB)")
        ax.grid(True)

    def _plot_timeseries(
        self,
        ax: plt.Axes,
        eeg_seg: np.ndarray,
        channels: List[int],
    ) -> None:
        """
        Overlay multiple EEG channel waveforms.

        Args:
            ax: Axes for timeseries.
            eeg_seg: EEG data (time × channels).
            channels: List of channel indices.
        """
        times = np.arange(self.start, self.stop) / self.fs
        # Draw each channel with decreasing prominence in reverse order to stack most important ones on top
        for level, idx in enumerate(channels[::-1]):
            name = self.dataset.channel_names[idx]
            linewidth = max(1.0, 2.5 - 0.4 * level)
            intensity = min(1.0, 0.2 + 0.15 * level)
            ax.plot(times, eeg_seg[:, idx],
                    label=name, linewidth=linewidth,
                    color=(intensity, intensity, intensity))
        ax.set_title(f"EEG Timeseries: {', '.join(self.dataset.channel_names[i] for i in channels)}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (µV)")
        ax.grid(True)
        ax.legend(loc="upper right", fontsize=8)

    def _plot_reconstruction(
        self,
        ax: plt.Axes,
        imu_channel: int,
    ) -> None:
        """
        Plot reconstructed vs. true IMU signal and MSE overlay.

        Args:
            ax: Axes for reconstruction plot.
            imu_channel: Index of IMU channel.
        """
        # Extract predicted and true signals
        pred = self.baseline_pred.detach().cpu().numpy()[:, imu_channel]
        true = self.baseline_true.detach().cpu().numpy()[:, imu_channel]
        times = np.arange(self.start, self.stop) / self.fs
        # Compute MSE time course
        mse = np.square(pred - true)[self.start:self.stop]

        ax.plot(times, pred[self.start:self.stop], label="Reconstructed", linestyle="-", color="black")
        ax.plot(times, true[self.start:self.stop], label="True", linestyle="--", color="green")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("IMU value")
        ax.grid(True)

        # Secondary axis for MSE
        ax2 = ax.twinx()
        ax2.plot(times, mse, label="MSE", linestyle=":", color="red")
        ax2.set_ylabel("MSE")

        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right")

    def plot_channel_importance(
        self,
        importance: List[float],
        scale: str = "linear",
        filename: str = "channel_importance.png",
    ) -> None:
        """
        Save a bar plot of EEG channel importance.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        self._plot_bar_importance(
            ax, importance, self.dataset.channel_names,
            title="EEG Channel Importance", scale=scale
        )
        os.makedirs(self.plotpath, exist_ok=True)
        fig.savefig(os.path.join(self.plotpath, filename))
        plt.close(fig)

    def plot_band_importance(
        self,
        importance: dict[str, float],
        scale: str = "linear",
        filename: str = "band_importance.png",
    ) -> None:
        """
        Save a bar plot of EEG frequency band importance.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        labels, values = zip(*importance.items())
        self._plot_bar_importance(
            ax, list(values), list(labels),
            title="EEG Band Importance", scale=scale
        )
        os.makedirs(self.plotpath, exist_ok=True)
        fig.savefig(os.path.join(self.plotpath, filename))
        plt.close(fig)

    def plot_heatmap_and_reconstruction(
        self,
        df: DataFrame,
        imu_channel: int = 0,
        filename: str = "heatmap_recon.png",
        figsize: Tuple[int, int] = (15, 7),
    ) -> None:
        """
        Composite plot: temporal heatmap and IMU reconstruction.
        """
        df_seg = df.iloc[self.start:self.stop]
        fig, axes = self._create_figure([3, 0.1, 2], figsize)
        self._plot_heatmap(axes[0], axes[1], df_seg, "Channel", "Temporal Importance")
        self._plot_reconstruction(axes[2], imu_channel)
        os.makedirs(self.plotpath, exist_ok=True)
        fig.savefig(os.path.join(self.plotpath, filename))
        plt.close(fig)

    def plot_with_spectrogram(
        self,
        df: DataFrame,
        imu_channel: int = 0,
        spec_channel: int = 0,
        filename: str = "spectrogram_analysis.png",
        figsize=(15, 12),
    ):
        """
        Composite plot: temporal heatmap, spectogram and IMU reconstruction.
        """
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
        plt.savefig(os.path.join(self.plotpath, filename))
        plt.close()

    def plot_with_timeseries(
        self,
        df: DataFrame,
        imu_channel: int = 0,
        spec_channel: list = (0),
        filename: str = "Analysis_with_timeseries.png",
        figsize=(15, 12),
    ):
        """
        Composite plot: temporal heatmap, timeseries and IMU reconstruction.
        """
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
        plt.savefig(os.path.join(self.plotpath, filename))
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
        """
        Composite plot:
        1. Heatmap channel importance
        2. Heatmap channel frequency importance
        3. Heatmap integrated gradient
        4. Spectogram
        5. IMU reconstruction
        """
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
        plt.savefig(os.path.join(self.plotpath, filename))
        plt.close()
