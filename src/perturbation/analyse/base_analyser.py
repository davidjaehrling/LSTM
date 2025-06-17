from torch.utils.data import DataLoader
from src.imu_recon.utils import reconstruct_signal
from pandas import DataFrame
import pickle
from typing import Union
import  os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.signal import spectrogram
from scipy.signal import morlet2




class BaseAnalyser:
    def __init__(self, model, dataset, loader: DataLoader, config: dict, fs: float = 125.0, vis_window: tuple = (0,5000)):
        self.model = model
        self.dataset = dataset
        self.loader = loader
        self.fs = fs
        self.config = config
        self.start = vis_window[0]
        self.stop = vis_window[1]
        # baseline full reconstruction
        self.baseline_pred, self.baseline_true = reconstruct_signal(
            self.model, self.loader, self.dataset
        )
        self.plotpath = f"plots/{self.config['dataset']}/"
        self.respath = f"results/{self.config['dataset']}/"
        os.makedirs(self.plotpath, exist_ok=True)
        os.makedirs(self.respath, exist_ok=True)

    def save_pickle(self, data: Union[DataFrame, dict], filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load_pickle(self, filepath: str) -> Union[DataFrame, dict]:
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def _create_figure(self, height_ratios, figsize):
        """Helper to create figure and GridSpec axes."""
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(len(height_ratios), 1, height_ratios=height_ratios, hspace=0.3)
        axes = [fig.add_subplot(gs[i]) for i in range(len(height_ratios))]
        return fig, axes

    def _plot_heatmap(self, ax, cax, df_seg, ylabel, title, cmap='viridis'):
        data = df_seg.T.values
        im = ax.imshow(
            data, aspect='auto', extent=[self.start, self.stop, 0, df_seg.shape[1]],
            origin='lower', cmap=cmap
        )
        # separators
        for i in range(1, df_seg.shape[1]):
            ax.hlines(i, self.start, self.stop, colors='white', linestyles='dotted', linewidth=0.5)
        ax.set_yticks(np.arange(0.5, df_seg.shape[1] + 0.5))
        ax.set_yticklabels(df_seg.columns)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
        return im, cbar

    def _plot_spectrogram(self, ax, cax, eeg_seg, spec_channel, fs, method='morlet'):
        """
        Plot time-frequency power for a single EEG channel segment.
        Supports STFT or Morlet CWT for better time-frequency tradeoff.
        Args:
            ax: axis to plot on
            cax: axis for colorbar
            eeg_seg: [T] 1D EEG segment for a single channel or [T,C] full segment
            spec_channel: int or channel name to select
            fs: sampling frequency
            method: 'stft' or 'morlet'
        """
        # resolve channel index if name provided
        if eeg_seg.ndim == 2:
            if not isinstance(spec_channel, int):
                try:
                    spec_channel = self.dataset.channel_names.index(spec_channel)
                except ValueError:
                    raise ValueError(f"spec_channel must be int or valid channel name, got {spec_channel}")
            data = eeg_seg[:, spec_channel]
        else:
            data = eeg_seg  # already 1D
        # choose method
        if method == 'stft':
            # classical spectrogram (STFT)
            f, t, Sxx = spectrogram(data, fs=fs, nperseg=128, noverlap=64)
            power = 10 * np.log10(Sxx + 1e-10)
            # convert time to sample index
            times = t * fs
            im = ax.pcolormesh(times, f, power, shading='gouraud', cmap='magma')
            ax.set_ylabel('Frequency (Hz)')
        else:
            # Morlet Continuous Wavelet Transform
            from scipy.signal import cwt, morlet2
            w = 6.0  # Morlet parameter
            # define frequencies of interest
            freqs = np.linspace(1, fs/2, num=50)
            # compute scales for Morlet: s = w/(2*pi*f)
            scales = w / (2 * np.pi * freqs)
            # perform CWT
            cwtmatr = cwt(data, morlet2, scales, w=w)
            power = np.abs(cwtmatr) ** 2
            # time axis as sample indices
            times = np.arange(len(data))
            im = ax.pcolormesh(times, freqs, power, shading='gouraud', cmap='magma')
            ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'Spectrogram ({method})')
        # colorbar
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_label('Power (dB)')
        ax.grid(True)
        return im, cbar

    def _plot_reconstruction(self, ax, df_seg, imu_channel):
        imu_pred = self.baseline_pred.detach().cpu().numpy()[:, imu_channel]
        imu_true = self.baseline_true.detach().cpu().numpy()[:, imu_channel]
        mse = np.square(imu_pred - imu_true)[self.start:self.stop]
        x = np.arange(self.start, self.stop)
        ax.plot(x, imu_pred[self.start:self.stop], label='Reconstructed IMU', color='black')
        ax.plot(x, imu_true[self.start:self.stop], label='True IMU', color='green')
        ax.set_ylabel('IMU Value')
        ax.set_xlabel('Time Step')
        ax.grid(True)
        ax3 = ax.twinx()
        ax3.plot(x, mse, label='MSE', color='red', linestyle=':')
        ax3.set_ylabel('MSE')
        # legend merge
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper right')
        return ax, ax3

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
        df_seg = df.iloc[self.start:self.stop]
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
        df_seg = df.iloc[self.start:self.stop]
        eeg_seg = self.dataset.eeg.detach().cpu().numpy()[self.start:self.stop]
        fig, axes = self._create_figure([2, 0.1, 2, 0.1, 2], figsize)
        # Heatmap
        self._plot_heatmap(axes[0], axes[1], df_seg, 'Channel', 'Temporal Importance Heatmap')
        # Spectrogram
        self._plot_spectrogram(axes[2], axes[3], eeg_seg, spec_channel, self.fs, method="morlet")
        # Reconstruction
        self._plot_reconstruction(axes[4], df_seg, imu_channel)
        plt.tight_layout()
        plt.savefig(self.plotpath + filename)
        plt.show()