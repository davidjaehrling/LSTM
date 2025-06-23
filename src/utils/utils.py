# src/lstm_train/utils.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import Tensor
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def plot_reconstruction(
    true: Tensor,
    pred: Tensor,
    channel: int = 0,
    timesteps: Optional[Tuple[int, int]] = None,
    title: str = "Reconstruction of Target-Output",
) -> plt.Figure:
    """
    Create a matplotlib Figure comparing predicted and true Output signals.

    Args:
        true (Tensor): shape [T, C] or [C], true Output data.
        pred (Tensor): same shape as true, predicted Output data.
        channel (int): which channel (axis) to plot.
        timesteps (int, optional): truncate to first N timesteps.
        title (str): plot title.

    Returns:
        fig (plt.Figure): Matplotlib figure object.
    """
    if true.ndim == 2:
        true = true[:, channel]
        pred = pred[:, channel]

    if timesteps is not None:
        true = true[timesteps[0]:timesteps[1]]
        pred = pred[timesteps[0]:timesteps[1]]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(true.cpu().numpy(), label="True", linewidth=2)
    ax.plot(pred.cpu().numpy(), label="Predicted", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized Target")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def reconstruct_signal(
    net: torch.nn.Module,
    loader: DataLoader,
    base_ds
) -> tuple[Tensor, Tensor]:
    """
    Stitch window-wise predictions from `loader` back into one continuous signal.

    Returns
    -------
    full_pred : Tensor  [seq_len, out_dim]  – de-standardised prediction
    full_true : Tensor  [seq_len, out_dim]  – de-standardised ground truth
    """
    # Example prediction to get sequence shapes
    x0, y0 = next(iter(loader))
    preds, true, _ = net.predict(x0, y0)


    window  = preds.shape[1]
    stride  = window // 2
    out_dim = base_ds.out.shape[1]

    # Determine first & last window indices covered by loader
    first_w = 0
    last_w  = len(loader.dataset) - 1

    seq_len = (last_w - first_w) * stride + window       # length of segment we reconstruct

    # Buffers for overlap-add stitching
    pred_sum   = torch.zeros(seq_len, out_dim)
    pred_count = torch.zeros(seq_len, 1)

    true_sum = torch.zeros(seq_len, out_dim)

    net.eval()
    win_ptr = 0                                          # counts windows we’ve seen so far
    with torch.no_grad():
        for x_batch, y_batch in loader:                        # loader must have shuffle=False
            preds, true, _ = net.predict(x_batch, y_batch)                         # [B, T, imu_dim]
            preds = base_ds.destandardize(preds)     # de-standardise
            true = base_ds.destandardize(true)


            bsz, _, _ = preds.shape
            for i in range(bsz):
                # Map window index → local start in this test segment
                global_w_idx = win_ptr
                half = window // 2
                local_start = (global_w_idx - first_w) * stride + half
                local_end = local_start + half
                pred_half = preds[i, half:]  # shape [half, imu_dim]

                pred_sum[local_start:local_end] += pred_half
                pred_count[local_start:local_end] += 1

                true_sum[local_start:local_end] += true[i, half:]

                win_ptr += 1

    full_pred = pred_sum / pred_count.clamp(min=1)
    full_true = true_sum / pred_count.clamp(min=1)
    return full_pred, full_true


def bandpass_filter(data: np.ndarray, low: float, high: float, fs: float, order: int = 5) -> np.ndarray:
    nyquist = 0.5 * fs
    low /= nyquist
    high /= nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)
