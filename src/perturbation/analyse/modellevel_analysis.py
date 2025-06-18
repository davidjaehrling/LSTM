import os
from typing import Any, Dict, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pandas import DataFrame
from captum.attr import IntegratedGradients

from src.perturbation.analyse.base_analyser import BaseAnalyser


class LSTMWrapper(torch.nn.Module):
    """
    Wrap LSTM-based model to target and mean a specific Target channel's output for attribution.

    Attributes:
        model: Underlying PyTorch model.
        target_ch: Index of Target channel to attribute.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        target_ch: int = 0,
    ) -> None:
        super().__init__()
        self.model = model
        self.target_ch = target_ch

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass to extract target output for attribution.

        Args:
            x: Input of shape [B, T, C].

        Returns:
            Tensor of shape [B] representing aggregated Target channel output.
        """
        # Model returns [B, T, D]
        output = self.model(x)
        # Select channel across time: average over timesteps
        return output[:, :, self.target_ch].mean(dim=1)


class ModelLevelAnalysis(BaseAnalyser):
    """
    Explain Input->Output reconstruction at model level using attribution methods from captum.

    Attributes:
        inp_full: Full Input sequence for analysis.
        start: Start index for investigated Window.
        stop: End index for investigated Window.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Any,
        loader: DataLoader,
        config: Dict[str, Any],
        fs: float = 125.0,
        vis_window: Tuple[int, int] = (0, 5000),
    ) -> None:
        # Initialize parent and compute baseline reconstructions
        super().__init__(model, dataset, loader, config, fs)
        # Store full Input and visualization window
        self.inp_full = dataset.inp.clone().detach()
        self.start, self.stop = vis_window

    def get_baseline(
        self,
        x: Tensor,
        mode: str = "zero",
        noise_std: float = 0.0,
    ) -> Tensor:
        """
        Create a baseline input for attribution.

        Args:
            x: Input tensor shape [1, T, C].
            mode: 'zero', 'mean', or 'mean+noise'.
            noise_std: Std dev for Gaussian noise if mode includes noise.

        Returns:
            Baseline tensor same shape as x.
        """
        # zero will set baseline to 0
        if mode == "zero":
            return torch.zeros_like(x)
        # mean will set baseline to mean of the whole dataset
        if mode == "mean":
            # Expand dataset mean over batch and time
            base = self.dataset.mean_inp[None, None, :]
            return base.expand_as(x)
        # mean will set baseline to mean of the whole dataset and adds random noise
        if mode == "mean+noise":
            base = self.dataset.mean_inp[None, None, :].expand_as(x)
            noise = torch.randn_like(x) * noise_std
            return base + noise
        raise ValueError(f"Unknown baseline mode: {mode}")

    def compute_integrated_gradients(
        self,
        target_ch: int = 0,
        n_steps: int = 20,
        internal_bs: int = 5,
    ) -> DataFrame:
        """
        Compute Integrated Gradients attributions for EEG inputs.

        Args:
            target_ch: Target channel index.
            n_steps: Number of IG steps.
            internal_bs: Batch size for IG internal computation.

        Returns:
            DataFrame of shape [T_full, C] with attributions.
        """
        # Wrap model to target specific output
        wrapper = LSTMWrapper(self.model, target_ch)
        ig = IntegratedGradients(wrapper)

        # Extract partial EEG window for attribution
        sub_inp = self.inp_full[self.start : self.stop]
        x = sub_inp.unsqueeze(0).requires_grad_(True)
        # Baseline input
        baseline = self.get_baseline(x, mode="zero", noise_std=0.05)

        print("Computing Integrated Gradients...")
        attributions = ig.attribute(
            x,
            baseline,
            n_steps=n_steps,
            internal_batch_size=internal_bs,
            return_convergence_delta=False,
        )
        # attributions shape [1, T_partial, C]
        attr_np = attributions.squeeze(0).detach().cpu().numpy()

        # Build full-length DataFrame, filling outside window with NaN
        T_full = self.inp_full.shape[0]
        df_full = pd.DataFrame(
            np.nan,
            index=np.arange(T_full),
            columns=self.dataset.channel_names,
        )
        df_partial = pd.DataFrame(attr_np, columns=self.dataset.channel_names)
        df_full.iloc[self.start : self.stop] = df_partial.values
        return df_full

