from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split


class BaseDataset(Dataset[Tuple[Tensor, Tensor]]):
    """
    Base for time-series datasets
    Provides windowing parameters and train/val/test splitting.
    """

    def __init__(
        self,
        window: int,
        stride: int,
    ) -> None:
        """
        Initialize window and stride for segmentation.

        Args:
            window: Number of time samples per segment.
            stride: Step size between windows.
        """
        self.window: int = window
        self.stride: int = stride

    @abstractmethod
    def __len__(self) -> int:
        """
        Total number of segments available.
        """
        ...  # implemented by subclasses

    @abstractmethod
    def __getitem__(
        self,
        idx: int,
    ) -> Any:
        """
        Retrieve the idx-th window of data.

        Returns:
            A tuple or array of tensors (in_window, out_window).
        """

    def train_test_val_split(
        self,
        batch_size: int,
        splits: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Split into train/validation/test DataLoaders.

        Args:
            batch_size: Batch size for all loaders.
            splits: Fractions for (train, val, test).
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        total = len(self)
        # Compute split sizes
        train_frac, val_frac, test_frac = splits
        n_train = int(total * train_frac)
        n_val = int(total * val_frac)
        n_test = total - n_train - n_val

        # Deterministic split
        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds, test_ds = random_split(
            self,
            [n_train, n_val, n_test],
            generator=generator,
        )

        # Build DataLoaders
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def destandardize(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Reverse any standardization applied to OUT data.

        Default is identity; override in subclass if needed.

        Args:
            x: Standardized OUT tensor.

        Returns:
            De-standardized OUT tensor.
        """
        return x
