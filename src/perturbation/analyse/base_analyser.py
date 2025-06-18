import os
import pickle
from typing import Any, Dict, Tuple, Union

import torch
from torch.utils.data import DataLoader
from pandas import DataFrame

from src.imu_recon.utils import reconstruct_signal


class BaseAnalyser:
    """
    Base class for perturbation and model-level analysis of EEG→IMU reconstruction models.

    Attributes:
        model: Trained PyTorch model for reconstruction.
        dataset: Dataset providing EEG and IMU data.
        loader: DataLoader to load Data in batches and for train/test/val split
        fs: Sampling frequency of EEG/IMU signals (Hz).
        config: Model configuration dictionary.
        baseline_pred: Model predictions on unperturbed data.
        baseline_true: Ground-truth IMU signals.
        plotpath: Directory path for saving plots.
        respath: Directory path for saving results.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Any,
        loader: DataLoader,
        config: Dict[str, Any],
        fs: float = 125.0,
    ) -> None:
        # Initialize core attributes
        self.model = model
        self.dataset = dataset
        self.loader = loader
        self.fs = fs
        self.config = config

        # Perform baseline full-signal reconstruction
        self.baseline_pred, self.baseline_true = reconstruct_signal(
            self.model, self.loader, self.dataset
        )

        # Prepare directories for outputs
        dataset_id = str(self.config.get("dataset", "default"))
        self.plotpath = os.path.join("plots", dataset_id)
        self.respath = os.path.join("results", dataset_id)
        os.makedirs(self.plotpath, exist_ok=True)  # ensure directories exist
        os.makedirs(self.respath, exist_ok=True)

    def save_pickle(
        self,
        data: Union[DataFrame, Dict[str, Any]],
        filepath: str,
    ) -> None:
        """
        Serialize and save DataFrame or dictionary to a pickle file.

        Args:
            data: pandas DataFrame or result dictionary to pickle.
            filepath: File path where pickle will be stored.
        """
        with open(os.path.join(self.respath, filepath), 'wb') as f:
            pickle.dump(data, f)

    def load_pickle(self, filepath: str) -> Union[DataFrame, Dict[str, Any]]:
        """
        Load and return pickled DataFrame or dictionary from file.

        Args:
            filepath: Path to the pickle file.

        Returns:
            Unpickled pandas DataFrame or result dictionary.
        """
        with open(os.path.join(self.respath, filepath), 'rb') as f:
            return pickle.load(f)
