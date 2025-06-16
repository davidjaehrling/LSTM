from torch.utils.data import DataLoader
from src.imu_recon.utils import reconstruct_signal
from pandas import DataFrame
import pickle
from typing import Union
import  os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec


class BaseAnalyser:
    def __init__(self, model, dataset, loader: DataLoader, config: dict, fs: float = 125.0):
        self.model = model
        self.dataset = dataset
        self.loader = loader
        self.fs = fs
        self.config = config
        # baseline full reconstruction
        self.baseline_pred, self.baseline_true = reconstruct_signal(
            self.model, self.loader, self.dataset
        )
        self.plotpath =     f"plots/{self.config['dataset']}/"
        self.respath =      f"results/{self.config['dataset']}/"

        os.makedirs(self.plotpath, exist_ok=True)
        os.makedirs(self.respath, exist_ok=True)


    def save_pickle(self, data: Union[DataFrame, dict], filepath: str):
        """
        Saves a DataFrame or dict to disk using pickle.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load_pickle(self, filepath: str) -> Union[DataFrame, dict]:
        """
        Loads a DataFrame or dict from disk using pickle.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data

