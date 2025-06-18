from torch.utils.data import DataLoader
from src.imu_recon.utils import reconstruct_signal
from pandas import DataFrame
import pickle
from typing import Union
import  os



class BaseAnalyser:
    def __init__(self, model, dataset, loader: DataLoader, config: dict, fs: float = 125.0, vis_window: tuple = (0,5000)):
        self.model = model
        self.dataset = dataset
        self.loader = loader
        self.fs = fs
        self.config = config
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
