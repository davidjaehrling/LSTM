import torch.nn as nn
from torch import Tensor
import torch
from torch.utils.data import DataLoader
from typing import Union

class LSTMRegressor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        config: dict,

    ):
        super().__init__()
        self.bidirectional = config["bidirectional"]
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=config["hidden_dim"],
            num_layers=config["num_layers"],
            batch_first=True,
            dropout=config["dropout"] if config["num_layers"] > 1 else 0.0,
            bidirectional=config["bidirectional"],
        )
        self.head = nn.Linear(config["hidden_dim"] * (2 if config["bidirectional"] else 1), out_dim)
        self.opt = torch.optim.Adam(self.parameters(), lr=config["lr"])
        self.crit = torch.nn.MSELoss()

    def forward(self, x: Tensor, return_states: bool =False) -> Tensor:
        out, (h, c) = self.lstm(x)
        if return_states:
            return self.head(out), h, c, out
        return self.head(out)                 # [B, out_dim]

    def train_epoch(self, loader: DataLoader):
        for x, y in loader:
            self.opt.zero_grad()
            loss = self.crit(self(x), y)
            loss.backward()
            self.opt.step()
        return loss

    def predict(self, x: Tensor, y: Tensor) -> Union[Tensor, float]:
        preds = self(x)
        loss = self.crit(preds, y)
        return  preds, y,  loss

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
