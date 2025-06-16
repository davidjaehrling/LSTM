# src/imu_recon/models/lstm_tokenized_emb.py
from __future__ import annotations
from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from src.imu_recon.tokenizer import TOTEM
from src.imu_recon.models import LSTMRegressor


class TokenizedEmbeddingLSTMRegressor(nn.Module):
    """
    EEG window  ->  TOTEM tokenizer (frozen)  ->  code-book embeddings
                 ->  LSTM  ->  IMU reconstruction
    """

    def __init__(
        self,
        config: dict,
        out_dim: int,
    ):
        super().__init__()
        totem_ckpt = "tokenizer/vqvae_model.pth"

        # 1)  load the pretrained tokenizer (VQ-VAE)
        self.tokenizer = TOTEM(str(totem_ckpt))   # keeps full VQ-VAE inside
        self.tokenizer.vqvae.eval()
        for p in self.tokenizer.vqvae.parameters():
            p.requires_grad = False                             # encoder stays frozen
        self.compression_factor = self.tokenizer.vqvae.compression_factor
        # 2)  create an nn.Embedding that shares the code-book weights
        codebook_weight: Tensor = self.tokenizer.vqvae.vq._embedding.weight  # [N, D]
        self.codebook = nn.Embedding.from_pretrained(
            codebook_weight, freeze=True
        )
        emb_dim = codebook_weight.shape[1]

        # 3)  downstream LSTM
        self.lstm = LSTMRegressor(
            in_dim  = emb_dim,
            config = config,
            out_dim = out_dim,
        )

        self.opt = torch.optim.Adam(self.parameters(), lr=config["lr"])
        self.crit = torch.nn.MSELoss()

    @staticmethod
    def _tokenize_batch(
        tokenizer: TOTEM, batch_eeg: Tensor
    ) -> Tensor:
        """
        Slow but simple python-loop → token indices tensor.

        Args
        ----
        batch_eeg : [B, T, C] (torch.float32, on CPU or GPU)

        Returns
        -------
        idx : LongTensor [B, T']  (token indices)
        """
        idx_list: List[Tensor] = []

        # We’ll collapse EEG channels to 1D by averaging (⇢ shape [T])
        # Feel free to replace by first channel or PCA etc.
        for eeg in batch_eeg:                            # iterate over batch (B)
            signal_1d = eeg.mean(dim=1).cpu().numpy()    # -> np.ndarray [T]
            indices   = tokenizer.tokenize(
                torch.tensor(signal_1d).unsqueeze(0)     # Tokenizer expects [1, T]
            )                                            # returns LongTensor [1, T']
            idx_list.append(indices.squeeze(0))          # -> [T']

        # pad to max-length so we can stack
        max_len = max(t.size(0) for t in idx_list)
        idx_padded = [torch.nn.functional.pad(t, (0, max_len - t.size(0)), value=0)
                      for t in idx_list]

        return torch.stack(idx_padded)                   # [B, T']

    # ---------- forward ----------
    def forward(self, x: Tensor) -> Tensor:
        """
        x : [B, T, EEG_ch]  (already windowed)
        returns
        ------
        preds : [B, T', imu_dim]   (sequence-to-sequence prediction)
        """

        # 1)  tokenise each EEG sample → indices
        token_idx = self._tokenize_batch(self.tokenizer, x)       # [B, T']
        token_idx = token_idx.to(x.device)

        # 2)  indices → embeddings
        z = self.codebook(token_idx)                              # [B, T', emb_dim]
        z = z.squeeze(2)

        # 3)  LSTM → IMU
        preds = self.lstm(z)                                      # [B, T', imu_dim]
        return preds

    def train_epoch(self, loader: DataLoader):
        for x, y in loader:
            self.opt.zero_grad()
            preds = self(x)
            y_ds = y.unfold(dimension=1, size=self.compression_factor, step=self.compression_factor).mean(dim=-1)
            loss = self.crit(preds, y_ds)
            loss.backward()
            self.opt.step()
        return loss

    def predict(self, x: Tensor, y: Tensor) -> Union[Tensor, float]:
        preds = self(x)
        y_ds = y.unfold(
            dimension=1, size=self.compression_factor, step=self.compression_factor
        ).mean(dim=-1)
        loss = self.crit(preds, y_ds)
        return preds, y_ds, loss

    # ---------- utility ----------
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
