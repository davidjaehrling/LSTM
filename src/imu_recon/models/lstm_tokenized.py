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


class TokenizedLSTMRegressor(nn.Module):
    """
    EEG window  ->  TOTEM tokenizer (frozen)  ->  code-book embeddings
                 ->  LSTM  ->  IMU reconstruction
    """

    def __init__(
        self,
        in_dim: int,
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
            in_dim  = in_dim,
            config = config,
            out_dim = out_dim,
        )

        self.opt = torch.optim.Adam(self.parameters(), lr=config["lr"])
        self.crit = torch.nn.MSELoss()

    # def _tokenize_batch(
    #     self, batch_eeg: Tensor
    # ) -> Tensor:
    #     """
    #     Tokenizes a batch of multichannel EEG windows using a pretrained TOTEM tokenizer.
    #
    #     Args
    #     ----
    #     tokenizer   : instance of TokenizerTimeSeries
    #     batch_eeg   : [B, T=128, C=16]  → full-resolution EEG
    #
    #     Returns
    #     -------
    #     token_idx : LongTensor [B, T′, C] where T′ = T // compression_factor
    #     """
    #     B, T, C = batch_eeg.shape
    #     token_seqs = []
    #
    #     for b in range(B):
    #         eeg = batch_eeg[b]  # [T, C]
    #
    #         tokens_per_ch = []
    #         for c in range(C):
    #             ch_signal = eeg[:, c].unsqueeze(0)  # [1, T]
    #             with torch.no_grad():
    #                 token = self.tokenizer.tokenize(ch_signal)  # [1, T′]
    #             tokens_per_ch.append(token.squeeze(0))  # [T′]
    #
    #         tokens_per_ch = torch.stack(tokens_per_ch, dim=1)  # [T′, C]
    #         token_seqs.append(tokens_per_ch)
    #
    #     return torch.stack(token_seqs, dim=0)  # [B, T′, C]

    def _tokenize_batch(self, batch_eeg: Tensor) -> Tensor:
        """
        Fast tokenization of multichannel EEG using pretrained TOTEM tokenizer.

        Args
        ----
        batch_eeg : Tensor of shape [B, T, C]

        Returns
        -------
        token_idx : Tensor of shape [B, T', C]
        """
        B, T, C = batch_eeg.shape

        # Reshape to merge batch and channel: [B*C, T]
        eeg_reshaped = batch_eeg.permute(0, 2, 1).reshape(-1, T)

        with torch.no_grad():
            # Tokenize all signals at once: output [B*C, T′]
            tokens = self.tokenizer.tokenize(
                eeg_reshaped
            )  # assuming this works with batched input

        # Reshape back: [B, C, T′] → permute to [B, T′, C]
        T_prime = tokens.numel() // (B * C)
        tokens = tokens.view(B, C, T_prime).permute(0, 2, 1)  # [B, T′, C]
        return tokens

    # ---------- forward ----------
    def forward(self, x: Tensor) -> Tensor:
        """
        x : [B, T, EEG_ch]  (already windowed)
        returns
        ------
        preds : [B, T', imu_dim]   (sequence-to-sequence prediction)
        """

        # 1)  tokenise each EEG sample → indices
        tokens = self._tokenize_batch(batch_eeg=x)       # [B, T']
        tokens = tokens.to(x.device).float()

        # 2)  LSTM → IMU
        preds = self.lstm(tokens)                                      # [B, T', imu_dim]
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
