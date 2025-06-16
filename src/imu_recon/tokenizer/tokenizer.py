# src/imu_recon/tokenizer/tokenizer.py
from typing import List, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.TOTEM.anomaly_detection.lib.models.vqvae import vqvae


class DummyLogger:
    def log_metric(self, *args, **kwargs):
        pass

class TOTEM():
    def __init__(self, vocab_path: Union[str, None] = None):

        self.vqvae = self.load_vqvae(vocab_path)
        self.vqvae.eval()  # Set to evaluation mode
        self.log = DummyLogger()

    def tokenize(self, series: Union[np.ndarray, List[float]], frame=True, max_length=-1, **kwargs):
        """
        Tokenizes a time series using closest match from trained coefficient vocab.
        """
        _, quantized, _, data_recon, _, embedding_weight, encoding_indices, _ = self.vqvae.shared_eval(
            series, None, 'val', self.log)
        return encoding_indices

    def detokenize(self, tokens: List[int], **kwargs) -> np.ndarray:
        """
        Converts token sequence back into a time series using the VQ-VAE decoder.
        Args:
            tokens: List of token indices to decode
            kwargs: Additional arguments (e.g., device for computation)
        Returns:
            np.ndarray: Reconstructed time series
        """
        # Convert tokens to quantized embeddings
        with torch.no_grad():
            # Get embeddings from codebook
            #token_tensor = torch.tensor(tokens, dtype=torch.long).to(self.vqvae.vq._embedding.weight.device)
            token_tensor = tokens.clone().detach().long().to(self.vqvae.vq._embedding.weight.device)

            quantized = self.vqvae.vq._embedding(token_tensor)

            # Reshape for decoder (add batch and channel dimensions)
            quantized = quantized.unsqueeze(0).transpose(1, 2)  # [1, embedding_dim, seq_len]

            # Decode to time series
            reconstructed = self.vqvae.decoder(quantized, self.vqvae.compression_factor)

        return reconstructed.squeeze().cpu().numpy()

    @staticmethod
    def load_vqvae(load_path):
        checkpoint = torch.load(load_path)
        config = checkpoint['config']

        # Recreate model architecture
        loaded_model = vqvae(config)

        # Load trained weights
        loaded_model.load_state_dict(checkpoint['model_state_dict'])

        # Optional: Verify codebook
        loaded_model.vq._embedding.weight.data = checkpoint['codebook']

        return loaded_model



    @staticmethod
    def train(series_list: List[np.ndarray],
              compression_factor: int = 16,
              num_embeddings: int = 512,
              save_path: str = "vqvae_model.pth",
              num_epochs=200):
        """
        Trains the tokenizer by extracting patches, fitting polynomials, and quantizing coeffs.
        Saves a fixed vocab of most common quantized coeffs.
        """

        data_tensor = torch.tensor(series_list, dtype=torch.float32)

        # Create DataLoader
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        vqvae_config = {
            "model_name": "vqvae",
            "model_save_name": "vqvae",
            "pretrained": False,
            "learning_rate": 1e-3,
            "num_training_updates": 120000,
            "block_hidden_size": 128,
            "num_residual_layers": 2,
            "res_hidden_size": 64,
            "embedding_dim": 64,
            "num_embeddings": num_embeddings,
            "commitment_cost": 0.25,
            "compression_factor": compression_factor,
        }

        model = vqvae(vqvae_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            for batch in dataloader:
                inputs = batch[0]

                # Dummy comet_logger to avoid errors (not used here)
                class DummyLogger:
                    def log_metric(self, *args, **kwargs):
                        pass

                logger = DummyLogger()

                loss, _, _, _, _, _, _, _ = model.shared_eval(inputs, optimizer, 'train', logger)
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

        torch.save({
            'model_state_dict': model.state_dict(),
            'config': vqvae_config,
            'codebook': model.vq._embedding.weight.detach().cpu()
        }, save_path)