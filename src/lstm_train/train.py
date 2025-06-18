import torch
from src.utils import EEGIMUDataset
from src.lstm_train.models import LSTMRegressor
from src.utils.utils import plot_reconstruction, reconstruct_signal
from pathlib import Path
from typing import Dict
import json

# Tensorboard imports
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def main(config: Dict) -> float:
    # Define Dataset
    csv_folder = f"../../data/{config['data_type']}/"
    csv_path = sorted(Path(csv_folder).glob("*.csv"))[config["dataset"]]
    ds = EEGIMUDataset(csv_path, window=config["window"], stride=config["stride"], bandpass = config["bandpass"])

    # TRAIN/TEST/VAL SPLIT
    train_loader, val_loader, test_loader = ds.train_test_val_split(config["batch_size"])

    # LOAD MODEL
    net = LSTMRegressor(in_dim=ds.inp_dim, config=config, out_dim=ds.out_dim)

    # START TENSORBOARD WRITER
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M')}"
    writer = SummaryWriter(log_dir=f"../../saved_models/runs/{run_name}")
    example_input_batch, _ = next(iter(train_loader))
    writer.add_graph(net, example_input_batch)

    # TRAINING
    best_val_loss, best_epoch = float("inf"), 0
    for epoch in range(config["epochs"]):
        train_loss = net.train_epoch(train_loader)
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                preds, _, loss = net.predict(x_val, y_val)
                val_loss += loss.item() * x_val.size(0)

        val_loss /= len(val_loader.dataset)

        # Tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        print(f"epoch: {epoch} train loss: {train_loss:.4f} val loss: {val_loss:.4f}")

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), f"../../saved_models/weights/{run_name}.pt")
        elif epoch - best_epoch >= config["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    # TESTING
    test_loss = 0.0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            preds, _, loss = net.predict(x_test, y_test)
            test_loss += loss.item() * x_test.size(0)
    test_loss /= len(test_loader.dataset)
    config["test_loss"] = test_loss

    # save config as JSON
    with open(f"../../saved_models/configs/{run_name}.json", "w") as f:
        json.dump(config, f, indent=4)

    # PLOT EXAMPLE RECONSTRUCTION
    full_pred, full_true = reconstruct_signal(
        net=net, loader=test_loader, base_ds=ds
    )

    # Plot an entire IMU channel (e.g. x-axis)
    fig = plot_reconstruction(
        true=full_true,
        pred=full_pred,
        channel=0,
        timesteps=(300, 600),
        title="Test-segment IMU X-axis reconstruction",
    )

    fig.savefig(f"../../saved_models/reconstruction_plots/reconstruction_example_{run_name}.png")
    writer.add_figure("IMU Reconstruction Example", fig, global_step=0)
    metrics = {"val_loss": best_val_loss, "test_loss": test_loss}
    config["bandpass"] = str(config["bandpass"])    # make bandpass a string so it can be passed to tensorboard
    writer.add_hparams(config, metrics)
    writer.close()

    return test_loss,  net.num_parameters


if __name__ == "__main__":

    config = {
        "batch_size": 32,
        "hidden_dim": 224,
        "num_layers": 3,
        "lr": 0.005,
        "dropout": 0.2,
        "bidirectional": True,
        "window": 128,
        "dataset": 0,
        "stride": 64,
        "model": "LSTM",
        "epochs": 0,
        "patience": 20,
        "bandpass": (1, 12),
        "data_type": "EEG_IMU",
    }

    loss, compl = main(config)
