import json
import torch
from src.imu_recon.models import LSTMRegressor
from src.imu_recon.dataset import EEGIMUDataset
from pathlib import Path
from src.perturbation.analyse import ChannelAnalyzer, FrequencyAnalyzer, ModelLevelAnalysis, Plotter
import numpy as np


def main():
    modelname, vis_window = "run_20250607_1945", (1500,2000)        # trained on dataset 0
    # modelname, vis_window = "run_20250607_2210", (1500,2000)        # trained on dataset 0
    # modelname, vis_window = "run_20250612_1823", (3500,4000)      # trained on dataset 1
    # modelname, vis_window = "run_20250614_1646", (300,1100)       # trained on dataset 2

    weights_path = f"../../saved_models/weights/{modelname}.pt"
    config_path = f"../../saved_models/configs/{modelname}.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    # DEFINE DATASET
    print(f"Dataset used {config['dataset']} with test loss: {config['test_loss']:.3f}")
    csv_folder = "../../data/EEG_IMU/"
    csv_path = sorted(Path(csv_folder).glob("*.csv"))[config["dataset"]]
    ds = EEGIMUDataset(
        csv_path, window=config["window"], stride=config["stride"], bandpass=config["bandpass"]
    )

    # TRAIN/TEST/VAL SPLIT
    train_loader, val_loader, test_loader = ds.train_test_val_split(
        config["batch_size"]
    )

    # LOAD MODEL
    model = LSTMRegressor(in_dim=16, config=config, out_dim=12)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # PERTURBATION ANALYSIS
    # CHANNEL ANALYSIS
    ca = ChannelAnalyzer(model, ds, test_loader, config)
    # channel_deltas = ca.analyze_channel_importance()
    # ca.save_pickle(channel_deltas, f"channel_deltas.pkl")
    channel_deltas = ca.load_pickle(f"channel_deltas.pkl")
    important_channel = np.argsort(channel_deltas)[-6:][::-1].tolist()  # get first 4 important channels
    # df_ch_imp = ca.compute_temporal_importance()
    # ca.save_pickle(df_ch_imp, f"tmp_channel_imp.pkl")
    df_ch_imp = ca.load_pickle(f"tmp_channel_imp.pkl")

    # FREQUENCY ANALYSIS
    fa = FrequencyAnalyzer(model, ds, test_loader, config)
    # band_deltas = fa.analyze_band_importance()
    # fa.save_pickle(band_deltas, f"band_deltas.pkl")
    band_deltas = fa.load_pickle(f"band_deltas.pkl")
    # df_freq_imp = fa.compute_temporal_importance()
    # fa.save_pickle(df_freq_imp, f"tmp_band_importance.pkl")
    df_freq_imp = fa.load_pickle(f"tmp_band_importance.pkl")

    # MODEL LEVEL ANALYSIS
    mla = ModelLevelAnalysis(model, ds, test_loader, config)
    # df_ig = mla.compute_integrated_gradients(imu_channel=0)
    # mla.save_pickle(df_ig, f"integrated_gradients.pkl")
    df_ig = mla.load_pickle(f"integrated_gradients.pkl")


    # PLOT
    plotter = Plotter(model, ds, test_loader, config, vis_window=vis_window)

    plotter.plot_channel_importance(channel_deltas)
    plotter.plot_band_importance(band_deltas)

    plotter.plot_with_timeseries(
        df_freq_imp,
        spec_channel=important_channel,
        filename="timeseries_analysis_frequencies.png",
    )

    plotter.plot_with_spectrogram(
        df_ig,
        spec_channel= important_channel,
        filename = "spectrogram_analysis_integrated_gradients.png",
    )

    plotter.plot_combined(df_ch_imp, df_freq_imp, df_ig, spec_channel=important_channel)

    input("Press enter to exit")
if __name__ == "__main__":
    main()
