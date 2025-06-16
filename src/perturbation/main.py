import json
import torch
from src.imu_recon.models import LSTMRegressor
from src.imu_recon.dataset import EEGIMUDataset
from pathlib import Path
from src.perturbation.analyse import ChannelAnalyzer, FrequencyAnalyzer, ModelLevelAnalysis



def main():
    modelname, vis_window = "run_20250607_1945", (1500,2000)        # trained on dataset 0
    # modelname, vis_window = "run_20250607_2210", (1500,2000)        # trained on dataset 0
    # modelname, vis_window = "run_20250612_1823", (3500,4000)      # trained on dataset 1
    # modelname, vis_window = "run_20250614_1646", (300,1100)       # trained on dataset 2

    start = vis_window[0]
    stop = vis_window[1]

    weights_path = f"../../saved_models/weights/{modelname}.pt"
    config_path = f"../../saved_models/configs/{modelname}.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    # Define Dataset
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
    # ca.save_pickle(channel_deltas, ca.respath + f"channel_deltas.pkl")
    channel_deltas = ca.load_pickle(ca.respath + f"channel_deltas.pkl")
    ca.plot_channel_importance(channel_deltas)
    # df_imp = ca.compute_temporal_importance()
    # ca.save_pickle(df_imp,ca.respath + f"tmp_channel_imp.pkl")
    df_imp = ca.load_pickle(ca.respath + f"tmp_channel_imp.pkl")
    ca.plot_temporal_heatmap_and_reconstruction(
       start=start, stop=stop, df=df_imp
    )

    # FREQUENCY ANALYSIS
    fa = FrequencyAnalyzer(model, ds, test_loader, config)
    # band_deltas = fa.analyze_band_importance()
    # fa.save_pickle(band_deltas, fa.respath + f"band_deltas.pkl")
    band_deltas = fa.load_pickle(fa.respath + f"band_deltas.pkl")
    fa.plot_band_importance(band_deltas)#, scale="log", name="log")
    # df_temp = fa.compute_temporal_importance()
    # fa.save_pickle(df_temp, fa.respath + f"tmp_band_importance.pkl")
    df_temp = fa.load_pickle(fa.respath + f"tmp_band_importance.pkl")
    fa.plot_temporal_heatmap_and_reconstruction(
        df_temp, start=start, stop=stop
    )

    # MODEL LEVEL ANALYSIS
    mla = ModelLevelAnalysis(model, ds, test_loader, config)

    df_ig = mla.compute_integrated_gradients(imu_channel=0, start=start, stop=stop)
    mla.save_pickle(df_ig, mla.respath + f"integrated_gradients.pkl")
    df_ig = mla.load_pickle(mla.respath + f"integrated_gradients.pkl")

    mla.plot_model_level_heatmap_and_reconstruction(
        df_ig, start=start, stop=stop, name="_integrated_gradients", title = "Model-Level attribution Heatmap on the integrated gradients for averaged Imu output"
    )

    input("Press enter to exit")
if __name__ == "__main__":
    main()
