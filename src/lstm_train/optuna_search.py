from src.lstm_train.train import main
import optuna


def objective(trial):
    bandpass_options = [(1,30), (1,20), (1,12)]
    config = {'batch_size': trial.suggest_categorical("batch_size", [32, 64, 128]),
              'hidden_dim': trial.suggest_int("hidden_dim", 64, 256, step=32),
              'num_layers': trial.suggest_int("num_layers", 2, 4),
              'lr': trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
              'dropout': trial.suggest_float("dropout", 0.0, 0.5),
              'bidirectional': trial.suggest_categorical("bidirectional", [True, False]),
              'window': trial.suggest_categorical("window_size", [128]),
              'dataset_id': trial.suggest_categorical("dataset_id", [0]),
              'bandpass': trial.suggest_categorical("bandpass", bandpass_options)}

    config["stride"] = config["window"] // 2
    config["model"] = "LSTM"
    config["epochs"] = 200
    config["patience"] = 20
    config["dataset"] = "EEG_ET"

    loss, parameter_count = main(config)

    return loss, parameter_count


if __name__ == "__main__":
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        study_name=f"EEG-ET_study",
        sampler=optuna.samplers.NSGAIISampler()
    )
    study.optimize(lambda trial: objective(trial), n_trials=50)
