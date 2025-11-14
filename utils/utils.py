import hydra
import pandas as pd


def initialize_logger(cfg):
    if cfg.params.custom_project:
        # Custom logger initialization
        logger = hydra.utils.instantiate(
            cfg.logger,
            workspace=cfg.dataset.comet_workspace,
            project_name=cfg.params.custom_project,
        )
        return logger

    if cfg.dataset.comet_workspace:
        logger = hydra.utils.instantiate(
            cfg.logger,
            workspace=cfg.dataset.comet_workspace,
            project_name="test",
        )

        return logger
    else:
        raise ValueError("Comet.ml workspace not specified in the configuration.")


def load_datasets(cfg):
    Y_df = pd.read_csv(cfg.dataset.data, parse_dates=["ds"])
    futr_df = pd.read_csv(cfg.dataset.futr, parse_dates=["ds"])
    return Y_df, futr_df


def load_model_config(cfg):
    def model_config(trial):
        return {
            "start_padding_enabled": trial.suggest_categorical(),
            "n_blocks": trial.suggest_categorical(),
            "mlp_units": trial.suggest_categorical(),
            "scaler_type": trial.suggest_categorical(),
            "max_steps": trial.suggest_categorical(),
            "input_size": trial.suggest_categorical(),
            "n_pool_kernel_size": trial.suggest_categorical(),
            "n_freq_downsample": trial.suggest_categorical(),
            "learning_rate": trial.suggest_categorical(),
            "batch_size": trial.suggest_categorical(),
            "windows_batch_size": trial.suggest_categorical(),
            "random_seed": trial.suggest_categorical(),
            "activation": trial.suggest_categorical(),
            "interpolation_mode": trial.suggest_categorical(),
            "val_check_steps": trial.suggest_categorical(),
        }

    return model_config
