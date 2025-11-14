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
            "start_padding_enabled": cfg.start_padding_enabled,
            "n_blocks": cfg.n_blocks * [1],
            "mlp_units": cfg.mlp_units.n * cfg.mlp_units.shape,
            "scaler_type": cfg.scaler_type,
            "max_steps": trial.suggest_categorical("max_steps", tuple(cfg.max_steps)),
            "input_size": trial.suggest_categorical(
                "input_size", tuple(cfg.input_size)
            ),
            "n_pool_kernel_size": trial.suggest_categorical(
                "n_pool_kernel_size", tuple(cfg.n_pool_kernel_size)
            ),
            "n_freq_downsample": trial.suggest_categorical(
                "n_freq_downsample", tuple(cfg.n_freq_downsample)
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                low=cfg.learning_rate.low,
                high=cfg.learning_rate.high,
                log=cfg.learning_rate.log,
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size", tuple(cfg.batch_size)
            ),
            "windows_batch_size": trial.suggest_categorical(
                "window_batch_size", tuple(cfg.window_batch_size)
            ),
            "random_seed": trial.suggest_categorical(
                "random_seed", low=cfg.random_seed.low, high=cfg.random_seed.high
            ),
            "activation": trial.suggest_categorical("activation", (cfg.activation)),
            "interpolation_mode": trial.suggest_categorical(
                "interpolation_mode",
                (cfg.interpolation_mode),
            ),
            "val_check_steps": trial.suggest_categorical(
                "val_check_steps", (cfg.val_check_steps)
            ),
        }

    return model_config
