from __future__ import print_function

import logging

import hydra
import optuna  # noqa: F401
import ray.tune as tune  # noqa: F401
import rootutils
from neuralforecast.auto import AutoNHITS
from neuralforecast.core import NeuralForecast
from omegaconf import OmegaConf

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

OmegaConf.register_new_resolver(name="merge", resolver=lambda x, y: str(x) + str(y))

PROJECT_ROOT = rootutils.setup_root(
    __file__, indicator=".project-root", pythonpath=True
)

import utils  # noqa: F401


@hydra.main(
    version_base="1.3",
    config_path=str(PROJECT_ROOT / "config"),
    config_name="nhits.yaml",
)
def main(cfg):
    print("Config loaded successfully")
    print(cfg)

    logger = utils.initialize_logger(cfg)

    Y_df, futr_df = utils.load_datasets(cfg)

    # Define validation and test size
    n_time = len(Y_df.ds.unique())
    val_size = int(0.2 * n_time)
    test_size = int(0.2 * n_time)
    print(n_time)
    print(val_size)

    loss = hydra.utils.instantiate(cfg.loss, level=cfg.params.levels)

    model_config_func = utils.load_model_config(cfg.model_params)

    models = [
        AutoNHITS(
            h=cfg.params.h,
            loss=loss,
            config=model_config_func,
            num_samples=cfg.params.n_samples,
            backend="optuna",
        )
    ]

    nf = NeuralForecast(models=models, freq=cfg.params.freq)

    Y_hat_df = nf.cross_validation(
        df=Y_df, val_size=val_size, test_size=test_size, n_windows=None
    )

    # best config
    results = nf.models[0].results.trials_dataframe()

    best_config = results.iloc[0, :].to_dict()

    val_loss = best_config["value"]
    config_params = best_config["user_attrs_ALL_PARAMS"]
    config_params["val_loss"] = val_loss

    # logging to comet logger
    logger.log_parameters(
        {"initial_search_config": cfg, "best_config": config_params},
    )

    logger.log_metrics({"val_loss": val_loss})

    # losses
    # print("MAE: ", mae(y_hat, y_true))
    # print("MSE: ", mse(y_hat, y_true))


if __name__ == "__main__":
    main()
