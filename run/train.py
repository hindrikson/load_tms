from __future__ import print_function

import logging

import comet_ml  # noqa: F401
import hydra
import optuna  # noqa: F401
import ray.tune as tune  # noqa: F401
import rootutils
from dotenv import load_dotenv
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
    val_size = int(cfg.params.val_size * n_time)
    test_size = int(cfg.params.test_size * n_time)
    print(n_time)
    print(val_size)

    # Plot the data splits
    plot = utils.val_test_plot(Y_df, val_size, test_size)
    logger.log_figure(figure=plot, figure_name="val_test_split.png")

    # Load loss and model config
    loss = hydra.utils.instantiate(cfg.loss, level=cfg.params.levels)
    config = utils.load_nhits_model_config(cfg.model_params)

    # Initialize models
    models = [
        AutoNHITS(
            h=cfg.params.h,
            loss=loss,
            config=config,
            num_samples=cfg.params.n_samples,
            backend="optuna",
        )
    ]
    nf = NeuralForecast(models=models, freq=cfg.params.freq)

    # Fit models
    Y_hat_df = nf.cross_validation(
        df=Y_df, val_size=val_size, test_size=test_size, n_windows=None
    )
    # best config
    results = nf.models[0].results.trials_dataframe()
    best_config = results.iloc[0, :].to_dict()

    val_loss = best_config["value"]
    config_params = best_config["user_attrs_ALL_PARAMS"]
    config_params["val_loss"] = val_loss

    # Log loss and best config parameters
    logger.log_metrics({"val_loss": val_loss})
    logger.log_parameters(
        {"initial_search_config": cfg, "best_config": config_params},
    )

    # Log test results
    fig = utils.plot_test_forecast(Y_hat_df, levels=cfg.params.levels)

    # losseslogg
    """
    y_true = Y_hat_df.y.values
    y_hat = Y_hat_df["AutoNHITS"].values

    n_series = len(Y_df.unique_id.unique())

    y_true = y_true.reshape(n_series, -1, horizon)
    y_hat = y_hat.reshape(n_series, -1, horizon)

    print("Parsed results")
    print("2. y_true.shape (n_series, n_windows, n_time_out):\t", y_true.shape)
    print("2. y_hat.shape  (n_series, n_windows, n_time_out):\t", y_hat.shape)
    # print("MAE: ", mae(y_hat, y_true))
    # print("MSE: ", mse(y_hat, y_true))
    """


if __name__ == "__main__":
    load_dotenv(".env")
    main()
