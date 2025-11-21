from __future__ import print_function

import logging
import time

import comet_ml  # noqa: F401
import hydra
import optuna  # noqa: F401
import ray.tune as tune  # noqa: F401
import rootutils
from dotenv import load_dotenv
from neuralforecast.auto import AutoNHITS
from neuralforecast.core import NeuralForecast
from omegaconf import OmegaConf
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    start = time.time()

    logger = utils.initialize_logger(cfg)

    # set experiment name
    logger.set_name(cfg.params.experiment_name)

    Y_df, futr_df = utils.load_datasets(cfg)

    # Define validation and test size as percentage of time series length
    # n_time = len(Y_df.ds.unique())
    # val_size = int(cfg.params.val_size * n_time)
    # val_size = val_size - (val_size % cfg.params.h)
    # test_size = int(cfg.params.test_size * n_time)
    # test_size = test_size - (test_size % cfg.params.h)

    # Define the validation and test as number of windows
    val_size = cfg.params.val_size * cfg.dataset.h
    test_size = cfg.params.test_size * cfg.dataset.h

    if test_size % cfg.dataset.h != 0:
        raise ValueError("Test size must be multiple of horizon")
    if val_size % cfg.dataset.h != 0:
        raise ValueError("Validation size must be multiple of horizon")

    # Plot the data splits
    plot = utils.val_test_plot(Y_df, val_size, test_size)
    logger.log_figure(figure=plot, figure_name="val_test_split.png")

    # Load loss and model config
    loss = hydra.utils.instantiate(cfg.loss, level=cfg.params.levels)
    config = utils.load_nhits_default_config(cfg.model_params)

    # Initialize models
    models = [
        AutoNHITS(
            h=cfg.dataset.h,
            loss=loss,
            config=config,
            num_samples=cfg.params.n_samples,
            backend="optuna",
        )
    ]
    nf = NeuralForecast(models=models, freq=cfg.dataset.freq)

    # Fit models
    Y_hat_df = nf.cross_validation(
        df=Y_df,
        val_size=val_size,
        test_size=test_size,
        step_size=cfg.dataset.h,
        n_windows=cfg.params.n_windows,
        refit=cfg.params.refit,
    )
    if len(Y_hat_df) != test_size:
        raise ValueError(
            f"Forecast length {len(Y_hat_df)} does not match test size {test_size}"
        )

    # create figure
    fig = utils.add_temp_plot(Y_df, test_size)
    fig = utils.plot_test_forecast(Y_hat_df, fig, levels=cfg.params.levels)

    # log as html
    html_str = fig.to_html()
    logger.log_html(html_str)

    # log as asset
    html_path = "./my_plot.html"
    fig.write_html(html_path)
    logger.log_asset(html_path)

    # get best config
    results = nf.models[0].results.trials_dataframe()
    best_config = results.iloc[0, :].to_dict()
    test_loss = best_config["value"]

    config_params = best_config["user_attrs_ALL_PARAMS"]
    config_params["test_loss"] = test_loss

    # Log loss and best config parameters
    logger.log_metrics({"test_loss": test_loss})
    logger.log_parameters(
        {"best_config": config_params},
        {"params": cfg.params},
    )

    # log losses
    y_hat = Y_hat_df["AutoNHITS"].values
    y_true = Y_hat_df["y"].values

    mae_loss = mean_absolute_error(y_true, y_hat)
    mse_loss = mean_squared_error(y_true, y_hat)
    print(f"Test MAE: {mae_loss}")
    print(f"Test MSE: {mse_loss}")

    # Log test losses
    logger.log_metrics(
        {
            "test_mae": mae_loss,
            "test_mse": mse_loss,
        }
    )

    end = time.time()
    print(f"Total time: {(end - start) / 60:.2f} minutes")


if __name__ == "__main__":
    load_dotenv(".env")
    main()
