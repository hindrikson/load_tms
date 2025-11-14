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

    Y_df, futr_df = utils.load_datasets(cfg)

    # Define validation and test size
    n_time = len(Y_df.ds.unique())
    val_size = int(0.2 * n_time)
    test_size = int(0.2 * n_time)
    print(n_time)
    print(val_size)

    loss = hydra.utils.instantiate(cfg.loss, level=cfg.params.levels)

    model_config = utils.load_model_config(cfg.model_params)

    models = [
        AutoNHITS(
            h=cfg.params.h,
            loss=loss,
            config=cfg.model_params,
            num_samples=cfg.params.n_samples,
            backend="optuna",
        )
    ]

    nf = NeuralForecast(models=models, freq=cfg.params.freq)

    print(models[0].config)


if __name__ == "__main__":
    main()
