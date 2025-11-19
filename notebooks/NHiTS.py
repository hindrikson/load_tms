# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: tms
#     language: python
#     name: tms
# ---

# %%
import logging

import pandas as pd
import plotly.graph_objects as go
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoNHITS
from neuralforecast.core import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss
from ray import tune
from utilsforecast.losses import mae, mse

logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
pd.set_option("display.max_columns", 50)
# torch.set_float32_matmul_precision('high')

# %% [markdown]
# ## Functions


# %%
def plot(df_hist, df_hat, levels=None, model=None):
    """
    Plot historical data and forecasts with optional prediction intervals.

    Parameters:
    -----------
    df_hist : DataFrame
        Historical data with columns 'ds' and 'y'
    df_hat : DataFrame
        Forecast data with columns 'ds', 'unique_id', model predictions, and optional prediction intervals
    levels : list of int, optional
        Confidence levels for prediction intervals (e.g., [80, 90])
    """
    dash_styles = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]

    # Extract base model names (without -lo-XX or -hi-XX suffixes)
    models = []
    for col in df_hat.columns:
        if col in ["ds", "unique_id"]:
            continue
        # Check if it's a base model (not a level column)
        if not any(
            col.endswith(f"-lo-{level}") or col.endswith(f"-hi-{level}")
            for level in levels or []
        ):
            models.append(col)

    fig = go.Figure()

    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=df_hist["ds"],
            y=df_hist["y"],
            mode="lines",
            name="Historical",
            line=dict(color="black", width=2),
        )
    )

    # Add forecasts and prediction intervals
    colors = ["blue", "red", "green", "purple", "orange", "brown"]

    for i, model in enumerate(models):
        color = colors[i % len(colors)]

        # Add main forecast line
        fig.add_trace(
            go.Scatter(
                x=df_hat["ds"],
                y=df_hat[model],
                mode="lines",
                name=model,
                line=dict(dash=dash_styles[i % len(dash_styles)], color=color, width=2),
            )
        )

        # Add prediction intervals if levels are provided
        if levels:
            for level in sorted(levels, reverse=True):  # Plot wider intervals first
                lo_col = f"{model}-lo-{level}"
                hi_col = f"{model}-hi-{level}"

                if lo_col in df_hat.columns and hi_col in df_hat.columns:
                    # Add upper bound (invisible line, just for fill)
                    fig.add_trace(
                        go.Scatter(
                            x=df_hat["ds"],
                            y=df_hat[hi_col],
                            mode="lines",
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

                    # Add lower bound with fill
                    fig.add_trace(
                        go.Scatter(
                            x=df_hat["ds"],
                            y=df_hat[lo_col],
                            mode="lines",
                            line=dict(width=0),
                            fillcolor=f"rgba({int(color == 'blue') * 0},{int(color == 'red') * 255},{int(color == 'green') * 0},0.{100 - level // 2})",
                            fill="tonexty",
                            name=f"{model} {level}% PI",
                            hoverinfo="skip",
                        )
                    )

    fig.update_layout(
        title="Germany Historical vs Forecast",
        width=1400,
        height=500,
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
    )

    fig.show()


# %% [markdown]
# ## Load Data

# %%
Y_df = pd.read_csv(
    "/mnt/data/oe215/rhindrikson/datasets/load/entsoe/data.csv", parse_dates=["ds"]
)
futr_df = pd.read_csv(
    "/mnt/data/oe215/rhindrikson/datasets/load/entsoe/futr.csv", parse_dates=["ds"]
)

# %%
# Define validation and test size
n_time = len(Y_df.ds.unique())
val_size = int(0.2 * n_time)
test_size = int(0.2 * n_time)
Y_df.groupby("unique_id").head(2)

# %%
import matplotlib.pyplot as plt

# We are going to plot the temperature of the transformer
# and marking the validation and train splits
u_id = "load"
x_plot = pd.to_datetime(Y_df[Y_df.unique_id == u_id].ds)
y_plot = Y_df[Y_df.unique_id == u_id].y.values

x_val = x_plot[n_time - val_size - test_size]
x_test = x_plot[n_time - test_size]

fig = plt.figure(figsize=(15, 5))
fig.tight_layout()

plt.plot(x_plot, y_plot)
plt.xlabel("Date", fontsize=17)
plt.ylabel("Load [Hourly temperature]", fontsize=17)

plt.axvline(x_val, color="black", linestyle="-.")
plt.axvline(x_test, color="black", linestyle="-.")
plt.text(x_val, 5, "  Validation", fontsize=12)
plt.text(x_test, 5, "  Test", fontsize=12)

plt.grid()

# %% [markdown]
# ## Model

# %%
horizon = 7 * 24

nhits_config = {
    "learning_rate": tune.choice([1e-3]),  # Initial Learning rate
    "max_steps": tune.choice([1000]),  # Number of SGD steps
    "input_size": tune.choice([104 * horizon]),  # input_size = multiplier * horizon
    "batch_size": tune.choice([7]),  # Number of series in windows
    "windows_batch_size": tune.choice([256]),  # Number of windows in batch
    "n_pool_kernel_size": tune.choice([[2, 2, 2], [16, 8, 1]]),  # MaxPool's Kernelsize
    "n_freq_downsample": tune.choice(
        [[168, 24, 1], [24, 12, 1], [1, 1, 1]]
    ),  # Interpolation expressivity ratios
    "activation": tune.choice(["ReLU"]),  # Type of non-linear activation
    "n_blocks": tune.choice([[1, 1, 1]]),  # Blocks per each 3 stacks
    "mlp_units": tune.choice(
        [[[512, 512], [512, 512], [512, 512]]]
    ),  # 2 512-Layers per block for each stack
    "interpolation_mode": tune.choice(["linear"]),  # Type of multi-step interpolation
    "val_check_steps": tune.choice([100]),  # Compute validation every 100 epochs
    "random_seed": tune.randint(1, 10),
}

# %% [markdown]
# ## Instantiate Model
#
# To instantiate `AutoNHITS` you need to define:
#
# * `h`: forecasting horizon
# * `loss`: training loss. Use the `DistributionLoss` to produce probabilistic forecasts.
# * `config`: hyperparameter search space. If `None`, the `AutoNHITS` class will use a pre-defined suggested hyperparameter space.
# * `num_samples`: number of configurations explored.
#
# If num_samples equals 5, the AutoNHITS model will randomly sample 5 different combinations of hyperparameters from the search space defined in nhits_config.
# Each configuration will be trained and evaluated.
# The best performing configuration (based on validation performance) will be selected as the final model
#
# For loss, common distribution options include:
# - 'Normal' or 'Gaussian' - for normal/gaussian distribution
# - 'StudentT' or 'T' - for Student's t-distribution
# - 'NegativeBinomial' - for count data
# - 'Poisson' - for count data
# - 'Tweedie' - for non-negative continuous data
#
#

# %%
levels = [80, 90]
# loss=MQLoss(level=levels),
loss = DistributionLoss(distribution="Normal", level=levels)
models = [
    AutoNHITS(
        h=horizon,
        loss=loss,
        config=nhits_config,
        num_samples=5,
    )
]

# %%
# %%capture
nf = NeuralForecast(models=models, freq="h")

Y_hat_df = nf.cross_validation(
    df=Y_df, val_size=val_size, test_size=test_size, n_windows=None
)

# %% [markdown]
# ## Evaluate Results
#
# The `AutoNHITS` class contains a `results` tune attribute that stores information of each configuration explored. It contains the validation loss and best validation hyperparameter.

# %%
nf.models[0].results.get_best_result().config

# %%
y_true = Y_hat_df.y.values
y_hat = Y_hat_df["AutoNHITS"].values

n_series = len(Y_df.unique_id.unique())

y_true = y_true.reshape(n_series, -1, horizon)
y_hat = y_hat.reshape(n_series, -1, horizon)

print("Parsed results")
print("2. y_true.shape (n_series, n_windows, n_time_out):\t", y_true.shape)
print("2. y_hat.shape  (n_series, n_windows, n_time_out):\t", y_hat.shape)

# %%
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 11))
fig.tight_layout()

series = ["load"]
series_idx = 0

plt.figure(figsize=(14, 6))
plt.plot(y_true[series_idx, :, :].flatten(), label="True")
plt.plot(y_hat[series_idx, :, :].flatten(), label="Forecast")
plt.grid()
plt.xlabel("Time", fontsize=14)
plt.ylabel(series[series_idx], fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

# %% [markdown]
# Finally, we compute the test errors for the two metrics of interest:
#
# $\qquad MAE = \frac{1}{Windows * Horizon} \sum_{\tau} |y_{\tau} - \hat{y}_{\tau}| \qquad$ and $\qquad MSE = \frac{1}{Windows * Horizon} \sum_{\tau} (y_{\tau} - \hat{y}_{\tau})^{2} \qquad$

# %%
from neuralforecast.losses.numpy import mae, mse

print("MAE: ", mae(y_hat, y_true))
print("MSE: ", mse(y_hat, y_true))

# %%
