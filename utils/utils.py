import hydra
import pandas as pd
import plotly.graph_objects as go


def initialize_logger(cfg):
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


def load_nhits_model_config(cfg):
    if cfg.default:
        print("Using default NHITS model configuration.")
        return None

    def model_config(trial):
        print("Using custom NHITS model configuration.")
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


def val_test_plot(df, val_size, test_size):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for key, grp in df.groupby("unique_id"):
        plt.plot(grp["ds"], grp["y"], label=key)

    plt.axvline(
        x=df["ds"].unique()[-(val_size + test_size)],
        color="orange",
        linestyle="--",
        label="Validation Start (length: {})".format(val_size),
    )
    plt.axvline(
        x=df["ds"].unique()[-test_size],
        color="red",
        linestyle="--",
        label="Test Start, (length: {})".format(test_size),
    )

    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Time Series Data with Validation and Test Split")
    plt.legend()

    return plt


# Get only the last forecast window (test set)
# last_cutoff = Y_hat_df['cutoff'].max()
# Y_hat_test = Y_hat_df[Y_hat_df['cutoff'] == last_cutoff]


def plot_test_forecast(df_hat, levels=None):
    """
    Plot historical data and forecasts with optional prediction intervals.

    Parameters:
    -----------
    df_hat : DataFrame
        Forecast data with columns 'ds', 'unique_id', model predictions, and optional prediction intervals
    levels : list of int, optional
        Confidence levels for prediction intervals (e.g., [80, 90])
    """
    dash_styles = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]

    # Extract base model names (without -lo-XX or -hi-XX suffixes)
    models = []
    for col in df_hat.columns:
        if col in ["ds", "unique_id", "cutoff", "y"]:
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
            x=df_hat["ds"],
            y=df_hat["y"],
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

    return fig
