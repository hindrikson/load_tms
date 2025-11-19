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
    try:
        futr_df = pd.read_csv(cfg.dataset.futr, parse_dates=["ds"])
    except AttributeError:
        print("No future covariates provided.")
        futr_df = None
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
    latest_cutoff = df_hat["cutoff"].max()
    df_hat = df_hat[df_hat["cutoff"] == latest_cutoff].copy()

    df_hat = df_hat.tail(100)

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


def plot_test_forecast2(df_hat, levels=None, n_cutoffs=10):
    """
    Plot historical data and forecasts with optional prediction intervals.
    Parameters:
    -----------
    df_hat : DataFrame
        Forecast data with columns 'ds', 'unique_id', model predictions, and optional prediction intervals
    levels : list of int, optional
        Confidence levels for prediction intervals (e.g., [80, 90])
    n_cutoffs : int, optional
        Number of most recent cutoffs to plot (default: 10)
    """
    # Get the n most recent cutoffs
    latest_cutoffs = df_hat["cutoff"].nlargest(n_cutoffs).unique()
    df_hat = df_hat[df_hat["cutoff"].isin(latest_cutoffs)].copy()

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

    # Add historical data (from all cutoffs, taking unique values)
    hist_data = df_hat[["ds", "y"]].drop_duplicates(subset="ds").sort_values("ds")
    fig.add_trace(
        go.Scatter(
            x=hist_data["ds"],
            y=hist_data["y"],
            mode="lines",
            name="Historical",
            line=dict(color="black", width=2),
        )
    )

    # Add forecasts for each cutoff
    colors = [
        "blue",
        "red",
        "green",
        "purple",
        "orange",
        "brown",
        "pink",
        "cyan",
        "magenta",
        "yellow",
    ]

    for cutoff_idx, cutoff in enumerate(sorted(latest_cutoffs)):
        df_cutoff = df_hat[df_hat["cutoff"] == cutoff].sort_values("ds")

        for model_idx, model in enumerate(models):
            color = colors[(cutoff_idx * len(models) + model_idx) % len(colors)]

            # Add main forecast line
            fig.add_trace(
                go.Scatter(
                    x=df_cutoff["ds"],
                    y=df_cutoff[model],
                    mode="lines",
                    name=f"{model} (cutoff: {cutoff})",
                    line=dict(
                        dash=dash_styles[model_idx % len(dash_styles)],
                        color=color,
                        width=1.5,
                    ),
                    legendgroup=f"{model}-{cutoff}",
                )
            )

            # Add prediction intervals if levels are provided
            if levels:
                for level in sorted(levels, reverse=True):  # Plot wider intervals first
                    lo_col = f"{model}-lo-{level}"
                    hi_col = f"{model}-hi-{level}"

                    if lo_col in df_cutoff.columns and hi_col in df_cutoff.columns:
                        # Add upper bound (invisible line, just for fill)
                        fig.add_trace(
                            go.Scatter(
                                x=df_cutoff["ds"],
                                y=df_cutoff[hi_col],
                                mode="lines",
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo="skip",
                                legendgroup=f"{model}-{cutoff}",
                            )
                        )

                        # Add lower bound with fill
                        opacity = 0.1 + (0.3 / n_cutoffs) * (
                            cutoff_idx + 1
                        )  # Increase opacity for more recent cutoffs
                        fig.add_trace(
                            go.Scatter(
                                x=df_cutoff["ds"],
                                y=df_cutoff[lo_col],
                                mode="lines",
                                line=dict(width=0),
                                fillcolor=f"rgba(100, 100, 100, {opacity})",
                                fill="tonexty",
                                name=f"{model} {level}% PI (cutoff: {cutoff})",
                                hoverinfo="skip",
                                legendgroup=f"{model}-{cutoff}",
                            )
                        )

    fig.update_layout(
        title=f"Germany Historical vs Forecast (Last {n_cutoffs} Cutoffs)",
        width=1400,
        height=500,
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01),
    )

    return fig


def plot_test_forecast3(df_hat, levels=None, n_cutoffs=10):
    """
    Plot historical data and forecasts with optional prediction intervals.
    Parameters:
    -----------
    df_hat : DataFrame
        Forecast data with columns 'ds', 'unique_id', model predictions, and optional prediction intervals
    levels : list of int, optional
        Confidence levels for prediction intervals (e.g., [80, 90])
    n_cutoffs : int, optional
        Number of most recent cutoffs to include (default: 10)
    """
    # Get the n most recent cutoffs
    unique_cutoffs = sorted(df_hat["cutoff"].unique())
    recent_cutoffs = unique_cutoffs[-n_cutoffs:]

    # Filter for recent cutoffs
    df_hat = df_hat[df_hat["cutoff"].isin(recent_cutoffs)].copy()

    # Sort by cutoff and ds to ensure proper ordering
    df_hat = df_hat.sort_values(["cutoff", "ds"])

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
