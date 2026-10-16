"""Publication-grade visualization suite for renewable energy forecasting.

Generates and persists:
1. solar_power_timeseries.png: Solar PV generation and atmospheric variables.
2. wind_power_timeseries.png: Wind turbine output, wind speed, and pressure.
3. actual_vs_predicted.png: Actual vs predicted power for the best model.
4. models_comparison.png: Multi-model comparison over test window.
5. adaptive_learning_curves.png: LMS, NLMS, and RLS squared error convergence.
6. residual_errors.png: Residual error time-series and error distributions.
7. metrics_comparison.png: MAE and RMSE comparative bar charts.
8. signal_decomposition.png: Moving-average trend, diurnal cycle, and residuals.
9. dataset_splits.png: Chronological train/val/test partition visualization.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is available in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
# Headless backend for stable script execution
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def set_plot_style() -> None:
    """Set clean, publication-quality styling parameters."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8


def plot_solar_timeseries(
    df_solar: pd.DataFrame,
    save_path: Optional[Path] = None,
    n_display: int = 672,  # 7 days of 15-min data
) -> Path:
    """Plot solar power time-series alongside irradiance and temperature."""
    set_plot_style()
    if save_path is None:
        save_path = config.GRAPHS_DIR / "solar_power_timeseries.png"

    subset = df_solar.iloc[:n_display].copy()
    ts = pd.to_datetime(subset["timestamp"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # 1. Solar Power
    axes[0].plot(ts, subset["power_kw"], color="#d95f02", lw=1.8, label="PV Power Output (kW)")
    axes[0].set_ylabel("Power (kW)", fontweight="bold")
    axes[0].set_title("Solar Photovoltaic Generation & Environmental Profiles (Sample Week)", fontsize=13, fontweight="bold")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Solar Irradiance
    axes[1].plot(ts, subset["solar_irradiance"], color="#e7298a", lw=1.5, label="Solar Irradiance (W/m²)")
    axes[1].set_ylabel("GHI (W/m²)", fontweight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # 3. Temperature & Cloud Cover
    ax3_twin = axes[2].twinx()
    axes[2].plot(ts, subset["temperature"], color="#1b9e77", lw=1.5, label="Temperature (°C)")
    ax3_twin.plot(ts, subset["cloud_cover"], color="#7570b3", lw=1.2, linestyle="--", label="Cloud Cover (%)")

    axes[2].set_ylabel("Temp (°C)", fontweight="bold")
    ax3_twin.set_ylabel("Cloud Cover (%)", fontweight="bold")
    axes[2].set_xlabel("Observation Timestamp", fontweight="bold")
    axes[2].legend(loc="upper left")
    ax3_twin.legend(loc="upper right")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved: {save_path.name}")
    return save_path


def plot_wind_timeseries(
    df_wind: pd.DataFrame,
    save_path: Optional[Path] = None,
    n_display: int = 672,  # 7 days of 15-min data
) -> Path:
    """Plot wind power time-series alongside wind speed and atmospheric pressure."""
    set_plot_style()
    if save_path is None:
        save_path = config.GRAPHS_DIR / "wind_power_timeseries.png"

    subset = df_wind.iloc[:n_display].copy()
    ts = pd.to_datetime(subset["timestamp"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # 1. Wind Power
    axes[0].plot(ts, subset["power_kw"], color="#02818a", lw=1.8, label="Wind Turbine Power (kW)")
    axes[0].axhline(y=config.WIND_TURBINE_RATED_KW, color="red", linestyle=":", alpha=0.7, label=f"Rated Capacity ({config.WIND_TURBINE_RATED_KW} kW)")
    axes[0].set_ylabel("Power (kW)", fontweight="bold")
    axes[0].set_title("Wind Turbine Power Generation & Atmospheric Dynamics (Sample Week)", fontsize=13, fontweight="bold")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Wind Speed
    axes[1].plot(ts, subset["wind_speed"], color="#67a9cf", lw=1.5, label="Wind Speed (m/s)")
    axes[1].axhline(y=config.WIND_CUT_IN_SPEED, color="orange", linestyle="--", alpha=0.8, label=f"Cut-in Speed ({config.WIND_CUT_IN_SPEED} m/s)")
    axes[1].axhline(y=config.WIND_RATED_SPEED, color="green", linestyle="--", alpha=0.8, label=f"Rated Speed ({config.WIND_RATED_SPEED} m/s)")
    axes[1].set_ylabel("Speed (m/s)", fontweight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # 3. Atmospheric Pressure & Direction
    ax3_twin = axes[2].twinx()
    axes[2].plot(ts, subset["pressure"], color="#756bb1", lw=1.5, label="Atmospheric Pressure (hPa)")
    ax3_twin.plot(ts, subset["wind_direction"], color="#bcbddc", lw=1.0, linestyle=":", label="Wind Direction (°)")

    axes[2].set_ylabel("Pressure (hPa)", fontweight="bold")
    ax3_twin.set_ylabel("Direction (°)", fontweight="bold")
    axes[2].set_xlabel("Observation Timestamp", fontweight="bold")
    axes[2].legend(loc="upper left")
    ax3_twin.legend(loc="upper right")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved: {save_path.name}")
    return save_path


def plot_dataset_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "power_kw",
    save_path: Optional[Path] = None,
) -> Path:
    """Visualize chronological train, validation, and test split partitions."""
    set_plot_style()
    if save_path is None:
        save_path = config.GRAPHS_DIR / "dataset_splits.png"

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(pd.to_datetime(train_df["timestamp"]), train_df[target_col], color="#2b83ba", lw=1.0, label=f"Training Set ({len(train_df)} samples, 70%)")
    ax.plot(pd.to_datetime(val_df["timestamp"]), val_df[target_col], color="#fdae61", lw=1.0, label=f"Validation Set ({len(val_df)} samples, 15%)")
    ax.plot(pd.to_datetime(test_df["timestamp"]), test_df[target_col], color="#d7191c", lw=1.0, label=f"Test Set ({len(test_df)} samples, 15%)")

    # Add vertical partition boundaries
    val_start = pd.to_datetime(val_df["timestamp"].iloc[0])
    test_start = pd.to_datetime(test_df["timestamp"].iloc[0])
    ax.axvline(x=val_start, color="black", linestyle="--", alpha=0.8, lw=1.5)
    ax.axvline(x=test_start, color="black", linestyle="--", alpha=0.8, lw=1.5)

    ax.set_title("Chronological Partitioning: Training (70%), Validation (15%), Test (15%) - Zero Leakage", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{target_col} (kW)", fontweight="bold")
    ax.set_xlabel("Timestamp", fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved: {save_path.name}")
    return save_path


def plot_actual_vs_predicted(
    timestamps: pd.Series,
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    horizon_steps: int,
    save_path: Optional[Path] = None,
    n_display: int = 384,  # 4 days of 15-min data
) -> Path:
    """Plot actual vs predicted power for the selected top model."""
    set_plot_style()
    if save_path is None:
        save_path = config.GRAPHS_DIR / "actual_vs_predicted.png"

    ts = pd.to_datetime(timestamps).iloc[:n_display]
    y_act = y_actual[:n_display]
    y_pr = y_pred[:n_display]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(ts, y_act, color="#252525", lw=1.8, label="Actual Power (Observed)")
    ax1.plot(ts, y_pr, color="#02818a", lw=1.6, linestyle="--", label=f"Forecast ({model_name}, {horizon_steps * 15}m ahead)")
    ax1.set_ylabel("Power (kW)", fontweight="bold")
    ax1.set_title(f"Actual vs. Predicted Power Output ({model_name} Model - Held-out Test Partition)", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Tracking error
    err = y_act - y_pr
    ax2.plot(ts, err, color="#e41a1c", lw=1.2, label="Prediction Error e(t)")
    ax2.axhline(0, color="black", linestyle=":", lw=1.0)
    ax2.set_ylabel("Error (kW)", fontweight="bold")
    ax2.set_xlabel("Observation Timestamp", fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved: {save_path.name}")
    return save_path


def plot_models_comparison(
    timestamps: pd.Series,
    y_actual: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
    n_display: int = 288,  # 3 days
) -> Path:
    """Plot comparison of all forecasting models over a representative test window."""
    set_plot_style()
    if save_path is None:
        save_path = config.GRAPHS_DIR / "models_comparison.png"

    ts = pd.to_datetime(timestamps).iloc[:n_display]
    y_act = y_actual[:n_display]

    palette = {
        "Persistence": "#969696",
        "StaticRidge": "#fe9929",
        "LMS": "#3690c0",
        "NLMS": "#238b45",
        "RLS": "#88419d",
        "Kalman": "#02818a",
    }

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(ts, y_act, color="#000000", lw=2.2, label="Actual Power", zorder=5)

    for name, preds in predictions_dict.items():
        color = palette.get(name, "#4daf4a")
        style = "--" if name in ["Persistence", "StaticRidge"] else "-"
        ax.plot(ts, preds[:n_display], color=color, lw=1.5, linestyle=style, label=f"{name}", alpha=0.85)

    ax.set_title("Multi-Model Renewable Power Forecasting Comparison (Test Partition)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Power Output (kW)", fontweight="bold")
    ax.set_xlabel("Timestamp", fontweight="bold")
    ax.legend(loc="upper right", frameon=True, ncol=3)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved: {save_path.name}")
    return save_path


def plot_adaptive_learning_curves(
    learning_curves: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    window: int = 50,
) -> Path:
    """Plot LMS, NLMS, and RLS adaptation error convergence curves."""
    set_plot_style()
    if save_path is None:
        save_path = config.GRAPHS_DIR / "adaptive_learning_curves.png"

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {"LMS": "#3690c0", "NLMS": "#238b45", "RLS": "#88419d", "Kalman": "#02818a"}

    for name, errors in learning_curves.items():
        if not errors:
            continue
        err_arr = np.array(errors, dtype=np.float64)
        # Apply smoothing window for clear visual comparison
        if len(err_arr) > window:
            kernel = np.ones(window) / float(window)
            smoothed = np.convolve(err_arr, kernel, mode="valid")
            x_vals = np.arange(window // 2, window // 2 + len(smoothed))
        else:
            smoothed = err_arr
            x_vals = np.arange(len(smoothed))

        color = colors.get(name, "#555555")
        ax.plot(x_vals, smoothed, label=f"{name} (Smoothed MSE)", color=color, lw=1.8)

    ax.set_yscale("log")
    ax.set_title("Adaptive Signal Processing Learning Curves: Instantaneous Squared Error Convergence", fontsize=12, fontweight="bold")
    ax.set_ylabel("Log Mean Squared Error (MSE)", fontweight="bold")
    ax.set_xlabel("Iteration / Time Step k", fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved: {save_path.name}")
    return save_path


def plot_residual_errors(
    y_actual: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
) -> Path:
    """Plot residual error time-series and error histograms across models."""
    set_plot_style()
    if save_path is None:
        save_path = config.GRAPHS_DIR / "residual_errors.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    palette = {"Persistence": "#969696", "StaticRidge": "#fe9929", "LMS": "#3690c0", "NLMS": "#238b45", "RLS": "#88419d", "Kalman": "#02818a"}

    # Left: Residual distributions
    for name, preds in predictions_dict.items():
        res = y_actual - preds
        color = palette.get(name, "#333333")
        ax1.hist(res, bins=40, density=True, alpha=0.35, color=color, label=f"{name} (std={np.std(res):.2f})")

    ax1.set_title("Residual Error Probability Distributions (e = y - y_hat)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Residual Error (kW)", fontweight="bold")
    ax1.set_ylabel("Probability Density", fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Right: Residual time-series for top 3 models
    top_models = [m for m in ["RLS", "Kalman", "NLMS", "LMS"] if m in predictions_dict][:3]
    for name in top_models:
        res = y_actual[:300] - predictions_dict[name][:300]
        color = palette.get(name, "#333333")
        ax2.plot(res, label=f"{name} Residual", color=color, lw=1.2, alpha=0.85)

    ax2.axhline(0, color="black", linestyle="--", lw=1.0)
    ax2.set_title("Test Residual Sequence Tracking (First 300 Steps)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Sample Step k", fontweight="bold")
    ax2.set_ylabel("Residual Error (kW)", fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved: {save_path.name}")
    return save_path


def plot_metrics_comparison(
    metrics_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> Path:
    """Plot MAE and RMSE comparative bar charts across all tested models."""
    set_plot_style()
    if save_path is None:
        save_path = config.GRAPHS_DIR / "metrics_comparison.png"

    models = metrics_df["Model"].values
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    rects1 = ax.bar(x - width / 2, metrics_df["MAE"].values, width, label="MAE (kW)", color="#41b6c4", edgecolor="#253494", lw=1.2)
    rects2 = ax.bar(x + width / 2, metrics_df["RMSE"].values, width, label="RMSE (kW)", color="#225ea8", edgecolor="#081d58", lw=1.2)

    ax.set_ylabel("Error in Physical Units (kW)", fontweight="bold")
    ax.set_title("Forecasting Performance Benchmark: MAE and RMSE Comparison", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Attach value labels
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}", xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}", xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved: {save_path.name}")
    return save_path


def plot_signal_decomposition(
    df_decomp: pd.DataFrame,
    save_path: Optional[Path] = None,
    n_display: int = 672,  # 7 days
) -> Path:
    """Plot additive signal decomposition: Original, Trend, Diurnal Seasonality, Residuals."""
    set_plot_style()
    if save_path is None:
        save_path = config.GRAPHS_DIR / "signal_decomposition.png"

    subset = df_decomp.iloc[:n_display]
    ts = pd.to_datetime(subset["timestamp"])

    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(ts, subset["original"], color="#252525", lw=1.5)
    axes[0].set_ylabel("Original (kW)", fontweight="bold")
    axes[0].set_title("Multi-Component Signal Decomposition: y(t) = Trend(t) + Diurnal(t) + Residual(t)", fontsize=12, fontweight="bold")

    axes[1].plot(ts, subset["trend"], color="#2b83ba", lw=1.8)
    axes[1].set_ylabel("Trend (kW)", fontweight="bold")

    axes[2].plot(ts, subset["diurnal"], color="#fdae61", lw=1.5)
    axes[2].set_ylabel("Diurnal (kW)", fontweight="bold")

    axes[3].plot(ts, subset["residual"], color="#d7191c", lw=1.0)
    axes[3].axhline(0, color="black", linestyle=":", lw=1.0)
    axes[3].set_ylabel("Residual (kW)", fontweight="bold")
    axes[3].set_xlabel("Timestamp", fontweight="bold")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved: {save_path.name}")
    return save_path
