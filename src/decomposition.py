"""Signal decomposition module for renewable energy time-series analysis.

Implements:
1. Moving-Average Trend Extraction & Diurnal Profile Residual Analysis:
   y(t) = Trend(t) + Diurnal(t) + Residual(t)
2. Multi-Resolution Wavelet / Filter-Bank Decomposition (Pyramid DWT) in pure NumPy/SciPy
   to isolate synoptic trend, meso-scale variations, and turbulent fluctuations.
3. CSV persistence and statistical residual diagnostics.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Ensure project root is available in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.signal import convolve

import config


def moving_average_trend(series: np.ndarray, window_size: int = config.DECOMPOSITION_MA_WINDOW) -> np.ndarray:
    """Extract smooth low-frequency trend using a centered moving average.

    Parameters
    ----------
    series : np.ndarray
        1D power signal array.
    window_size : int
        Window length (default 96 steps = 24 hours of 15-min intervals).

    Returns
    -------
    np.ndarray
        Estimated smooth trend of the same length as series.
    """
    n = len(series)
    trend = np.full(n, np.nan, dtype=np.float64)

    # Use centered window with edge reflection/padding
    half_w = window_size // 2
    pad_front = np.repeat(series[0], half_w)
    pad_back = np.repeat(series[-1], half_w)
    padded = np.concatenate([pad_front, series, pad_back])

    kernel = np.ones(window_size) / float(window_size)
    smoothed = convolve(padded, kernel, mode="valid")

    # Match length exactly
    trend = smoothed[:n]
    return trend


def extract_diurnal_profile(
    detrended: np.ndarray,
    timestamps: pd.Series,
    period_steps: int = 96,
) -> np.ndarray:
    """Compute average 24-hour diurnal seasonal component.

    Parameters
    ----------
    detrended : np.ndarray
        Signal with moving-average trend subtracted.
    timestamps : pd.Series
        Datetime timestamps corresponding to series.
    period_steps : int
        Number of steps per day (96 for 15-min frequency).

    Returns
    -------
    np.ndarray
        Cyclic diurnal seasonal component.
    """
    ts = pd.to_datetime(timestamps)
    # Calculate step index in the day: 0 to 95
    if isinstance(ts, pd.DatetimeIndex):
        step_of_day = (ts.hour * 4 + ts.minute // 15) % period_steps
    else:
        step_of_day = (ts.dt.hour * 4 + ts.dt.minute // 15) % period_steps
    if hasattr(step_of_day, "values"):
        step_of_day = step_of_day.values

    # Compute mean for each step of day
    diurnal_profile = np.zeros(period_steps)
    for s in range(period_steps):
        mask = step_of_day == s
        if np.any(mask):
            diurnal_profile[s] = np.mean(detrended[mask])

    return diurnal_profile[step_of_day]


def multiresolution_wavelet_decompose(
    series: np.ndarray,
    levels: int = config.MULTIRESOLUTION_LEVELS,
) -> Dict[str, np.ndarray]:
    """Perform multi-resolution stationary wavelet decomposition using Haar wavelets.

    Decomposes the renewable signal into low-frequency baseline approximation (A)
    and hierarchical detail components (D1, D2, D3) capturing fast turbulence,
    hourly shifts, and multi-hour atmospheric changes.

    Parameters
    ----------
    series : np.ndarray
        1D input signal.
    levels : int
        Number of hierarchical resolution levels (default 3).

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with keys 'approximation', 'detail_1', 'detail_2', ..., 'reconstruction'
    """
    current_approx = series.copy().astype(np.float64)
    details = {}

    for lvl in range(1, levels + 1):
        # Scale factor 2^(lvl-1)
        step = 2 ** (lvl - 1)
        next_approx = np.zeros_like(current_approx)
        detail = np.zeros_like(current_approx)

        # Haar moving average filter at scale step
        for i in range(len(series)):
            i_prev = max(0, i - step)
            next_approx[i] = 0.5 * (current_approx[i] + current_approx[i_prev])
            detail[i] = 0.5 * (current_approx[i] - current_approx[i_prev])

        details[f"detail_{lvl}"] = detail
        current_approx = next_approx

    decomp_dict = {"approximation": current_approx}
    decomp_dict.update(details)

    # Reconstructed signal verification
    recon = current_approx + sum(details.values())
    decomp_dict["reconstruction"] = recon

    return decomp_dict


def decompose_signal(
    series: np.ndarray,
    timestamps: pd.Series,
    ma_window: int = config.DECOMPOSITION_MA_WINDOW,
    diurnal_period: int = 96,
) -> pd.DataFrame:
    """Execute complete additive signal decomposition: y(t) = Trend(t) + Diurnal(t) + Residual(t).

    Parameters
    ----------
    series : np.ndarray
        1D renewable power time-series.
    timestamps : pd.Series
        Datetime timestamps.
    ma_window : int
        Window for trend extraction.
    diurnal_period : int
        Number of steps per diurnal cycle (96 for 15-min intervals).

    Returns
    -------
    pd.DataFrame
        DataFrame with ['timestamp', 'original', 'trend', 'diurnal', 'residual']
    """
    trend = moving_average_trend(series, window_size=ma_window)
    detrended = series - trend
    diurnal = extract_diurnal_profile(detrended, timestamps, period_steps=diurnal_period)
    residual = series - trend - diurnal

    df_decomp = pd.DataFrame(
        {
            "timestamp": timestamps,
            "original": np.round(series, 3),
            "trend": np.round(trend, 3),
            "diurnal": np.round(diurnal, 3),
            "residual": np.round(residual, 3),
        }
    )
    return df_decomp


def save_decomposition_csv(df_decomp: pd.DataFrame, source_name: str) -> Path:
    """Save decomposed components to outputs/reports/."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.REPORTS_DIR / f"decomposition_{source_name}.csv"
    df_decomp.to_csv(out_path, index=False)
    print(f"[Decomposition] Saved decomposition table to: {out_path.relative_to(config.PROJECT_ROOT)}")
    return out_path


if __name__ == "__main__":
    from src.data_loader import load_dataset
    df, _, target = load_dataset("solar")
    df_decomp = decompose_signal(df[target].values, df["timestamp"])
    save_decomposition_csv(df_decomp, "solar")
    print(df_decomp.head())
