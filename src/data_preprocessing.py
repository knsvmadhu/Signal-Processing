"""Data cleaning, preprocessing, chronological partitioning, and lag-feature extraction.

Strictly preserves causality and prevents future data leakage:
- Normalization parameters (mean/std or min/max) are fitted solely on the training partition.
- Chronological train / validation / test partitioning (e.g., 70% / 15% / 15%).
- Causal tapped delay line feature generation for multi-step ahead forecasting.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is available in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

import config


def clean_dataset(
    df: pd.DataFrame,
    target_col: str = "power_kw",
    source_type: str = "auto",
    save_processed: bool = True,
) -> Tuple[pd.DataFrame, Path]:
    """Clean renewable energy dataset, handle missing values, and enforce physical limits.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset with parsed 'timestamp' column.
    target_col : str
        Target forecasting column name.
    source_type : str
        'solar', 'wind', 'combined', or 'auto' (inferred from columns).
    save_processed : bool
        If True, saves the processed dataset in config.DATA_PROCESSED_DIR.

    Returns
    -------
    Tuple[pd.DataFrame, Path]
        (cleaned_df, saved_processed_path)
    """
    df_clean = df.copy()

    # 1. Infer source type if 'auto'
    if source_type == "auto":
        cols = set(df_clean.columns)
        if "solar_irradiance" in cols and "wind_speed" in cols:
            source_type = "combined"
        elif "solar_irradiance" in cols:
            source_type = "solar"
        elif "wind_speed" in cols:
            source_type = "wind"
        else:
            source_type = "custom"

    # 2. Missing value imputation using causal forward-fill followed by backward-fill
    # (Avoids lookahead bias while handling short sensor dropouts)
    initial_nans = int(df_clean.isna().sum().sum())
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

    # Interpolate time gaps linearly, then fill remaining edges
    df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(
        method="linear", limit=config.MAX_CONSECUTIVE_MISSING
    )
    df_clean[numeric_cols] = df_clean[numeric_cols].ffill().bfill()
    remaining_nans = int(df_clean.isna().sum().sum())

    if initial_nans > 0:
        print(
            f"[Preprocessing] Resolved {initial_nans - remaining_nans}/{initial_nans} missing values."
        )

    # 3. Physical Boundary Enforcement
    if target_col in df_clean.columns:
        df_clean[target_col] = np.maximum(0.0, df_clean[target_col])

    if source_type == "solar":
        if "solar_irradiance" in df_clean.columns:
            df_clean["solar_irradiance"] = np.clip(df_clean["solar_irradiance"], 0.0, 1300.0)
        if "humidity" in df_clean.columns:
            df_clean["humidity"] = np.clip(df_clean["humidity"], 0.0, 100.0)
        if "cloud_cover" in df_clean.columns:
            df_clean["cloud_cover"] = np.clip(df_clean["cloud_cover"], 0.0, 100.0)
        if target_col == "power_kw":
            df_clean[target_col] = np.clip(df_clean[target_col], 0.0, config.SOLAR_CAPACITY_KW)

    elif source_type == "wind":
        if "wind_speed" in df_clean.columns:
            df_clean["wind_speed"] = np.clip(df_clean["wind_speed"], 0.0, 50.0)
        if "wind_direction" in df_clean.columns:
            df_clean["wind_direction"] = df_clean["wind_direction"] % 360.0
        if "pressure" in df_clean.columns:
            df_clean["pressure"] = np.clip(df_clean["pressure"], 800.0, 1100.0)
        if target_col == "power_kw":
            df_clean[target_col] = np.clip(df_clean[target_col], 0.0, config.WIND_TURBINE_RATED_KW)

    # 4. Save processed file
    processed_path = config.DATA_PROCESSED_DIR / f"{source_type}_processed.csv"
    if save_processed:
        config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(processed_path, index=False)
        print(f"[Preprocessing] Saved cleaned data to: {processed_path.relative_to(config.PROJECT_ROOT)}")

    return df_clean, processed_path


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time-series data chronologically into train, validation, and test sets.

    Ensures zero temporal overlap and guarantees strict causal separation.

    Parameters
    ----------
    df : pd.DataFrame
        Chronologically sorted DataFrame.
    train_ratio : float
        Proportion for training (default 0.70).
    val_ratio : float
        Proportion for hyperparameter tuning / model validation (default 0.15).
    test_ratio : float
        Proportion for final unbiased held-out evaluation (default 0.15).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    total_len = len(df)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    print(
        f"[Split] Chronological Partition: "
        f"Train={len(train_df)} ({train_df['timestamp'].iloc[0]} to {train_df['timestamp'].iloc[-1]}) | "
        f"Val={len(val_df)} ({val_df['timestamp'].iloc[0]} to {val_df['timestamp'].iloc[-1]}) | "
        f"Test={len(test_df)} ({test_df['timestamp'].iloc[0]} to {test_df['timestamp'].iloc[-1]})"
    )

    return train_df, val_df, test_df


def create_lagged_features(
    series: np.ndarray,
    order: int = config.DEFAULT_FILTER_ORDER,
    horizon: int = config.DEFAULT_HORIZON,
    exogenous: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct causal tapped delay line feature matrix X and target vector y.

    For forecasting horizon `h` steps ahead:
    At forecast origin index `t` (where `order - 1 <= t < N - h`):
    - Input vector x[t] = [y[t], y[t-1], ..., y[t - order + 1], exog_features(t)]^T
    - Target d[t] = y[t + h]

    Parameters
    ----------
    series : np.ndarray
        1D target time series array (e.g. power in kW).
    order : int
        Filter order / tapped delay line length (number of historical lags).
    horizon : int
        Prediction horizon in time steps (1 step = 15 min, 2 steps = 30 min, etc.).
    exogenous : Optional[np.ndarray]
        2D array of shape (N, n_exog) containing contemporaneous exogenous variables.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X : Feature matrix of shape (num_samples, order + n_exog)
        y : Target array of shape (num_samples,)
    """
    n_total = len(series)
    # Valid indices where we have 'order' past values and 'horizon' future value
    # Start: order - 1 (has lags 0 through order-1)
    # End: n_total - horizon (target is t + horizon < n_total)
    start_idx = order - 1
    end_idx = n_total - horizon

    if end_idx <= start_idx:
        raise ValueError(
            f"Series length {n_total} is too short for filter order {order} and horizon {horizon}."
        )

    num_samples = end_idx - start_idx
    lag_matrix = np.zeros((num_samples, order), dtype=np.float64)

    for i, t in enumerate(range(start_idx, end_idx)):
        # x(t) = [y(t), y(t-1), ..., y(t - order + 1)]
        lag_matrix[i, :] = series[t - order + 1 : t + 1][::-1]

    if exogenous is not None:
        if exogenous.ndim == 1:
            exog_slice = exogenous[start_idx:end_idx, np.newaxis]
        else:
            exog_slice = exogenous[start_idx:end_idx, :]
        X = np.hstack([lag_matrix, exog_slice])
    else:
        X = lag_matrix

    y = np.array([series[t + horizon] for t in range(start_idx, end_idx)], dtype=np.float64)

    return X, y


class TimeSeriesScaler:
    """Zero-leakage normalization scaler for time-series features and targets.

    Fitted strictly on training data statistics (mean and standard deviation)
    and applied consistently across validation and test partitions.
    """

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.mean_x: Optional[np.ndarray] = None
        self.std_x: Optional[np.ndarray] = None
        self.mean_y: Optional[float] = None
        self.std_y: Optional[float] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "TimeSeriesScaler":
        """Compute mean and standard deviation on training set only."""
        self.mean_x = np.mean(X_train, axis=0)
        self.std_x = np.std(X_train, axis=0)
        self.std_x[self.std_x < self.epsilon] = 1.0  # Avoid zero division for flat features

        self.mean_y = float(np.mean(y_train))
        self.std_y = float(np.std(y_train))
        if self.std_y < self.epsilon:
            self.std_y = 1.0

        return self

    def transform_x(self, X: np.ndarray) -> np.ndarray:
        """Standardize feature matrix using training statistics."""
        if self.mean_x is None or self.std_x is None:
            raise RuntimeError("TimeSeriesScaler has not been fitted yet.")
        return (X - self.mean_x) / self.std_x

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        """Standardize target array using training statistics."""
        if self.mean_y is None or self.std_y is None:
            raise RuntimeError("TimeSeriesScaler has not been fitted yet.")
        return (y - self.mean_y) / self.std_y

    def inverse_transform_y(self, y_scaled: np.ndarray) -> np.ndarray:
        """Convert standardized predictions back to original physical units (kW)."""
        if self.mean_y is None or self.std_y is None:
            raise RuntimeError("TimeSeriesScaler has not been fitted yet.")
        return y_scaled * self.std_y + self.mean_y
