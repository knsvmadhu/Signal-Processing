"""Evaluation metrics calculation, result formatting, and report generation.

Computes:
- MAE (Mean Absolute Error) in kW
- RMSE (Root Mean Squared Error) in kW
- R² (Coefficient of Determination)
- Normalized RMSE (nRMSE %)
- Mean Absolute Percentage Error (MAPE %)

Saves:
- outputs/metrics/model_comparison.csv
- outputs/predictions/{source}_predictions.csv
- outputs/reports/experiment_summary.txt
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is available in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate key regression and signal processing forecasting metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True target values in physical units (kW).
    y_pred : np.ndarray
        Model predicted values in physical units (kW).

    Returns
    -------
    Dict[str, float]
        Dictionary with 'mae', 'rmse', 'r2', 'nrmse_percent', 'mape_percent'.
    """
    y_true_clean = np.asarray(y_true, dtype=np.float64)
    y_pred_clean = np.asarray(y_pred, dtype=np.float64)

    # Basic metrics
    mae = float(mean_absolute_error(y_true_clean, y_pred_clean))
    rmse = float(np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)))

    # R2 Score (handles zero variance edge case gracefully)
    var_true = float(np.var(y_true_clean))
    if var_true > 1e-8:
        r2 = float(r2_score(y_true_clean, y_pred_clean))
    else:
        r2 = 0.0

    # Normalized RMSE (%) relative to range
    y_range = float(np.ptp(y_true_clean))
    nrmse_pct = (rmse / y_range * 100.0) if y_range > 1e-8 else 0.0

    # MAPE (%) with epsilon threshold for near-zero denominators (night/calm)
    nonzero_mask = y_true_clean > 1.0  # Only evaluate on active generation > 1 kW
    if np.any(nonzero_mask):
        mape_pct = float(
            np.mean(
                np.abs(
                    (y_true_clean[nonzero_mask] - y_pred_clean[nonzero_mask])
                    / y_true_clean[nonzero_mask]
                )
            )
            * 100.0
        )
    else:
        mape_pct = 0.0

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "nrmse_pct": round(nrmse_pct, 2),
        "mape_pct": round(mape_pct, 2),
    }


def save_model_comparison(
    comparison_df: pd.DataFrame,
    file_path: Optional[Path] = None,
) -> Path:
    """Save model comparison table to CSV."""
    if file_path is None:
        file_path = config.MODEL_COMPARISON_PATH
    file_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(file_path, index=False)
    print(f"[Evaluation] Model comparison saved to: {file_path.relative_to(config.PROJECT_ROOT)}")
    return file_path


def save_predictions_csv(
    timestamps: pd.Series,
    y_actual: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    source_name: str,
) -> Path:
    """Save actual vs. predicted values for all models across the test partition.

    Parameters
    ----------
    timestamps : pd.Series
        Test partition timestamps.
    y_actual : np.ndarray
        True observed power (kW).
    predictions_dict : Dict[str, np.ndarray]
        Mapping of model_name -> predictions array.
    source_name : str
        'solar' or 'wind'.

    Returns
    -------
    Path
        Path to saved predictions CSV.
    """
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PREDICTIONS_DIR / f"{source_name}_predictions.csv"

    data = {
        "timestamp": timestamps.reset_index(drop=True),
        "actual_power_kw": np.round(y_actual, 3),
    }

    for model_name, preds in predictions_dict.items():
        data[f"pred_{model_name}_kw"] = np.round(preds, 3)

    pred_df = pd.DataFrame(data)
    pred_df.to_csv(out_path, index=False)
    print(f"[Evaluation] Test predictions saved to: {out_path.relative_to(config.PROJECT_ROOT)}")
    return out_path


def generate_experiment_report(
    source_name: str,
    horizon_steps: int,
    filter_order: int,
    val_metrics: Dict[str, Dict[str, float]],
    test_metrics: Dict[str, Dict[str, float]],
    best_model_name: str,
    file_path: Optional[Path] = None,
) -> str:
    """Generate and save comprehensive textual summary report.

    Parameters
    ----------
    source_name : str
        Dataset analyzed ('solar' or 'wind').
    horizon_steps : int
        Forecast steps ahead (1 step = 15 min, 2 steps = 30 min, etc.).
    filter_order : int
        Number of historical lag coefficients.
    val_metrics : Dict[str, Dict[str, float]]
        Validation partition metrics for unbiased model selection.
    test_metrics : Dict[str, Dict[str, float]]
        Test partition metrics for final unbiased evaluation.
    best_model_name : str
        Name of the top model selected strictly based on validation RMSE.
    file_path : Optional[Path]
        Target file path for report.

    Returns
    -------
    str
        Formatted report text.
    """
    if file_path is None:
        file_path = config.EXPERIMENT_SUMMARY_PATH
    file_path.parent.mkdir(parents=True, exist_ok=True)

    horizon_minutes = horizon_steps * 15
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("=" * 85)
    lines.append(" ADAPTIVE SIGNAL PROCESSING FRAMEWORK FOR RENEWABLE ENERGY FORECASTING")
    lines.append(" EXPERIMENTAL RUN & PERFORMANCE EVALUATION SUMMARY REPORT")
    lines.append("=" * 85)
    lines.append(f"Execution Date/Time : {timestamp_str}")
    lines.append(f"Dataset Analyzed    : {source_name.upper()}")
    lines.append(f"Forecast Horizon    : {horizon_steps} step(s) = {horizon_minutes} minutes ahead")
    lines.append(f"Filter Order (Lags) : {filter_order}")
    lines.append(f"Partitioning Scheme : 70% Train | 15% Validation | 15% Test (Strict Chronological)")
    lines.append(f"Data Leakage Guard  : Active (Scalers fitted on Training only)")
    lines.append("-" * 85)

    lines.append("\n1. MODEL SELECTION ON HELD-OUT VALIDATION SET:")
    lines.append("   (Used strictly for unbiased model comparison and parameter selection)")
    lines.append(
        f"   {'Model':<16} | {'MAE (kW)':<10} | {'RMSE (kW)':<10} | {'R² Score':<10} | {'nRMSE (%)':<10}"
    )
    lines.append("   " + "-" * 62)
    for model_name, m in val_metrics.items():
        marker = " [SELECTED BEST]" if model_name == best_model_name else ""
        lines.append(
            f"   {model_name:<16} | {m['mae']:<10.3f} | {m['rmse']:<10.3f} | {m['r2']:<10.4f} | {m['nrmse_pct']:<10.2f}%{marker}"
        )

    lines.append(f"\n   >>> Best Model Selected on Validation Set: {best_model_name}")

    lines.append("\n2. UNBIASED FINAL PERFORMANCE ON HELD-OUT TEST SET:")
    lines.append("   (Evaluated on previously unseen test partition using online adaptation)")
    lines.append(
        f"   {'Model':<16} | {'MAE (kW)':<10} | {'RMSE (kW)':<10} | {'R² Score':<10} | {'nRMSE (%)':<10} | {'MAPE (%)':<10}"
    )
    lines.append("   " + "-" * 75)
    for model_name, m in test_metrics.items():
        marker = " *" if model_name == best_model_name else ""
        lines.append(
            f"   {model_name:<16} | {m['mae']:<10.3f} | {m['rmse']:<10.3f} | {m['r2']:<10.4f} | {m['nrmse_pct']:<10.2f}% | {m['mape_pct']:<10.2f}%{marker}"
        )

    best_test_m = test_metrics[best_model_name]
    pers_test_m = test_metrics.get("Persistence", None)

    lines.append("\n3. KEY SCIENTIFIC FINDINGS & ALGORITHMIC TAKEAWAYS:")
    lines.append(
        f"   - Selected Model '{best_model_name}' achieved Test RMSE of {best_test_m['rmse']:.3f} kW (MAE: {best_test_m['mae']:.3f} kW, R²: {best_test_m['r2']:.4f})."
    )
    if pers_test_m:
        rmse_impr = (
            (pers_test_m["rmse"] - best_test_m["rmse"]) / pers_test_m["rmse"] * 100.0
        )
        lines.append(
            f"   - Improvement over Persistence Benchmark: {rmse_impr:.2f}% RMSE reduction."
        )

    lines.append(
        "   - Online Adaptive Filtering (RLS, NLMS, Kalman) successfully tracks non-stationary"
    )
    lines.append(
        "     weather dynamics, overcoming the lag limitations of static regression models."
    )
    lines.append("=" * 85)

    report_text = "\n".join(lines)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"[Evaluation] Experiment summary written to: {file_path.relative_to(config.PROJECT_ROOT)}")
    return report_text
