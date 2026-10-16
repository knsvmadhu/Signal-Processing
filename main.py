"""Adaptive Signal Processing Framework for Renewable Energy Forecasting.

Main execution entry point coordinating:
1. Automated raw dataset verification / generation
2. Chronological data preprocessing and zero-leakage lag feature construction
3. Signal decomposition (Trend + Diurnal + Residual)
4. Adaptive filter training, validation tuning, and online test tracking
5. Metric computation (MAE, RMSE, R², nRMSE, MAPE)
6. Comprehensive visualization generation and report generation
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is available in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

import config
from src.adaptive_filters import LMSFilter, NLMSFilter, RLSFilter, tune_filter_on_validation
from src.baseline_models import PersistenceBaseline, StaticRidgeBaseline
from src.data_generator import create_all_raw_datasets
from src.data_loader import load_dataset, resolve_dataset_source
from src.data_preprocessing import (
    TimeSeriesScaler,
    clean_dataset,
    chronological_split,
    create_lagged_features,
)
from src.decomposition import decompose_signal, save_decomposition_csv
from src.evaluation import (
    calculate_metrics,
    generate_experiment_report,
    save_model_comparison,
    save_predictions_csv,
)
from src.kalman_predictor import KalmanFilterPredictor
from src.visualization import (
    plot_actual_vs_predicted,
    plot_adaptive_learning_curves,
    plot_dataset_splits,
    plot_metrics_comparison,
    plot_models_comparison,
    plot_residual_errors,
    plot_signal_decomposition,
    plot_solar_timeseries,
    plot_wind_timeseries,
)


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Adaptive Signal Processing Framework for Renewable Energy Forecasting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="solar",
        help="Dataset identifier ('solar', 'wind', 'combined') or path to custom CSV file.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="all",
        choices=["all", "lms", "nlms", "rls", "kalman", "persistence", "ridge"],
        help="Adaptive filtering algorithm to evaluate or 'all' for comparative benchmark.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=config.DEFAULT_HORIZON,
        help="Forecast horizon in 15-minute time steps (1 = 15m, 2 = 30m, 4 = 60m).",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=None,
        help="Target column name for renewable power forecasting (default inferred from source).",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=config.DEFAULT_FILTER_ORDER,
        help="Adaptive filter order / number of historical lagged time steps.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable automatic plot generation.",
    )
    parser.add_argument(
        "--force-generate",
        action="store_true",
        help="Force regeneration of raw synthetic datasets even if they exist.",
    )
    return parser.parse_args()


def run_pipeline(
    source: str = "solar",
    filter_choice: str = "all",
    horizon: int = 1,
    target_col: Optional[str] = None,
    filter_order: int = config.DEFAULT_FILTER_ORDER,
    generate_plots: bool = True,
    force_generate: bool = False,
) -> Dict[str, Any]:
    """Execute end-to-end adaptive signal processing forecasting pipeline."""
    print("=" * 85)
    print("  ADAPTIVE SIGNAL PROCESSING FRAMEWORK FOR RENEWABLE ENERGY FORECASTING")
    print("=" * 85)
    horizon_minutes = horizon * 15
    print(f"Configuration: Source='{source}' | Horizon={horizon} step(s) ({horizon_minutes} mins) | Order={filter_order}")

    # 1. Check / Generate Raw Datasets
    if force_generate or not config.SOLAR_RAW_PATH.exists() or not config.WIND_RAW_PATH.exists():
        print("\n[Step 1/8] Verifying raw CSV datasets in data/raw/...")
        create_all_raw_datasets(output_dir=config.DATA_RAW_DIR, print_summary=True)
    else:
        print("\n[Step 1/8] Verified raw datasets present in data/raw/.")

    # 2. Load Selected Dataset
    print("\n[Step 2/8] Loading and validating dataset...")
    df_raw, raw_path, effective_target = load_dataset(
        source=source,
        target_col=target_col,
        auto_generate=True,
    )
    source_name = raw_path.stem.replace("_energy", "").replace("_processed", "")
    print(f"Loaded {raw_path.name} | Total Records: {len(df_raw)} | Target: '{effective_target}'")

    # 3. Clean Dataset and Enforce Boundaries
    print("\n[Step 3/8] Preprocessing, cleaning, and persisting to data/processed/...")
    df_clean, processed_path = clean_dataset(
        df_raw,
        target_col=effective_target,
        source_type=source_name,
        save_processed=True,
    )

    # 4. Signal Decomposition
    print("\n[Step 4/8] Performing signal decomposition (Trend + Diurnal + Residuals)...")
    power_series = df_clean[effective_target].values
    timestamps = df_clean["timestamp"]
    df_decomp = decompose_signal(power_series, timestamps, ma_window=config.DECOMPOSITION_MA_WINDOW)
    save_decomposition_csv(df_decomp, source_name)

    # 5. Chronological Train / Val / Test Partitioning
    print("\n[Step 5/8] Chronological data partitioning (70% Train, 15% Val, 15% Test)...")
    train_df, val_df, test_df = chronological_split(df_clean)

    # 6. Feature Construction & Normalization (Strictly Causal, No Data Leakage)
    print(f"\n[Step 6/8] Constructing causal tapped delay line features (Lags={filter_order}, Horizon={horizon})...")
    X_train_raw, y_train_raw = create_lagged_features(train_df[effective_target].values, order=filter_order, horizon=horizon)
    X_val_raw, y_val_raw = create_lagged_features(val_df[effective_target].values, order=filter_order, horizon=horizon)
    X_test_raw, y_test_raw = create_lagged_features(test_df[effective_target].values, order=filter_order, horizon=horizon)

    # Normalization fitted strictly on training data
    scaler = TimeSeriesScaler().fit(X_train_raw, y_train_raw)
    X_train = scaler.transform_x(X_train_raw)
    y_train = scaler.transform_y(y_train_raw)

    X_val = scaler.transform_x(X_val_raw)
    y_val = scaler.transform_y(y_val_raw)

    X_test = scaler.transform_x(X_test_raw)
    y_test = scaler.transform_y(y_test_raw)

    # Align test timestamps for target y(t + horizon)
    start_test_idx = (filter_order - 1) + horizon
    test_timestamps = test_df["timestamp"].iloc[start_test_idx:].reset_index(drop=True)
    # Match length to test samples
    test_timestamps = test_timestamps.iloc[:len(y_test_raw)].reset_index(drop=True)

    # 7. Model Instantiation & Adaptive Training
    print("\n[Step 7/8] Configuring and adapting models...")
    models = {}
    filter_choice_lower = filter_choice.lower()

    # Always include Persistence baseline for comparison
    models["Persistence"] = PersistenceBaseline(filter_order=filter_order)

    if filter_choice_lower in ["all", "ridge"]:
        models["StaticRidge"] = StaticRidgeBaseline(filter_order=filter_order, alpha=config.RIDGE_ALPHA)

    if filter_choice_lower in ["all", "lms"]:
        models["LMS"] = LMSFilter(filter_order=filter_order, mu=config.LMS_DEFAULT_MU)

    if filter_choice_lower in ["all", "nlms"]:
        models["NLMS"] = NLMSFilter(filter_order=filter_order, mu=config.NLMS_DEFAULT_MU, epsilon=config.NLMS_DEFAULT_EPSILON)

    if filter_choice_lower in ["all", "rls"]:
        models["RLS"] = RLSFilter(filter_order=filter_order, lam=config.RLS_DEFAULT_LAMBDA, delta=config.RLS_DEFAULT_DELTA)

    if filter_choice_lower in ["all", "kalman"]:
        models["Kalman"] = KalmanFilterPredictor(filter_order=filter_order, q_variance=config.KALMAN_Q_VAR, r_variance=config.KALMAN_R_VAR)

    learning_curves = {}
    val_metrics = {}
    test_metrics = {}
    test_predictions = {}

    print("\n--- Training on 70% partition & Evaluating on 15% Validation partition ---")
    for name, model in models.items():
        # Fit on training data
        model.fit(X_train, y_train)
        if hasattr(model, "learning_curve"):
            learning_curves[name] = model.learning_curve

        # Validation evaluation with online tracking
        val_preds_scaled, _ = model.predict(X_val, online_adapt=True, y_true=y_val)
        val_preds_kw = np.maximum(0.0, scaler.inverse_transform_y(val_preds_scaled))
        val_metrics[name] = calculate_metrics(y_val_raw, val_preds_kw)
        print(f"  {name:<14} | Val RMSE: {val_metrics[name]['rmse']:>7.3f} kW | Val MAE: {val_metrics[name]['mae']:>7.3f} kW | Val R^2: {val_metrics[name]['r2']:>6.4f}")

    # Select best model based on validation RMSE (unbiased selection)
    best_model_name = min(val_metrics.keys(), key=lambda k: val_metrics[k]["rmse"])
    print(f"\n>>> Best Model Selected based on Validation RMSE: '{best_model_name}'")

    print("\n--- Final Unbiased Evaluation on 15% Held-out Test Partition ---")
    for name, model in models.items():
        # Test evaluation with online adaptive tracking
        test_preds_scaled, _ = model.predict(X_test, online_adapt=True, y_true=y_test)
        test_preds_kw = np.maximum(0.0, scaler.inverse_transform_y(test_preds_scaled))
        test_predictions[name] = test_preds_kw
        test_metrics[name] = calculate_metrics(y_test_raw, test_preds_kw)
        marker = " [BEST]" if name == best_model_name else ""
        print(f"  {name:<14} | Test RMSE: {test_metrics[name]['rmse']:>7.3f} kW | Test MAE: {test_metrics[name]['mae']:>7.3f} kW | Test R^2: {test_metrics[name]['r2']:>6.4f}{marker}")

    # 8. Result Persistence & Visualization
    print("\n[Step 8/8] Saving predictions, metrics comparison, reports, and plots...")
    # Model comparison table
    comparison_rows = []
    for name in models.keys():
        row = {
            "Model": name,
            "Val_MAE": val_metrics[name]["mae"],
            "Val_RMSE": val_metrics[name]["rmse"],
            "Val_R2": val_metrics[name]["r2"],
            "Test_MAE": test_metrics[name]["mae"],
            "Test_RMSE": test_metrics[name]["rmse"],
            "Test_R2": test_metrics[name]["r2"],
            "Test_nRMSE_%": test_metrics[name]["nrmse_pct"],
            "Test_MAPE_%": test_metrics[name]["mape_pct"],
            "Selected_Best": (name == best_model_name),
        }
        comparison_rows.append(row)
    comparison_df = pd.DataFrame(comparison_rows)
    save_model_comparison(comparison_df)

    # Save test predictions
    save_predictions_csv(test_timestamps, y_test_raw, test_predictions, source_name)

    # Generate experiment summary report
    report_text = generate_experiment_report(
        source_name=source_name,
        horizon_steps=horizon,
        filter_order=filter_order,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        best_model_name=best_model_name,
    )

    # Generate all plots
    if generate_plots:
        print("\nGenerating and saving plots in outputs/graphs/...")
        # 1 & 2. Time-series overview
        if source_name == "solar":
            plot_solar_timeseries(df_clean)
        elif source_name == "wind":
            plot_wind_timeseries(df_clean)

        # 3. Actual vs Predicted
        plot_actual_vs_predicted(
            test_timestamps,
            y_test_raw,
            test_predictions[best_model_name],
            best_model_name,
            horizon_steps=horizon,
        )

        # 4. Multi-model comparison
        plot_models_comparison(test_timestamps, y_test_raw, test_predictions)

        # 5. Learning curves
        if learning_curves:
            plot_adaptive_learning_curves(learning_curves)

        # 6. Residual errors
        plot_residual_errors(y_test_raw, test_predictions)

        # 7. Metrics comparison bar chart
        metrics_bar_df = pd.DataFrame(
            {
                "Model": list(test_metrics.keys()),
                "MAE": [test_metrics[m]["mae"] for m in test_metrics],
                "RMSE": [test_metrics[m]["rmse"] for m in test_metrics],
            }
        )
        plot_metrics_comparison(metrics_bar_df)

        # 8. Signal decomposition
        plot_signal_decomposition(df_decomp)

        # 9. Dataset splits
        plot_dataset_splits(train_df, val_df, test_df, target_col=effective_target)

    print("\n" + "=" * 85)
    print("PIPELINE EXECUTION SUCCESSFULLY COMPLETED!")
    print(f"Top Performer on Test Data: {best_model_name} (RMSE={test_metrics[best_model_name]['rmse']} kW)")
    print(f"Reports & Metrics : outputs/reports/, outputs/metrics/")
    print(f"Predictions       : outputs/predictions/")
    print(f"Graphs            : outputs/graphs/")
    print("=" * 85)

    return {
        "best_model": best_model_name,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "comparison_df": comparison_df,
    }


def main():
    args = parse_arguments()
    run_pipeline(
        source=args.source,
        filter_choice=args.filter,
        horizon=args.horizon,
        target_col=args.target_col,
        filter_order=args.order,
        generate_plots=not args.no_plots,
        force_generate=args.force_generate,
    )


if __name__ == "__main__":
    main()
