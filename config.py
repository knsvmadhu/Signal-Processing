"""Configuration parameters for the Adaptive Signal Processing Framework for Renewable Energy Forecasting.

Provides centralized paths, physical constants, simulation parameters,
data split ratios, and filter hyperparameter specifications.
"""

from pathlib import Path

# ==============================================================================
# Path Configurations (Cross-platform compatibility using pathlib)
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GRAPHS_DIR = OUTPUTS_DIR / "graphs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
REPORTS_DIR = OUTPUTS_DIR / "reports"
METRICS_DIR = OUTPUTS_DIR / "metrics"

# Raw dataset paths
SOLAR_RAW_PATH = DATA_RAW_DIR / "solar_energy.csv"
WIND_RAW_PATH = DATA_RAW_DIR / "wind_energy.csv"
COMBINED_RAW_PATH = DATA_RAW_DIR / "combined_renewable_energy.csv"

# Processed dataset paths
SOLAR_PROCESSED_PATH = DATA_PROCESSED_DIR / "solar_processed.csv"
WIND_PROCESSED_PATH = DATA_PROCESSED_DIR / "wind_processed.csv"

# Output artifacts paths
MODEL_COMPARISON_PATH = METRICS_DIR / "model_comparison.csv"
SOLAR_PREDICTIONS_PATH = PREDICTIONS_DIR / "solar_predictions.csv"
WIND_PREDICTIONS_PATH = PREDICTIONS_DIR / "wind_predictions.csv"
EXPERIMENT_SUMMARY_PATH = REPORTS_DIR / "experiment_summary.txt"

# Ensure runtime directories exist
for directory in [
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    GRAPHS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    METRICS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Simulation & Synthetic Data Generation Parameters
# ==============================================================================
RANDOM_SEED = 42
DEFAULT_SAMPLE_COUNT = 6000  # >= 5,000 records required (6,000 steps * 15 min = 62.5 days)
SAMPLE_FREQUENCY = "15min"
SIMULATION_START_DATE = "2026-01-01 00:00:00"

# Solar Photovoltaic (PV) System Parameters
SOLAR_CAPACITY_KW = 100.0          # Nameplate capacity in kW
SOLAR_PANEL_AREA_M2 = 550.0        # Effective PV surface area in square meters
SOLAR_BASE_EFFICIENCY = 0.19       # Nominal panel conversion efficiency (19%)
SOLAR_TEMP_COEFF = -0.004          # Power temperature coefficient (-0.4%/°C relative to 25°C)
SOLAR_INVERTER_EFFICIENCY = 0.96   # Inverter AC/DC conversion efficiency

# Wind Turbine System Parameters
WIND_TURBINE_RATED_KW = 500.0      # Rated power capacity in kW
WIND_CUT_IN_SPEED = 3.0            # Cut-in wind speed in m/s (turbine activates)
WIND_RATED_SPEED = 12.0            # Rated wind speed in m/s (turbine reaches rated power)
WIND_CUT_OUT_SPEED = 25.0          # Cut-out wind speed in m/s (feathered for safety)
WIND_TURBINE_HUB_HEIGHT = 80.0     # Hub height in meters

# ==============================================================================
# Data Preprocessing & Partitioning
# ==============================================================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Missing value handling threshold
MAX_CONSECUTIVE_MISSING = 8  # 2 hours at 15-min intervals

# ==============================================================================
# Signal Decomposition Parameters
# ==============================================================================
# 96 steps of 15 min = 24 hours diurnal cycle
DECOMPOSITION_MA_WINDOW = 96
MULTIRESOLUTION_LEVELS = 3

# ==============================================================================
# Adaptive Filtering Default Hyperparameters
# ==============================================================================
DEFAULT_HORIZON = 1          # Number of 15-minute steps ahead (1 step = 15 min, 2 = 30 min)
DEFAULT_FILTER_ORDER = 8     # Number of tapped delay line coefficients (lagged inputs)

# LMS Parameters
LMS_DEFAULT_MU = 0.015       # Adaptation step-size

# NLMS Parameters
NLMS_DEFAULT_MU = 0.50       # Normalized adaptation step-size
NLMS_DEFAULT_EPSILON = 1e-4  # Regularization constant to avoid divide-by-zero

# RLS Parameters
RLS_DEFAULT_LAMBDA = 0.992   # Exponential forgetting factor (0 < lambda <= 1)
RLS_DEFAULT_DELTA = 100.0    # Initial inverse covariance scaling (P(0) = delta * I)

# Kalman Predictor Parameters
KALMAN_STATE_DIM = 4         # Dimensionality of state vector [y(k), y(k-1), trend, velocity]
KALMAN_Q_VAR = 1e-3          # Process noise covariance scale
KALMAN_R_VAR = 1e-2          # Measurement noise variance scale

# Baseline Parameters
RIDGE_ALPHA = 1.0            # L2 penalty parameter for Ridge Regression
