# Adaptive Signal Processing Framework for Renewable Energy Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Signal Processing](https://img.shields.io/badge/Framework-Adaptive%20Filtering-brightgreen.svg)]()

An end-to-end Python framework for multi-step ahead solar photovoltaic and wind turbine power forecasting using classical and modern adaptive signal processing techniques: **Least Mean Squares (LMS)**, **Normalized Least Mean Squares (NLMS)**, **Recursive Least Squares (RLS)**, and the **State-Space Kalman Filter Predictor**, benchmarked against **Persistence** and **Static Ridge Autoregression**.

---

## 1. Executive Summary & Problem Statement

Renewable energy generation from solar and wind assets exhibits extreme non-stationarity, intermittency, and meteorological turbulence. Traditional batch machine learning models assume static relationships between features and power output, requiring periodic retraining and failing to adapt to sudden micro-climatic shifts, cloud transients, and wind gusts.

**Adaptive Signal Processing** reformulates forecasting as a recursive online estimation problem. As new sensor observations arrive, adaptive filters continuously track dynamic environmental coefficients in real time without retraining from scratch. This framework provides an end-to-end pipeline:
- **Causal Data Generation & Ingestion**: 15-minute resolution datasets (>= 5,000 samples) with physically realistic photovoltaic and aerodynamic models.
- **Signal Decomposition**: Moving-average trend extraction, 24-hour diurnal profiling, and stationary multi-resolution wavelet analysis.
- **Adaptive Estimation**: LMS, NLMS, RLS, and Kalman Filter algorithms formulated for tapped-delay line forecasting.
- **Rigorous Evaluation**: Zero-leakage chronological partitioning (70% Train, 15% Validation, 15% Test), unbiased validation-based model selection, and reporting across MAE, RMSE, R², nRMSE, and MAPE.
- **Automated Visualization**: Publication-quality plots of time-series, actual vs. predicted power, learning curves, residuals, and decomposition.

---

## 2. Project Directory Structure

```text
SP Project/
│
├── main.py                     # Main CLI pipeline orchestrator
├── config.py                   # Central paths, physical constants & filter hyperparameters
├── requirements.txt            # Minimal required dependencies
├── README.md                   # Project overview, quickstart & usage guide
│
├── data/
│   ├── raw/                    # Raw time-series datasets (>= 5,000 records each)
│   │   ├── solar_energy.csv
│   │   ├── wind_energy.csv
│   │   └── combined_renewable_energy.csv
│   │
│   └── processed/              # Cleaned, validated, and bounds-enforced datasets
│       ├── solar_processed.csv
│       └── wind_processed.csv
│
├── src/
│   ├── __init__.py             # Package initializer
│   ├── data_generator.py       # Physically grounded PV & wind aerodynamic simulator
│   ├── data_loader.py          # Data ingestion, path resolution & column validator
│   ├── data_preprocessing.py   # Missing value cleaning, chronological splitting & scalers
│   ├── decomposition.py        # Moving-average trend, diurnal cycle & wavelet DWT
│   ├── adaptive_filters.py     # LMS, NLMS, and RLS transversal filter algorithms
│   ├── kalman_predictor.py     # State-space Kalman filter with Joseph form stability
│   ├── baseline_models.py      # Persistence and Static Ridge regression benchmarks
│   ├── evaluation.py           # MAE, RMSE, R², nRMSE, MAPE & report generation
│   └── visualization.py        # Publication-grade plotting suite (Agg backend)
│
├── outputs/
│   ├── graphs/                 # Visualizations (PNG 180 DPI)
│   ├── predictions/            # Test partition predictions (CSV)
│   ├── reports/                # Decompositions and summary reports (TXT / CSV)
│   └── metrics/                # Benchmark metrics comparison table (CSV)
│
├── tests/
│   ├── test_data.py            # Unit tests for generator, loader, cleaning & decomposition
│   └── test_filters.py         # Unit tests for filters, Kalman predictor, metrics & pipeline
│
└── docs/
    └── project_documentation.md # In-depth mathematical theory, derivations & analysis
```

---

## 3. Dataset Specifications & Provenance

All datasets are sampled at **15-minute intervals** for 6,000 consecutive records (~62.5 continuous days starting `2026-01-01 00:00:00`):

| Dataset | File Path | Columns | Target Column | Physical Capacity |
| :--- | :--- | :--- | :--- | :--- |
| **Solar Energy** | `data/raw/solar_energy.csv` | `timestamp`, `solar_irradiance` ($W/m^2$), `temperature` ($^\circ C$), `humidity` ($\%$), `cloud_cover` ($\%$), `power_kw` | `power_kw` | 100.0 kW array ($550 m^2$, 19% eff) |
| **Wind Energy** | `data/raw/wind_energy.csv` | `timestamp`, `wind_speed` ($m/s$), `wind_direction` ($^\circ$), `temperature` ($^\circ C$), `pressure` ($hPa$), `power_kw` | `power_kw` | 500.0 kW turbine ($v_{ci}=3, v_r=12, v_{co}=25 m/s$) |
| **Combined System**| `data/raw/combined_renewable_energy.csv` | 12 synchronized meteorological & generation features | `total_power_kw` | 600.0 kW hybrid microgrid |

> **Provenance Note**: The datasets are synthetic and generated using first-principles physical equations: astronomical solar geometry with cloud attenuation derating, and Weibull-distributed autoregressive wind speeds coupled to an aerodynamic piecewise cubic turbine power curve. Controlled anomalies (0.3% missing entries) verify data cleaning robustness.

---

## 4. Signal Processing & Adaptive Filtering Algorithms

### A. Tapped Delay Line Formulation
For a forecast horizon $h$ steps ahead (where $h=1 \Rightarrow 15\text{ min}$, $h=2 \Rightarrow 30\text{ min}$):
At forecast origin $t$:
$$\mathbf{x}(t) = [y(t), y(t-1), \dots, y(t - p + 1)]^T \in \mathbb{R}^p$$
$$\text{Target: } d(t) = y(t + h)$$
Strict chronological causality guarantees zero future information leakage.

### B. Filter Mathematical Summary
1. **LMS (Least Mean Squares)**:
   $$\hat{y}(k) = \mathbf{w}^T(k) \mathbf{x}(k), \quad e(k) = d(k) - \hat{y}(k)$$
   $$\mathbf{w}(k+1) = \mathbf{w}(k) + \mu \, e(k) \, \mathbf{x}(k)$$
2. **NLMS (Normalized LMS)**:
   $$\mathbf{w}(k+1) = \mathbf{w}(k) + \frac{\mu}{\epsilon + \|\mathbf{x}(k)\|^2} \, e(k) \, \mathbf{x}(k)$$
3. **RLS (Recursive Least Squares)**:
   $$\mathbf{g}(k) = \frac{\mathbf{P}(k-1) \mathbf{x}(k)}{\lambda + \mathbf{x}^T(k) \mathbf{P}(k-1) \mathbf{x}(k)}, \quad \xi(k) = d(k) - \mathbf{w}^T(k-1) \mathbf{x}(k)$$
   $$\mathbf{w}(k) = \mathbf{w}(k-1) + \mathbf{g}(k) \xi(k), \quad \mathbf{P}(k) = \frac{1}{\lambda} \left[ \mathbf{P}(k-1) - \mathbf{g}(k) \mathbf{x}^T(k) \mathbf{P}(k-1) \right]$$
4. **State-Space Kalman Filter Predictor**:
   Time update (predict) and measurement update (correct) with Joseph-stabilized error covariance matrix:
   $$\mathbf{P}(k|k) = (\mathbf{I} - \mathbf{K}_k \mathbf{x}_k^T) \mathbf{P}(k|k-1) (\mathbf{I} - \mathbf{K}_k \mathbf{x}_k^T)^T + \mathbf{K}_k R \mathbf{K}_k^T$$
5. **Persistence Benchmark**: $\hat{y}(t+h) = y(t)$
6. **Static Ridge Baseline**: Batch closed-form L2-regularized AR model without online adaptation.

---

## 5. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Windows PowerShell / Linux / macOS terminal

### Windows PowerShell Quickstart
```powershell
# 1. Clone or navigate to the project directory
cd "c:\Users\HP\Documents\ALL Projects\SP Project"

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 6. CLI Execution Commands & Examples

### A. Run full solar benchmark across all filters (Horizon = 15 mins)
```powershell
python main.py --source solar --filter all --horizon 1
```

### B. Run wind turbine forecasting with RLS (Horizon = 30 mins)
```powershell
python main.py --source wind --filter rls --horizon 2
```

### C. Run solar forecasting with NLMS (Horizon = 15 mins)
```powershell
python main.py --source solar --filter nlms --horizon 1
```

### D. Run custom dataset with configurable target column
```powershell
python main.py --source data/raw/solar_energy.csv --target-col power_kw --order 8
```

### E. Run automated test suite
```powershell
python -m unittest discover tests
```

---

## 7. Generated Outputs & Artifacts

All runs automatically generate structured outputs:

- **Metrics Table**: [`outputs/metrics/model_comparison.csv`](file:///outputs/metrics/model_comparison.csv)
- **Predictions**: [`outputs/predictions/solar_predictions.csv`](file:///outputs/predictions/solar_predictions.csv), [`outputs/predictions/wind_predictions.csv`](file:///outputs/predictions/wind_predictions.csv)
- **Reports**: [`outputs/reports/experiment_summary.txt`](file:///outputs/reports/experiment_summary.txt), [`outputs/reports/decomposition_solar.csv`](file:///outputs/reports/decomposition_solar.csv)
- **Visualizations**:
  - `outputs/graphs/solar_power_timeseries.png`
  - `outputs/graphs/wind_power_timeseries.png`
  - `outputs/graphs/actual_vs_predicted.png`
  - `outputs/graphs/models_comparison.png`
  - `outputs/graphs/adaptive_learning_curves.png`
  - `outputs/graphs/residual_errors.png`
  - `outputs/graphs/metrics_comparison.png`
  - `outputs/graphs/signal_decomposition.png`
  - `outputs/graphs/dataset_splits.png`

---

## 8. License & Citation
Developed for faculty demonstration and research in renewable energy signal processing. Released under the MIT License.
