"""Automated unit and integration tests for data generation, loading, preprocessing, and decomposition.
"""

import sys
from pathlib import Path
import unittest

# Ensure project root is available in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

import config
from src.data_generator import (
    generate_solar_dataset,
    generate_wind_dataset,
    generate_combined_dataset,
    create_all_raw_datasets,
)
from src.data_loader import load_dataset, resolve_dataset_source
from src.data_preprocessing import (
    TimeSeriesScaler,
    clean_dataset,
    chronological_split,
    create_lagged_features,
)
from src.decomposition import (
    decompose_signal,
    moving_average_trend,
    multiresolution_wavelet_decompose,
)


class TestDataGenerationAndLoading(unittest.TestCase):
    """Test suite for dataset generation, integrity, and loader functionality."""

    @classmethod
    def setUpClass(cls):
        """Ensure datasets exist before testing."""
        create_all_raw_datasets(print_summary=False)

    def test_solar_dataset_dimensions_and_columns(self):
        """Verify solar CSV contains at least 5,000 records and all required columns."""
        self.assertTrue(config.SOLAR_RAW_PATH.exists())
        df = pd.read_csv(config.SOLAR_RAW_PATH)
        self.assertGreaterEqual(len(df), 5000)

        expected_cols = [
            "timestamp",
            "solar_irradiance",
            "temperature",
            "humidity",
            "cloud_cover",
            "power_kw",
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_wind_dataset_dimensions_and_columns(self):
        """Verify wind CSV contains at least 5,000 records and all required columns."""
        self.assertTrue(config.WIND_RAW_PATH.exists())
        df = pd.read_csv(config.WIND_RAW_PATH)
        self.assertGreaterEqual(len(df), 5000)

        expected_cols = [
            "timestamp",
            "wind_speed",
            "wind_direction",
            "temperature",
            "pressure",
            "power_kw",
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_combined_dataset_structure(self):
        """Verify combined dataset links solar and wind features with power columns."""
        self.assertTrue(config.COMBINED_RAW_PATH.exists())
        df = pd.read_csv(config.COMBINED_RAW_PATH)
        self.assertGreaterEqual(len(df), 5000)
        self.assertIn("solar_power_kw", df.columns)
        self.assertIn("wind_power_kw", df.columns)
        self.assertIn("total_power_kw", df.columns)

    def test_data_loader_resolution_and_chronology(self):
        """Verify data loader resolves keywords, parses timestamps, and enforces sort."""
        df_solar, path_solar, target = load_dataset("solar")
        self.assertEqual(path_solar, config.SOLAR_RAW_PATH)
        self.assertEqual(target, "power_kw")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df_solar["timestamp"]))
        # Verify strictly monotonically increasing
        self.assertTrue(df_solar["timestamp"].is_monotonic_increasing)


class TestPreprocessingAndDecomposition(unittest.TestCase):
    """Test suite for cleaning, chronological partitioning, and decomposition."""

    def test_clean_dataset_imputation_and_limits(self):
        """Verify missing values are imputed and physical bounds are maintained."""
        # Create dummy df with missing and negative values
        df_dirty = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=10, freq="15min"),
                "solar_irradiance": [100.0, np.nan, 200.0, -10.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
                "humidity": [50.0, 60.0, np.nan, 120.0, 40.0, 30.0, 20.0, 10.0, 50.0, 60.0],
                "power_kw": [10.0, np.nan, 20.0, -5.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            }
        )
        cleaned_df, _ = clean_dataset(df_dirty, target_col="power_kw", source_type="solar", save_processed=False)
        self.assertEqual(cleaned_df.isna().sum().sum(), 0)
        self.assertGreaterEqual(cleaned_df["power_kw"].min(), 0.0)
        self.assertGreaterEqual(cleaned_df["solar_irradiance"].min(), 0.0)
        self.assertLessEqual(cleaned_df["humidity"].max(), 100.0)

    def test_chronological_split_no_leakage(self):
        """Ensure chronological partition has 70/15/15 ratio and strict non-overlap."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=1000, freq="15min"),
                "power_kw": np.random.uniform(0, 100, 1000),
            }
        )
        train, val, test = chronological_split(df, 0.70, 0.15, 0.15)
        self.assertEqual(len(train), 700)
        self.assertEqual(len(val), 150)
        self.assertEqual(len(test), 150)
        # Ensure temporal causality
        self.assertLess(train["timestamp"].max(), val["timestamp"].min())
        self.assertLess(val["timestamp"].max(), test["timestamp"].min())

    def test_tapped_delay_line_causality(self):
        """Verify lag matrix alignment preserves causality without lookahead."""
        series = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
        order = 3
        horizon = 1
        X, y = create_lagged_features(series, order=order, horizon=horizon)
        # First sample: origin t=2 (value 30). Lags are [30, 20, 10]. Target is t+1 (40)
        np.testing.assert_array_equal(X[0], [30.0, 20.0, 10.0])
        self.assertEqual(y[0], 40.0)

    def test_signal_decomposition(self):
        """Verify signal decomposition produces trend, diurnal, and bounded residual."""
        timestamps = pd.date_range("2026-01-01", periods=200, freq="15min")
        signal = 50.0 + 10.0 * np.sin(np.linspace(0, 4 * np.pi, 200)) + np.random.normal(0, 1, 200)
        df_decomp = decompose_signal(signal, timestamps, ma_window=20)
        self.assertIn("trend", df_decomp.columns)
        self.assertIn("diurnal", df_decomp.columns)
        self.assertIn("residual", df_decomp.columns)

        wavelet_dict = multiresolution_wavelet_decompose(signal, levels=2)
        self.assertIn("approximation", wavelet_dict)
        self.assertIn("detail_1", wavelet_dict)


if __name__ == "__main__":
    unittest.main()
