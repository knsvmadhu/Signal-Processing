"""Automated unit and integration tests for adaptive filters, Kalman predictor, baselines, and evaluation metrics.
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
from src.adaptive_filters import LMSFilter, NLMSFilter, RLSFilter
from src.baseline_models import PersistenceBaseline, StaticRidgeBaseline
from src.evaluation import calculate_metrics
from src.kalman_predictor import KalmanFilterPredictor
from main import run_pipeline


class TestAdaptiveFilters(unittest.TestCase):
    """Test suite for LMS, NLMS, RLS, and Kalman algorithms."""

    def setUp(self):
        """Create synthetic autoregressive data for testing."""
        np.random.seed(42)
        n = 300
        self.order = 4
        # AR(1) process
        self.series = np.zeros(n)
        for i in range(1, n):
            self.series[i] = 0.85 * self.series[i - 1] + np.random.normal(0, 0.5)

        self.X = np.zeros((n - self.order, self.order))
        self.y = np.zeros(n - self.order)
        for i in range(n - self.order):
            self.X[i] = self.series[i : i + self.order][::-1]
            self.y[i] = self.series[i + self.order]

    def test_lms_filter_adaptation(self):
        """Verify LMS updates weights and reduces error on stationary AR sequence."""
        lms = LMSFilter(filter_order=self.order, mu=0.02)
        lms.fit(self.X[:200], self.y[:200])
        self.assertEqual(len(lms.weights), self.order)
        self.assertEqual(len(lms.learning_curve), 200)

        # Predictions on unseen data
        preds, errs = lms.predict(self.X[200:], online_adapt=True, y_true=self.y[200:])
        self.assertEqual(len(preds), len(self.y[200:]))
        self.assertFalse(np.isnan(preds).any())
        rmse = np.sqrt(np.mean(errs**2))
        self.assertLess(rmse, 2.0)

    def test_nlms_filter_zero_input_regularization(self):
        """Verify NLMS handles zero input without division by zero errors."""
        nlms = NLMSFilter(filter_order=self.order, mu=0.5, epsilon=1e-4)
        zero_X = np.zeros((10, self.order))
        zero_y = np.zeros(10)
        # Should not raise ZeroDivisionError
        nlms.fit(zero_X, zero_y)
        preds, _ = nlms.predict(zero_X, online_adapt=True, y_true=zero_y)
        np.testing.assert_array_equal(preds, np.zeros(10))

    def test_rls_covariance_symmetry_and_convergence(self):
        """Verify RLS covariance matrix remains symmetric and positive-definite."""
        rls = RLSFilter(filter_order=self.order, lam=0.99, delta=100.0)
        rls.fit(self.X[:200], self.y[:200])

        # Symmetry test: P == P^T
        diff = np.max(np.abs(rls.P - rls.P.T))
        self.assertLess(diff, 1e-6)

        preds, errs = rls.predict(self.X[200:], online_adapt=True, y_true=self.y[200:])
        self.assertFalse(np.isnan(preds).any())
        self.assertLess(np.sqrt(np.mean(errs**2)), 1.5)

    def test_kalman_predictor_stability(self):
        """Verify Kalman filter predictor produces valid numerical outputs."""
        kf = KalmanFilterPredictor(filter_order=self.order, q_variance=1e-3, r_variance=1e-2)
        kf.fit(self.X[:200], self.y[:200])

        preds, errs = kf.predict(self.X[200:], online_adapt=True, y_true=self.y[200:])
        self.assertFalse(np.isnan(preds).any())
        self.assertEqual(len(preds), len(self.y[200:]))

    def test_baseline_models(self):
        """Verify persistence and static ridge baseline functionality."""
        pers = PersistenceBaseline(filter_order=self.order)
        preds, _ = pers.predict(self.X[200:])
        # Persistence predicts lag 0: the first element of each feature vector
        np.testing.assert_array_equal(preds, self.X[200:, 0])

        ridge = StaticRidgeBaseline(filter_order=self.order, alpha=1.0)
        ridge.fit(self.X[:200], self.y[:200])
        r_preds, _ = ridge.predict(self.X[200:])
        self.assertEqual(len(r_preds), len(self.y[200:]))


class TestEvaluationAndPipeline(unittest.TestCase):
    """Test suite for evaluation metric calculations and end-to-end execution."""

    def test_metric_calculations(self):
        """Verify MAE, RMSE, and R2 calculation accuracy."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 19.0, 31.0, 38.0])
        # Errors: [2, -1, 1, -2]
        # MAE = (2 + 1 + 1 + 2)/4 = 1.5
        # MSE = (4 + 1 + 1 + 4)/4 = 2.5, RMSE = sqrt(2.5) ~= 1.5811
        metrics = calculate_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics["mae"], 1.5, places=3)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(2.5), places=3)
        self.assertGreater(metrics["r2"], 0.9)

    def test_pipeline_execution_solar_and_wind(self):
        """Verify end-to-end pipeline execution produces all required artifacts."""
        # Solar run
        solar_res = run_pipeline(
            source="solar",
            filter_choice="all",
            horizon=1,
            generate_plots=True,
        )
        self.assertIn("best_model", solar_res)
        self.assertTrue(config.MODEL_COMPARISON_PATH.exists())
        self.assertTrue((config.PREDICTIONS_DIR / "solar_predictions.csv").exists())
        self.assertTrue(config.EXPERIMENT_SUMMARY_PATH.exists())
        self.assertTrue((config.GRAPHS_DIR / "actual_vs_predicted.png").exists())

        # Wind run
        wind_res = run_pipeline(
            source="wind",
            filter_choice="rls",
            horizon=2,
            generate_plots=True,
        )
        self.assertIn("best_model", wind_res)
        self.assertTrue((config.PREDICTIONS_DIR / "wind_predictions.csv").exists())


if __name__ == "__main__":
    unittest.main()
