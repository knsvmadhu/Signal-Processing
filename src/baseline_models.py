"""Baseline forecasting models for benchmark comparisons.

Implements:
1. Persistence Baseline:
   Standard industry benchmark assuming power at forecast origin persists:
   y_hat(t + h) = y(t).
2. Static Ridge Autoregressive Baseline:
   Closed-form L2-regularized linear model trained once on training data
   without real-time online adaptation (demonstrating the advantage of adaptive tracking).
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

# Ensure project root is available in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

import config


class PersistenceBaseline:
    """Persistence benchmark predictor: y_hat(t + h) = y(t)."""

    def __init__(self, filter_order: int = config.DEFAULT_FILTER_ORDER):
        self.order = filter_order
        self.name = "Persistence"
        self.weights = np.zeros(self.order)
        if self.order > 0:
            self.weights[0] = 1.0  # Selects lag 0: y(t)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "PersistenceBaseline":
        """No parameter training required for persistence."""
        return self

    def predict(
        self,
        X: np.ndarray,
        online_adapt: bool = False,
        y_true: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict current value as future value: y_hat = X[:, 0]."""
        # The first column of X is the most recent observation y(t)
        y_pred = X[:, 0].copy()
        errors = np.zeros_like(y_pred)
        if y_true is not None:
            errors = y_true - y_pred
        return y_pred, errors


class StaticRidgeBaseline:
    """Static Ridge (L2-regularized AR) predictor without online adaptation.

    Solves:
        w* = (X_train^T * X_train + alpha * I)^(-1) * X_train^T * y_train
    """

    def __init__(
        self,
        filter_order: int = config.DEFAULT_FILTER_ORDER,
        alpha: float = config.RIDGE_ALPHA,
    ):
        self.order = filter_order
        self.alpha = alpha
        self.name = "StaticRidge"
        self.weights = np.zeros(self.order, dtype=np.float64)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "StaticRidgeBaseline":
        """Compute closed-form regularized OLS weights."""
        d = X_train.shape[1]
        XtX = np.dot(X_train.T, X_train)
        reg_matrix = self.alpha * np.eye(d)
        Xty = np.dot(X_train.T, y_train)

        # Solve (XtX + alpha*I) * w = Xty
        self.weights = np.linalg.solve(XtX + reg_matrix, Xty)
        return self

    def predict(
        self,
        X: np.ndarray,
        online_adapt: bool = False,
        y_true: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate static batch forecasts (weights remain fixed)."""
        y_pred = np.dot(X, self.weights)
        errors = np.zeros_like(y_pred)
        if y_true is not None:
            errors = y_true - y_pred
        return y_pred, errors


if __name__ == "__main__":
    X_dummy = np.array([[1.0, 0.9], [2.0, 1.1], [3.0, 2.0]])
    y_dummy = np.array([1.2, 2.1, 3.2])
    pers = PersistenceBaseline(filter_order=2)
    p_preds, _ = pers.predict(X_dummy, y_true=y_dummy)
    ridge = StaticRidgeBaseline(filter_order=2, alpha=0.1).fit(X_dummy, y_dummy)
    r_preds, _ = ridge.predict(X_dummy, y_true=y_dummy)
    print("Persistence Predictions:", p_preds)
    print("Ridge Predictions:", r_preds)
