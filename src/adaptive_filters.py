"""Adaptive Signal Processing Filters for Renewable Energy Forecasting.

Provides numerically stable, vectorized implementations of:
1. LMS (Least Mean Squares - Widrow-Hoff stochastic gradient descent)
2. NLMS (Normalized Least Mean Squares with energy normalization)
3. RLS (Recursive Least Squares with exponential forgetting factor)

All filters support:
- Offline pre-adaptation on training data
- Online real-time adaptive tracking on validation and test streams
- Hyperparameter tuning on validation partitions
- Instantaneous squared error logging for learning curve visualization
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is available in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

import config


class BaseAdaptiveFilter:
    """Abstract base class for linear adaptive transversal filters."""

    def __init__(self, filter_order: int, name: str = "AdaptiveFilter"):
        self.order = filter_order
        self.name = name
        self.weights: np.ndarray = np.zeros(self.order, dtype=np.float64)
        self.learning_curve: List[float] = []

    def reset(self) -> None:
        """Reset filter weights and adaptation history."""
        self.weights = np.zeros(self.order, dtype=np.float64)
        self.learning_curve = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BaseAdaptiveFilter":
        """Pre-adapt filter weights over training sequence."""
        raise NotImplementedError

    def predict(
        self,
        X: np.ndarray,
        online_adapt: bool = True,
        y_true: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate one-step or multi-step forecasts with optional real-time adaptation.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (N, filter_order).
        online_adapt : bool
            If True, updates filter weights sequentially as observations arrive.
        y_true : Optional[np.ndarray]
            True targets used for real-time error calculation and weight update.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (y_pred, error_array)
        """
        raise NotImplementedError


class LMSFilter(BaseAdaptiveFilter):
    """Least Mean Squares (LMS) Adaptive Filter.

    Mathematical Formulation:
    -------------------------
    Prediction:
        y_hat(k) = w^T(k) * x(k)
    Instantaneous error:
        e(k) = d(k) - y_hat(k)
    Weight update rule:
        w(k+1) = w(k) + mu * e(k) * x(k)
    """

    def __init__(self, filter_order: int, mu: float = config.LMS_DEFAULT_MU):
        super().__init__(filter_order, name="LMS")
        self.mu = mu

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LMSFilter":
        """Train LMS filter on historical data."""
        self.reset()
        n_samples = len(X_train)
        self.learning_curve = []

        for k in range(n_samples):
            x_k = X_train[k]
            d_k = y_train[k]
            y_hat = np.dot(self.weights, x_k)
            e_k = d_k - y_hat
            # Weight update
            self.weights += self.mu * e_k * x_k
            self.learning_curve.append(float(e_k**2))

        return self

    def predict(
        self,
        X: np.ndarray,
        online_adapt: bool = True,
        y_true: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forecast sequence with optional real-time online adaptation."""
        n_samples = len(X)
        y_pred = np.zeros(n_samples, dtype=np.float64)
        errors = np.zeros(n_samples, dtype=np.float64)

        w = self.weights.copy()

        for k in range(n_samples):
            x_k = X[k]
            y_hat = np.dot(w, x_k)
            y_pred[k] = y_hat

            if y_true is not None:
                e_k = y_true[k] - y_hat
                errors[k] = e_k
                if online_adapt:
                    w += self.mu * e_k * x_k

        if online_adapt and y_true is not None:
            self.weights = w  # Retain updated state

        return y_pred, errors


class NLMSFilter(BaseAdaptiveFilter):
    """Normalized Least Mean Squares (NLMS) Adaptive Filter.

    Normalizes step size by input power to maintain stable convergence
    despite wide dynamics in renewable power and weather features.

    Mathematical Formulation:
    -------------------------
    Prediction:
        y_hat(k) = w^T(k) * x(k)
    Instantaneous error:
        e(k) = d(k) - y_hat(k)
    Normalized weight update rule:
        w(k+1) = w(k) + [mu / (epsilon + ||x(k)||^2)] * e(k) * x(k)
    """

    def __init__(
        self,
        filter_order: int,
        mu: float = config.NLMS_DEFAULT_MU,
        epsilon: float = config.NLMS_DEFAULT_EPSILON,
    ):
        super().__init__(filter_order, name="NLMS")
        self.mu = mu
        self.epsilon = epsilon

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "NLMSFilter":
        """Pre-adapt NLMS filter coefficients on training sequence."""
        self.reset()
        n_samples = len(X_train)
        self.learning_curve = []

        for k in range(n_samples):
            x_k = X_train[k]
            d_k = y_train[k]
            y_hat = np.dot(self.weights, x_k)
            e_k = d_k - y_hat
            power_x = np.dot(x_k, x_k)
            step_norm = self.mu / (self.epsilon + power_x)
            self.weights += step_norm * e_k * x_k
            self.learning_curve.append(float(e_k**2))

        return self

    def predict(
        self,
        X: np.ndarray,
        online_adapt: bool = True,
        y_true: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forecast sequence with energy-normalized adaptation."""
        n_samples = len(X)
        y_pred = np.zeros(n_samples, dtype=np.float64)
        errors = np.zeros(n_samples, dtype=np.float64)

        w = self.weights.copy()

        for k in range(n_samples):
            x_k = X[k]
            y_hat = np.dot(w, x_k)
            y_pred[k] = y_hat

            if y_true is not None:
                e_k = y_true[k] - y_hat
                errors[k] = e_k
                if online_adapt:
                    power_x = np.dot(x_k, x_k)
                    step_norm = self.mu / (self.epsilon + power_x)
                    w += step_norm * e_k * x_k

        if online_adapt and y_true is not None:
            self.weights = w

        return y_pred, errors


class RLSFilter(BaseAdaptiveFilter):
    """Recursive Least Squares (RLS) Adaptive Filter.

    Minimizes exponentially weighted cumulative squared error with rapid convergence.

    Mathematical Formulation:
    -------------------------
    Gain Vector:
        g(k) = [P(k-1) * x(k)] / [lambda + x^T(k) * P(k-1) * x(k)]
    A Priori Error:
        xi(k) = d(k) - w^T(k-1) * x(k)
    Weight Update:
        w(k) = w(k-1) + g(k) * xi(k)
    Inverse Covariance Update:
        P(k) = (1 / lambda) * [P(k-1) - g(k) * x^T(k) * P(k-1)]
    """

    def __init__(
        self,
        filter_order: int,
        lam: float = config.RLS_DEFAULT_LAMBDA,
        delta: float = config.RLS_DEFAULT_DELTA,
    ):
        super().__init__(filter_order, name="RLS")
        self.lam = lam
        self.delta = delta
        self.P: np.ndarray = np.eye(self.order, dtype=np.float64) * self.delta

    def reset(self) -> None:
        """Reset weights, learning curve, and inverse covariance matrix."""
        super().reset()
        self.P = np.eye(self.order, dtype=np.float64) * self.delta

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RLSFilter":
        """Train RLS filter using recursive Riccati matrix updates."""
        self.reset()
        n_samples = len(X_train)
        self.learning_curve = []

        for k in range(n_samples):
            x_k = X_train[k]
            d_k = y_train[k]

            # P * x
            Px = np.dot(self.P, x_k)
            # Denominator: lambda + x^T * P * x
            denom = self.lam + np.dot(x_k, Px)
            # Gain vector: g(k)
            g_k = Px / denom

            # A priori error
            xi_k = d_k - np.dot(self.weights, x_k)

            # Weight update
            self.weights += g_k * xi_k

            # P matrix update: (P - g * x^T * P) / lambda
            self.P = (self.P - np.outer(g_k, Px)) / self.lam
            # Enforce symmetry to prevent numerical drift
            self.P = 0.5 * (self.P + self.P.T)

            self.learning_curve.append(float(xi_k**2))

        return self

    def predict(
        self,
        X: np.ndarray,
        online_adapt: bool = True,
        y_true: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forecast sequence with online recursive covariance updates."""
        n_samples = len(X)
        y_pred = np.zeros(n_samples, dtype=np.float64)
        errors = np.zeros(n_samples, dtype=np.float64)

        w = self.weights.copy()
        P = self.P.copy()

        for k in range(n_samples):
            x_k = X[k]
            y_hat = np.dot(w, x_k)
            y_pred[k] = y_hat

            if y_true is not None:
                xi_k = y_true[k] - y_hat
                errors[k] = xi_k

                if online_adapt:
                    Px = np.dot(P, x_k)
                    denom = self.lam + np.dot(x_k, Px)
                    g_k = Px / denom
                    w += g_k * xi_k
                    P = (P - np.outer(g_k, Px)) / self.lam
                    P = 0.5 * (P + P.T)

        if online_adapt and y_true is not None:
            self.weights = w
            self.P = P

        return y_pred, errors


def tune_filter_on_validation(
    filter_cls: Any,
    filter_order: int,
    param_name: str,
    param_values: List[float],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[Any, float, float]:
    """Tune a filter parameter on the validation partition to pick the best model.

    Parameters
    ----------
    filter_cls : class
        LMSFilter, NLMSFilter, or RLSFilter.
    filter_order : int
        Filter dimension.
    param_name : str
        'mu' or 'lam'.
    param_values : List[float]
        Candidate hyperparameter values.
    X_train, y_train : np.ndarray
        Training partition.
    X_val, y_val : np.ndarray
        Validation partition.

    Returns
    -------
    Tuple[Any, float, float]
        (best_fitted_filter, best_param_value, lowest_val_rmse)
    """
    best_filter = None
    best_param = param_values[0]
    lowest_rmse = float("inf")

    for val in param_values:
        kwargs = {param_name: val}
        model = filter_cls(filter_order=filter_order, **kwargs)
        model.fit(X_train, y_train)

        # Evaluate on validation with online adaptation
        preds, _ = model.predict(X_val, online_adapt=True, y_true=y_val)
        val_rmse = float(np.sqrt(np.mean((y_val - preds) ** 2)))

        if val_rmse < lowest_rmse:
            lowest_rmse = val_rmse
            best_param = val
            best_filter = model

    print(
        f"[Tuning] {best_filter.name}: Best {param_name}={best_param:.4f} "
        f"(Val RMSE={lowest_rmse:.4f})"
    )
    return best_filter, best_param, lowest_rmse
