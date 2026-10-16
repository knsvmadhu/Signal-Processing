"""State-space Kalman Filter Predictor for Renewable Energy Forecasting.

Implements an optimal linear quadratic estimator (Kalman filter) formulated
both as an adaptive state-space parameter tracker and dynamic autoregressive predictor.

Mathematical Formulation:
-------------------------
State dynamics (Random walk / time-varying coefficients):
    w(k) = w(k-1) + q(k),    q(k) ~ N(0, Q)

Observation model:
    d(k) = x^T(k) * w(k) + v(k),   v(k) ~ N(0, R)

Kalman Recursions:
1. Time Update (Predict):
    w_hat(k|k-1) = w_hat(k-1|k-1)
    P(k|k-1)     = P(k-1|k-1) + Q

2. Measurement Update (Correct):
    Innovation:  e(k) = d(k) - x^T(k) * w_hat(k|k-1)
    Covariance:  S(k) = x^T(k) * P(k|k-1) * x(k) + R
    Kalman Gain: K(k) = [P(k|k-1) * x(k)] / S(k)
    State Update: w_hat(k|k) = w_hat(k|k-1) + K(k) * e(k)
    Covariance Update (Joseph form for numerical stability):
        P(k|k) = (I - K(k)*x^T(k)) * P(k|k-1)
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


class KalmanFilterPredictor:
    """Kalman Filter adaptive parameter predictor for renewable time-series."""

    def __init__(
        self,
        filter_order: int,
        q_variance: float = config.KALMAN_Q_VAR,
        r_variance: float = config.KALMAN_R_VAR,
        initial_p_variance: float = 1.0,
    ):
        """Initialize Kalman Filter Predictor.

        Parameters
        ----------
        filter_order : int
            Dimension of input feature / lag vector x(k).
        q_variance : float
            Process noise covariance scalar Q = q_variance * I.
        r_variance : float
            Measurement noise variance scalar R.
        initial_p_variance : float
            Initial error covariance diagonal scale P(0) = p_var * I.
        """
        self.order = filter_order
        self.q_var = q_variance
        self.r_var = r_variance
        self.p_init = initial_p_variance
        self.name = "Kalman"

        self.weights = np.zeros(self.order, dtype=np.float64)
        self.P = np.eye(self.order, dtype=np.float64) * self.p_init
        self.Q = np.eye(self.order, dtype=np.float64) * self.q_var
        self.learning_curve: list[float] = []

    def reset(self) -> None:
        """Reset state estimate and covariance matrix."""
        self.weights = np.zeros(self.order, dtype=np.float64)
        self.P = np.eye(self.order, dtype=np.float64) * self.p_init
        self.Q = np.eye(self.order, dtype=np.float64) * self.q_var
        self.learning_curve = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "KalmanFilterPredictor":
        """Pre-adapt state estimate over training partition."""
        self.reset()
        n_samples = len(X_train)
        self.learning_curve = []

        for k in range(n_samples):
            x_k = X_train[k]
            d_k = y_train[k]

            # 1. Time Update (Predict)
            w_prior = self.weights
            P_prior = self.P + self.Q

            # 2. Measurement Update (Correct)
            y_pred = np.dot(w_prior, x_k)
            innovation = d_k - y_pred

            Px = np.dot(P_prior, x_k)
            S_k = np.dot(x_k, Px) + self.r_var
            K_k = Px / S_k

            self.weights = w_prior + K_k * innovation
            # Joseph form: (I - K x^T) P (I - K x^T)^T + K R K^T for numerical stability
            I_minus_Kx = np.eye(self.order) - np.outer(K_k, x_k)
            self.P = (
                np.dot(np.dot(I_minus_Kx, P_prior), I_minus_Kx.T)
                + np.outer(K_k, K_k) * self.r_var
            )

            self.learning_curve.append(float(innovation**2))

        return self

    def predict(
        self,
        X: np.ndarray,
        online_adapt: bool = True,
        y_true: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate forecasts with online Kalman filtering updates.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (N, filter_order).
        online_adapt : bool
            If True, applies measurement update sequentially as data points arrive.
        y_true : Optional[np.ndarray]
            Observed targets used for recursive state correction.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (predictions, errors)
        """
        n_samples = len(X)
        y_pred = np.zeros(n_samples, dtype=np.float64)
        errors = np.zeros(n_samples, dtype=np.float64)

        w = self.weights.copy()
        P = self.P.copy()

        for k in range(n_samples):
            x_k = X[k]

            # Predict step
            w_prior = w
            P_prior = P + self.Q if online_adapt else P

            y_hat = np.dot(w_prior, x_k)
            y_pred[k] = y_hat

            if y_true is not None:
                innovation = y_true[k] - y_hat
                errors[k] = innovation

                if online_adapt:
                    Px = np.dot(P_prior, x_k)
                    S_k = np.dot(x_k, Px) + self.r_var
                    K_k = Px / S_k

                    w = w_prior + K_k * innovation
                    I_minus_Kx = np.eye(self.order) - np.outer(K_k, x_k)
                    P = (
                        np.dot(np.dot(I_minus_Kx, P_prior), I_minus_Kx.T)
                        + np.outer(K_k, K_k) * self.r_var
                    )

        if online_adapt and y_true is not None:
            self.weights = w
            self.P = P

        return y_pred, errors


if __name__ == "__main__":
    from src.data_generator import generate_solar_dataset
    from src.data_preprocessing import create_lagged_features, TimeSeriesScaler

    df = generate_solar_dataset(n_samples=500, add_test_anomalies=False)
    X, y = create_lagged_features(df["power_kw"].values, order=4, horizon=1)
    scaler = TimeSeriesScaler().fit(X[:300], y[:300])
    X_s = scaler.transform_x(X)
    y_s = scaler.transform_y(y)

    kf = KalmanFilterPredictor(filter_order=4)
    kf.fit(X_s[:300], y_s[:300])
    preds, errs = kf.predict(X_s[300:], online_adapt=True, y_true=y_s[300:])
    print(f"Kalman Test RMSE (scaled): {np.sqrt(np.mean(errs**2)):.4f}")
