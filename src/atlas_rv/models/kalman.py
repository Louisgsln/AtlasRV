"""Sequential dynamic hedge-ratio estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DynamicRegressionResult:
    """Filtered state and one-step-ahead pricing errors."""

    intercept: pd.Series
    beta: pd.Series
    innovation: pd.Series
    innovation_variance: pd.Series

    def to_frame(self) -> pd.DataFrame:
        return pd.concat(
            [self.intercept, self.beta, self.innovation, self.innovation_variance], axis=1
        )


class KalmanDynamicRegression:
    """Random-walk Kalman filter for ``y_t = alpha_t + beta_t x_t + eps_t``.

    The innovation is computed from the state available *before* observing
    ``y_t``. It is therefore a genuine one-step pricing error rather than an
    in-sample residual.
    """

    def __init__(
        self,
        delta: float = 1e-6,
        observation_variance: float = 1e-3,
        initial_beta: float = 1.0,
    ) -> None:
        if not 0 < delta < 1:
            raise ValueError("delta must lie strictly between zero and one")
        if observation_variance <= 0:
            raise ValueError("observation_variance must be strictly positive")
        self.delta = delta
        self.observation_variance = observation_variance
        self.initial_beta = initial_beta

    def fit(self, y: pd.Series, x: pd.Series) -> DynamicRegressionResult:
        """Filter aligned finite series and preserve their original index."""

        aligned = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
        if len(aligned) < 3:
            raise ValueError("At least three aligned observations are required")
        if not aligned.index.is_monotonic_increasing:
            aligned = aligned.sort_index()

        state = np.array([0.0, self.initial_beta], dtype=float)
        covariance = np.eye(2, dtype=float)
        process_variance = self.delta / (1.0 - self.delta)
        process_covariance = process_variance * np.eye(2, dtype=float)

        states = np.empty((len(aligned), 2), dtype=float)
        innovations = np.empty(len(aligned), dtype=float)
        innovation_variances = np.empty(len(aligned), dtype=float)

        for position, (x_value, y_value) in enumerate(
            zip(aligned["x"].to_numpy(), aligned["y"].to_numpy(), strict=True)
        ):
            predicted_state = state
            predicted_covariance = covariance + process_covariance
            observation = np.array([1.0, x_value], dtype=float)
            innovation = float(y_value - observation @ predicted_state)
            innovation_variance = float(
                observation @ predicted_covariance @ observation
                + self.observation_variance
            )
            gain = predicted_covariance @ observation / innovation_variance
            state = predicted_state + gain * innovation
            covariance = predicted_covariance - np.outer(gain, observation) @ predicted_covariance
            covariance = 0.5 * (covariance + covariance.T)

            states[position] = state
            innovations[position] = innovation
            innovation_variances[position] = innovation_variance

        index = aligned.index
        return DynamicRegressionResult(
            intercept=pd.Series(states[:, 0], index=index, name="intercept"),
            beta=pd.Series(states[:, 1], index=index, name="beta"),
            innovation=pd.Series(innovations, index=index, name="spread"),
            innovation_variance=pd.Series(
                innovation_variances, index=index, name="innovation_variance"
            ),
        )
