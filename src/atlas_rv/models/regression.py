"""Causal hedge-ratio estimators with a common result contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from atlas_rv.models.kalman import DynamicRegressionResult, KalmanDynamicRegression


def _causal_ols(
    y: pd.Series,
    x: pd.Series,
    *,
    lookback: int | None,
    minimum_observations: int,
) -> DynamicRegressionResult:
    aligned = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna().sort_index()
    if len(aligned) < 3:
        raise ValueError("At least three aligned observations are required")
    if minimum_observations < 3:
        raise ValueError("minimum_observations must be at least three")
    if lookback is not None and lookback < minimum_observations:
        raise ValueError("lookback cannot be shorter than minimum_observations")

    y_values = aligned["y"].to_numpy(dtype=float)
    x_values = aligned["x"].to_numpy(dtype=float)
    intercepts = np.zeros(len(aligned), dtype=float)
    betas = np.ones(len(aligned), dtype=float)
    innovations = np.empty(len(aligned), dtype=float)
    variances = np.ones(len(aligned), dtype=float)

    intercept = 0.0
    beta = 1.0
    variance = 1.0
    for position in range(len(aligned)):
        start = 0 if lookback is None else max(0, position - lookback)
        if position - start >= minimum_observations:
            train_x = x_values[start:position]
            train_y = y_values[start:position]
            design = np.column_stack([np.ones(len(train_x)), train_x])
            coefficients, *_ = np.linalg.lstsq(design, train_y, rcond=None)
            intercept = float(coefficients[0])
            beta = float(coefficients[1])
            residuals = train_y - design @ coefficients
            if len(residuals) > 2:
                variance = max(float(np.var(residuals, ddof=2)), 1e-12)

        intercepts[position] = intercept
        betas[position] = beta
        innovations[position] = y_values[position] - intercept - beta * x_values[position]
        variances[position] = variance

    index = aligned.index
    return DynamicRegressionResult(
        intercept=pd.Series(intercepts, index=index, name="intercept"),
        beta=pd.Series(betas, index=index, name="beta"),
        innovation=pd.Series(innovations, index=index, name="spread"),
        innovation_variance=pd.Series(
            variances, index=index, name="innovation_variance"
        ),
    )


@dataclass(frozen=True)
class ExpandingOLSRegression:
    """Expanding OLS whose estimate at t uses observations strictly before t."""

    minimum_observations: int = 20

    def fit(self, y: pd.Series, x: pd.Series) -> DynamicRegressionResult:
        return _causal_ols(
            y,
            x,
            lookback=None,
            minimum_observations=self.minimum_observations,
        )


@dataclass(frozen=True)
class RollingOLSRegression:
    """Rolling OLS whose estimate at t excludes observation t."""

    lookback: int = 126
    minimum_observations: int = 20

    def fit(self, y: pd.Series, x: pd.Series) -> DynamicRegressionResult:
        return _causal_ols(
            y,
            x,
            lookback=self.lookback,
            minimum_observations=self.minimum_observations,
        )


def fit_hedge_model(
    y: pd.Series,
    x: pd.Series,
    *,
    model: str,
    kalman_delta: float,
    observation_variance: float,
    rolling_ols_lookback: int,
) -> DynamicRegressionResult:
    """Fit one of the supported causal hedge-ratio models."""

    if model == "kalman":
        return KalmanDynamicRegression(
            delta=kalman_delta,
            observation_variance=observation_variance,
        ).fit(y, x)
    if model == "rolling_ols":
        return RollingOLSRegression(lookback=rolling_ols_lookback).fit(y, x)
    if model == "expanding_ols":
        return ExpandingOLSRegression().fit(y, x)
    raise ValueError(f"Unsupported hedge model: {model}")
