"""Transparent performance and tail-risk statistics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) < 1e-15:
        return float("nan")
    return numerator / denominator


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute drawdowns from a simple-return series."""

    clean = returns.fillna(0.0).clip(lower=-0.999999)
    equity = (1.0 + clean).cumprod()
    running_peak = equity.cummax()
    return (equity / running_peak - 1.0).rename("drawdown")


def performance_metrics(
    returns: pd.Series,
    *,
    annualization: int = 252,
    turnover: pd.Series | None = None,
    positions: pd.Series | None = None,
) -> dict[str, float]:
    """Return interview-friendly performance, risk, and trading diagnostics."""

    clean = returns.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if clean.empty:
        return {"observations": 0.0}
    if annualization <= 0:
        raise ValueError("annualization must be strictly positive")

    observations = len(clean)
    clean_array = clean.to_numpy(dtype=float)
    total_return = float(np.prod(1.0 + np.clip(clean_array, -0.999999, None)) - 1.0)
    years = observations / annualization
    annual_return = (
        float((1.0 + total_return) ** (1.0 / years) - 1.0)
        if years > 0 and total_return > -1.0
        else float("nan")
    )
    daily_mean = float(np.mean(clean_array))
    daily_volatility = float(np.std(clean_array, ddof=1))
    annual_volatility = daily_volatility * math.sqrt(annualization)
    sharpe = _safe_ratio(daily_mean * math.sqrt(annualization), daily_volatility)

    downside = np.minimum(clean_array, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
    sortino = _safe_ratio(daily_mean * math.sqrt(annualization), downside_deviation)

    drawdown = drawdown_series(clean)
    maximum_drawdown = float(drawdown.min())
    calmar = _safe_ratio(annual_return, abs(maximum_drawdown))
    quantile = float(np.quantile(clean_array, 0.05))
    tail = clean[clean <= quantile]

    gains = float(np.sum(clean_array[clean_array > 0.0]))
    losses = float(-np.sum(clean_array[clean_array < 0.0]))
    active = clean[clean != 0.0]
    metrics: dict[str, float] = {
        "observations": float(observations),
        "total_return": total_return,
        "annualized_return": annual_return,
        "annualized_volatility": annual_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": maximum_drawdown,
        "calmar": calmar,
        "historical_var_95": -quantile,
        "expected_shortfall_95": (
            float(-np.mean(tail.to_numpy(dtype=float))) if not tail.empty else float("nan")
        ),
        "skew": float(skew(clean_array, bias=False)),
        "excess_kurtosis": float(kurtosis(clean_array, fisher=True, bias=False)),
        "hit_rate": float((active > 0.0).mean()) if not active.empty else float("nan"),
        "profit_factor": _safe_ratio(gains, losses),
    }

    if turnover is not None:
        aligned_turnover = turnover.reindex(clean.index).fillna(0.0)
        metrics["annualized_turnover"] = float(aligned_turnover.mean() * annualization)
        metrics["total_turnover"] = float(aligned_turnover.sum())

    if positions is not None:
        aligned_positions = positions.reindex(clean.index).fillna(0.0)
        entries = aligned_positions.ne(0.0) & aligned_positions.shift(1, fill_value=0.0).eq(0.0)
        metrics["trades"] = float(entries.sum())
        metrics["exposure"] = float(aligned_positions.ne(0.0).mean())

    return metrics
