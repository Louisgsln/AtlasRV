"""Cointegration, mean-reversion, and stability diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from atlas_rv.config import PairConfig


@dataclass(frozen=True)
class PairDiagnostics:
    pair: str
    observations: int
    static_beta: float
    coint_statistic: float
    coint_pvalue: float
    coint_qvalue: float
    adf_statistic: float
    adf_pvalue: float
    half_life_days: float
    hurst_exponent: float
    return_correlation: float
    first_half_beta: float
    second_half_beta: float
    beta_instability: float
    passes_research_gate: bool
    gate_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _ols_beta(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coefficients[0]), float(coefficients[1])


def _half_life(spread: pd.Series) -> float:
    lagged = spread.shift(1)
    change = spread.diff()
    aligned = pd.concat([change.rename("change"), lagged.rename("lagged")], axis=1).dropna()
    if len(aligned) < 20:
        return float("nan")
    _, speed = _ols_beta(
        aligned["change"].to_numpy(dtype=float),
        aligned["lagged"].to_numpy(dtype=float),
    )
    if speed >= 0.0:
        return float("inf")
    return float(-np.log(2.0) / speed)


def _hurst_exponent(values: pd.Series) -> float:
    array = values.dropna().to_numpy(dtype=float)
    maximum_lag = min(100, len(array) // 4)
    if maximum_lag < 10:
        return float("nan")
    lags = np.arange(2, maximum_lag)
    tau = np.array([np.std(array[lag:] - array[:-lag], ddof=1) for lag in lags])
    valid = np.isfinite(tau) & (tau > 0.0)
    if valid.sum() < 5:
        return float("nan")
    slope = np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)[0]
    return float(slope)


def _gate_reasons(
    *,
    coint_pvalue: float,
    adf_pvalue: float,
    half_life: float,
    hurst: float,
    beta_instability: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if coint_pvalue >= 0.05:
        reasons.append("cointegration")
    if adf_pvalue >= 0.05:
        reasons.append("stationarity")
    if not np.isfinite(half_life) or not 2.0 <= half_life <= 126.0:
        reasons.append("half_life")
    if not np.isfinite(hurst) or hurst >= 0.5:
        reasons.append("hurst")
    if beta_instability >= 0.5:
        reasons.append("beta_instability")
    return tuple(reasons)


def diagnose_pair(prices: pd.DataFrame, pair: PairConfig) -> PairDiagnostics:
    """Evaluate whether a pair merits backtesting before parameter tuning."""

    aligned = prices[[pair.y, pair.x]].dropna().astype(float).sort_index()
    if len(aligned) < 100:
        raise ValueError("At least 100 aligned observations are required")
    if (aligned <= 0.0).any(axis=None):
        raise ValueError("Diagnostics require positive price levels")

    log_y = np.log(aligned[pair.y])
    log_x = np.log(aligned[pair.x])
    intercept, beta = _ols_beta(log_y.to_numpy(), log_x.to_numpy())
    spread = log_y - intercept - beta * log_x

    coint_statistic, coint_pvalue, _ = coint(log_y, log_x, trend="c", autolag="aic")
    adf_statistic, adf_pvalue, *_ = adfuller(spread, regression="c", autolag="aic")
    half_life = _half_life(spread)
    hurst = _hurst_exponent(spread)
    returns = aligned.pct_change(fill_method=None).dropna()
    return_correlation = float(returns[pair.y].corr(returns[pair.x]))

    midpoint = len(aligned) // 2
    _, first_beta = _ols_beta(log_y.iloc[:midpoint].to_numpy(), log_x.iloc[:midpoint].to_numpy())
    _, second_beta = _ols_beta(log_y.iloc[midpoint:].to_numpy(), log_x.iloc[midpoint:].to_numpy())
    beta_instability = float(abs(second_beta - first_beta) / max(abs(beta), 1e-8))
    reasons = _gate_reasons(
        coint_pvalue=float(coint_pvalue),
        adf_pvalue=float(adf_pvalue),
        half_life=half_life,
        hurst=hurst,
        beta_instability=beta_instability,
    )

    return PairDiagnostics(
        pair=pair.name,
        observations=len(aligned),
        static_beta=beta,
        coint_statistic=float(coint_statistic),
        coint_pvalue=float(coint_pvalue),
        coint_qvalue=float(coint_pvalue),
        adf_statistic=float(adf_statistic),
        adf_pvalue=float(adf_pvalue),
        half_life_days=half_life,
        hurst_exponent=hurst,
        return_correlation=return_correlation,
        first_half_beta=first_beta,
        second_half_beta=second_beta,
        beta_instability=beta_instability,
        passes_research_gate=not reasons,
        gate_reasons=reasons,
    )
