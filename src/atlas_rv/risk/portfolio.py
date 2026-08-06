"""Causal correlation-aware allocation across relative-value sleeves."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from atlas_rv.risk.metrics import drawdown_series, performance_metrics

if TYPE_CHECKING:
    from atlas_rv.backtest.engine import PairBacktestResult


@dataclass(frozen=True)
class PortfolioResult:
    frame: pd.DataFrame
    weights: pd.DataFrame
    class_allocations: pd.DataFrame
    metrics: dict[str, float]


def _bounded_weights(values: pd.Series, maximum: float) -> pd.Series:
    finite = values.replace([np.inf, -np.inf], np.nan).dropna().clip(lower=0.0)
    result = pd.Series(0.0, index=values.index, dtype=float)
    if finite.empty or float(finite.sum()) <= 0.0:
        return result
    if len(finite) * maximum < 1.0 - 1e-12:
        result.loc[finite.index] = maximum
        return result

    active = list(finite.index)
    remaining = 1.0
    while active:
        active_values = finite.loc[active]
        proposal = remaining * active_values / float(active_values.sum())
        capped = proposal[proposal > maximum]
        if capped.empty:
            result.loc[active] = proposal
            break
        for label in capped.index:
            result.loc[label] = maximum
            remaining -= maximum
            active.remove(label)
        if remaining <= 1e-12:
            break
    return result


def _diversification_scores(
    history: pd.DataFrame,
    columns: pd.Index,
    penalty: float,
) -> pd.Series:
    if penalty == 0.0 or history.shape[1] < 2:
        return pd.Series(1.0, index=columns, dtype=float)
    correlation = history.corr().abs().reindex(index=columns, columns=columns)
    count = correlation.notna().sum(axis=1) - 1
    average = (correlation.sum(axis=1, skipna=True) - 1.0) / count.replace(0, np.nan)
    score = 1.0 / (1.0 + penalty * average.clip(lower=0.0))
    return score.reindex(columns).fillna(1.0)


def _class_allocations(
    weights: pd.DataFrame,
    results: Mapping[str, PairBacktestResult],
) -> pd.DataFrame:
    classes = sorted(
        {
            asset_class
            for result in results.values()
            for asset_class in result.pair.asset_classes
        }
    )
    allocations = pd.DataFrame(0.0, index=weights.index, columns=classes, dtype=float)
    for name, result in results.items():
        pair_classes = tuple(dict.fromkeys(result.pair.asset_classes))
        if not pair_classes:
            continue
        share = weights[name] / len(pair_classes)
        for asset_class in pair_classes:
            allocations[asset_class] += share
    return allocations


def combine_pair_results(
    results: Mapping[str, PairBacktestResult],
    *,
    annualization: int = 252,
    volatility_lookback: int = 63,
    correlation_lookback: int = 126,
    correlation_penalty: float = 1.0,
    rebalance_frequency: int = 21,
    max_weight: float = 0.35,
    allocation_cost_bps: float = 1.0,
    target_volatility: float | None = None,
    max_leverage: float = 1.5,
) -> PortfolioResult:
    """Combine sleeves using lagged volatility, correlation, and optional vol targeting."""

    if len(results) < 2:
        raise ValueError("At least two pair results are required")
    if min(volatility_lookback, correlation_lookback) < 10 or rebalance_frequency < 1:
        raise ValueError("Invalid lookback or rebalance frequency")
    if correlation_penalty < 0:
        raise ValueError("correlation_penalty cannot be negative")
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must lie in (0, 1]")
    if allocation_cost_bps < 0:
        raise ValueError("allocation_cost_bps cannot be negative")
    if target_volatility is not None and target_volatility <= 0:
        raise ValueError("target_volatility must be positive or None")
    if max_leverage <= 0:
        raise ValueError("max_leverage must be positive")

    returns = pd.concat(
        {name: result.frame["net_return"] for name, result in results.items()}, axis=1
    ).fillna(0.0)
    ex_ante_volatility = (
        returns.rolling(volatility_lookback, min_periods=volatility_lookback)
        .std(ddof=1)
        .shift(1)
        * np.sqrt(annualization)
    )
    inverse_volatility = 1.0 / ex_ante_volatility.replace(0.0, np.nan)

    weights = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns, dtype=float)
    first_rebalance = max(volatility_lookback, correlation_lookback)
    for row in range(first_rebalance, len(weights), rebalance_frequency):
        history = returns.iloc[max(0, row - correlation_lookback) : row]
        scores = _diversification_scores(
            history,
            returns.columns,
            correlation_penalty,
        )
        allocation = _bounded_weights(inverse_volatility.iloc[row] * scores, max_weight)

        if target_volatility is not None and len(history) >= correlation_lookback:
            covariance = history.cov().to_numpy(dtype=float) * annualization
            vector = allocation.to_numpy(dtype=float)
            portfolio_variance = float(vector @ covariance @ vector)
            if np.isfinite(portfolio_variance) and portfolio_variance > 0.0:
                scale = min(max_leverage, target_volatility / np.sqrt(portfolio_variance))
                allocation *= scale
        weights.iloc[row] = allocation
    weights = weights.ffill().fillna(0.0)

    allocation_turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    gross_return = (weights * returns).sum(axis=1)
    allocation_cost = allocation_turnover * allocation_cost_bps / 10_000.0
    net_return = (gross_return - allocation_cost).rename("portfolio_return")
    equity = (1.0 + net_return.clip(lower=-0.999999)).cumprod().rename("equity")
    drawdown = drawdown_series(net_return)

    frame = returns.add_prefix("sleeve_")
    frame["gross_return"] = gross_return
    frame["allocation_cost"] = allocation_cost
    frame["portfolio_return"] = net_return
    frame["portfolio_turnover"] = allocation_turnover
    frame["gross_allocation"] = weights.abs().sum(axis=1)
    weight_square_sum = weights.pow(2).sum(axis=1).replace(0.0, np.nan)
    frame["effective_bets"] = (1.0 / weight_square_sum).fillna(0.0)
    frame["equity"] = equity
    frame["drawdown"] = drawdown

    metrics = performance_metrics(
        net_return, annualization=annualization, turnover=allocation_turnover
    )
    metrics["average_effective_bets"] = float(frame["effective_bets"].mean())
    metrics["average_gross_allocation"] = float(frame["gross_allocation"].mean())
    metrics["max_sleeve_weight"] = float(\n        weights.abs().to_numpy(dtype=float).max(initial=0.0)\n    )
    class_allocations = _class_allocations(weights, results)
    return PortfolioResult(
        frame=frame,
        weights=weights,
        class_allocations=class_allocations,
        metrics=metrics,
    )
