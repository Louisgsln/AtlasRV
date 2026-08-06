"""Causal inverse-volatility allocation across independent RV sleeves."""

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
    metrics: dict[str, float]


def _bounded_weights(values: pd.Series, maximum: float) -> pd.Series:
    finite = values.replace([np.inf, -np.inf], np.nan).dropna().clip(lower=0.0)
    result = pd.Series(0.0, index=values.index, dtype=float)
    if finite.empty or float(finite.sum()) <= 0.0:
        return result
    if len(finite) * maximum < 1.0 - 1e-12:
        # Not enough independent sleeves are currently estimable. Respect the
        # concentration limit and leave the remainder in cash.
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


def combine_pair_results(
    results: Mapping[str, PairBacktestResult],
    *,
    annualization: int = 252,
    volatility_lookback: int = 63,
    rebalance_frequency: int = 21,
    max_weight: float = 0.35,
    allocation_cost_bps: float = 1.0,
) -> PortfolioResult:
    """Combine pair sleeves using lagged volatility and periodic rebalancing."""

    if len(results) < 2:
        raise ValueError("At least two pair results are required")
    if volatility_lookback < 10 or rebalance_frequency < 1:
        raise ValueError("Invalid lookback or rebalance frequency")
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must lie in (0, 1]")

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
    for row in range(volatility_lookback, len(weights), rebalance_frequency):
        weights.iloc[row] = _bounded_weights(inverse_volatility.iloc[row], max_weight)
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
    frame["equity"] = equity
    frame["drawdown"] = drawdown
    metrics = performance_metrics(
        net_return, annualization=annualization, turnover=allocation_turnover
    )
    return PortfolioResult(frame=frame, weights=weights, metrics=metrics)
