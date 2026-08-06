"""Causal market-regime labels and conditional performance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from atlas_rv.risk.metrics import performance_metrics


@dataclass(frozen=True)
class RegimeAnalysis:
    labels: pd.DataFrame
    metrics: pd.DataFrame


def classify_market_regimes(
    benchmark_returns: pd.Series,
    *,
    volatility_lookback: int = 63,
    trend_lookback: int = 126,
    threshold_lookback: int = 252,
) -> pd.DataFrame:
    """Label each bar using statistics available strictly before that bar."""

    if min(volatility_lookback, trend_lookback, threshold_lookback) < 10:
        raise ValueError("Regime lookbacks must be at least 10")
    returns = benchmark_returns.astype(float).replace([np.inf, -np.inf], np.nan)
    ex_ante_volatility = (
        returns.rolling(volatility_lookback, min_periods=volatility_lookback)
        .std(ddof=1)
        .shift(1)
    )
    volatility_threshold = (
        ex_ante_volatility.rolling(
            threshold_lookback,
            min_periods=max(volatility_lookback, threshold_lookback // 3),
        )
        .median()
        .shift(1)
    )
    trailing_growth = (
        (1.0 + returns.fillna(0.0))
        .rolling(trend_lookback, min_periods=trend_lookback)
        .apply(np.prod, raw=True)
        .shift(1)
        - 1.0
    )

    volatility_label = pd.Series("unknown", index=returns.index, dtype="object")
    valid_volatility = ex_ante_volatility.notna() & volatility_threshold.notna()
    volatility_label.loc[
        valid_volatility & (ex_ante_volatility > volatility_threshold)
    ] = "high_vol"
    volatility_label.loc[
        valid_volatility & (ex_ante_volatility <= volatility_threshold)
    ] = "low_vol"

    trend_label = pd.Series("unknown", index=returns.index, dtype="object")
    trend_label.loc[trailing_growth > 0.0] = "up"
    trend_label.loc[trailing_growth <= 0.0] = "down"

    labels = pd.DataFrame(index=returns.index)
    labels["volatility_regime"] = volatility_label
    labels["trend_regime"] = trend_label
    labels["regime"] = volatility_label.astype(str) + "__" + trend_label.astype(str)
    unknown = (volatility_label == "unknown") | (trend_label == "unknown")
    labels.loc[unknown, "regime"] = "unknown"
    return labels


def performance_by_regime(
    strategy_returns: pd.Series,
    labels: pd.DataFrame,
    *,
    annualization: int = 252,
) -> pd.DataFrame:
    """Calculate strategy metrics within each fully observed regime."""

    aligned = pd.concat(
        [strategy_returns.rename("strategy_return"), labels["regime"]], axis=1
    ).dropna()
    records: list[dict[str, float | str]] = []
    for regime, group in aligned.groupby("regime", sort=True):
        if regime == "unknown" or len(group) < 5:
            continue
        metrics = performance_metrics(
            group["strategy_return"],
            annualization=annualization,
        )
        records.append(
            {
                "regime": str(regime),
                "observations": metrics["observations"],
                "annualized_return": metrics["annualized_return"],
                "annualized_volatility": metrics["annualized_volatility"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
            }
        )
    if not records:
        return pd.DataFrame(
            columns=[
                "observations",
                "annualized_return",
                "annualized_volatility",
                "sharpe",
                "max_drawdown",
            ]
        )
    return pd.DataFrame.from_records(records).set_index("regime")


def analyze_regimes(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    annualization: int = 252,
) -> RegimeAnalysis:
    labels = classify_market_regimes(benchmark_returns)
    metrics = performance_by_regime(
        strategy_returns,
        labels,
        annualization=annualization,
    )
    return RegimeAnalysis(labels=labels, metrics=metrics)
