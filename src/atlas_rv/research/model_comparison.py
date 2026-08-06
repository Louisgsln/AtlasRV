"""Comparable evaluation of causal hedge-ratio estimators."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from atlas_rv.backtest.engine import PairBacktestResult, PairBacktester
from atlas_rv.config import PairConfig, StrategyConfig


@dataclass(frozen=True)
class ModelComparisonResult:
    results: dict[str, PairBacktestResult]
    metrics: pd.DataFrame


def compare_hedge_models(
    prices: pd.DataFrame,
    pair: PairConfig,
    base_config: StrategyConfig,
    *,
    models: tuple[str, ...] = ("expanding_ols", "rolling_ols", "kalman"),
    annualization: int = 252,
) -> ModelComparisonResult:
    """Run identical signals and costs while changing only the hedge estimator."""

    if len(set(models)) != len(models):
        raise ValueError("Model names must be unique")
    backtester = PairBacktester(annualization=annualization)
    results = {
        model: backtester.run(
            prices,
            pair,
            replace(base_config, hedge_model=model),
        )
        for model in models
    }
    metrics = pd.DataFrame(
        {model: result.metrics for model, result in results.items()}
    ).T
    metrics.index.name = "hedge_model"
    metrics["sharpe_rank"] = metrics["sharpe"].rank(
        ascending=False,
        method="min",
    )
    return ModelComparisonResult(results=results, metrics=metrics.sort_values("sharpe_rank"))
