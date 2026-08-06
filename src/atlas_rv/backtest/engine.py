"""A close-to-close pair backtester with explicit information timing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from atlas_rv.config import PairConfig, StrategyConfig
from atlas_rv.models.kalman import KalmanDynamicRegression
from atlas_rv.risk.metrics import drawdown_series, performance_metrics
from atlas_rv.signals.relative_value import generate_positions, rolling_zscore


@dataclass(frozen=True)
class PairBacktestResult:
    pair: PairConfig
    config: StrategyConfig
    frame: pd.DataFrame
    metrics: dict[str, float]


class PairBacktester:
    """Backtest one dynamic relative-value relationship.

    Timing convention
    -----------------
    Prices observed at close ``t`` update the model and produce target weights.
    Those weights earn the return from close ``t`` to close ``t+1``. Costs are
    charged when the new weights become active. Every model input is therefore
    available before the return it is used to trade.
    """

    def __init__(self, annualization: int = 252) -> None:
        if annualization <= 0:
            raise ValueError("annualization must be strictly positive")
        self.annualization = annualization

    def run(
        self,
        prices: pd.DataFrame,
        pair: PairConfig,
        config: StrategyConfig | None = None,
    ) -> PairBacktestResult:
        strategy = config or StrategyConfig()
        missing = sorted({pair.x, pair.y} - set(prices.columns))
        if missing:
            raise KeyError(f"Missing pair instruments: {', '.join(missing)}")

        aligned = prices[[pair.y, pair.x]].dropna().astype(float).sort_index()
        minimum_rows = max(strategy.zscore_lookback + 20, strategy.volatility_lookback + 20)
        if len(aligned) < minimum_rows:
            raise ValueError(f"Pair backtest requires at least {minimum_rows} aligned rows")
        if (aligned <= 0.0).any(axis=None):
            raise ValueError("Dynamic log-price regression requires strictly positive levels")
        if aligned.index.has_duplicates:
            raise ValueError("Price index cannot contain duplicates")

        log_prices = pd.DataFrame(
            np.log(aligned.to_numpy(dtype=float)),
            index=aligned.index,
            columns=aligned.columns,
        )
        model = KalmanDynamicRegression(
            delta=strategy.kalman_delta,
            observation_variance=strategy.observation_variance,
        ).fit(log_prices[pair.y], log_prices[pair.x])
        zscore = rolling_zscore(model.innovation, strategy.zscore_lookback)
        position = generate_positions(
            zscore,
            entry_z=strategy.entry_z,
            exit_z=strategy.exit_z,
            stop_z=strategy.stop_z,
        )

        beta = model.beta.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
        denominator = 1.0 + beta.abs()
        unscaled_targets = pd.DataFrame(index=aligned.index)
        unscaled_targets["weight_y"] = position.astype(float) / denominator
        unscaled_targets["weight_x"] = -position.astype(float) * beta / denominator

        asset_returns = aligned.pct_change(fill_method=None).fillna(0.0)
        unscaled_held = unscaled_targets.shift(1).fillna(0.0)
        unscaled_return = (
            unscaled_held["weight_y"] * asset_returns[pair.y]
            + unscaled_held["weight_x"] * asset_returns[pair.x]
        )

        if strategy.target_volatility is None:
            target_scale = pd.Series(1.0, index=aligned.index, name="target_scale")
        else:
            trailing_volatility = (
                unscaled_return.rolling(
                    strategy.volatility_lookback,
                    min_periods=strategy.volatility_lookback,
                ).std(ddof=1)
                * np.sqrt(self.annualization)
            )
            target_scale = (
                strategy.target_volatility / trailing_volatility.replace(0.0, np.nan)
            ).clip(lower=0.0, upper=strategy.max_leverage)
            target_scale = target_scale.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            target_scale.name = "target_scale"

        scaled_targets = unscaled_targets.mul(target_scale, axis=0)
        held_weights = scaled_targets.shift(1).fillna(0.0)
        gross_return = (
            held_weights["weight_y"] * asset_returns[pair.y]
            + held_weights["weight_x"] * asset_returns[pair.x]
        ).rename("gross_return")

        turnover = scaled_targets.diff().abs().sum(axis=1).shift(1).fillna(0.0)
        turnover.name = "turnover"
        transaction_cost = (turnover * strategy.cost_bps / 10_000.0).rename(
            "transaction_cost"
        )
        net_return = (gross_return - transaction_cost).rename("net_return")
        equity = (1.0 + net_return.clip(lower=-0.999999)).cumprod().rename("equity")
        drawdown = drawdown_series(net_return)

        frame = pd.DataFrame(index=aligned.index)
        frame[f"price_{pair.y}"] = aligned[pair.y]
        frame[f"price_{pair.x}"] = aligned[pair.x]
        frame["intercept"] = model.intercept
        frame["beta"] = beta
        frame["spread"] = model.innovation
        frame["innovation_variance"] = model.innovation_variance
        frame["zscore"] = zscore
        frame["position"] = position
        frame["target_scale"] = target_scale
        frame["target_weight_y"] = scaled_targets["weight_y"]
        frame["target_weight_x"] = scaled_targets["weight_x"]
        frame["held_weight_y"] = held_weights["weight_y"]
        frame["held_weight_x"] = held_weights["weight_x"]
        frame["gross_return"] = gross_return
        frame["turnover"] = turnover
        frame["transaction_cost"] = transaction_cost
        frame["net_return"] = net_return
        frame["equity"] = equity
        frame["drawdown"] = drawdown

        metrics = performance_metrics(
            net_return,
            annualization=self.annualization,
            turnover=turnover,
            positions=position,
        )
        return PairBacktestResult(pair=pair, config=strategy, frame=frame, metrics=metrics)
