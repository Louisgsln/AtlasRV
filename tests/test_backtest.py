from dataclasses import replace
from itertools import pairwise

import numpy as np
import pandas as pd

from atlas_rv.backtest.engine import PairBacktester
from atlas_rv.backtest.walk_forward import build_walk_forward_folds, run_walk_forward
from atlas_rv.config import StrategyConfig, WalkForwardConfig
from atlas_rv.data.synthetic import generate_cross_asset_universe


def test_backtest_is_causal_under_future_price_perturbation() -> None:
    universe = generate_cross_asset_universe(observations=650, seed=11)
    pair = universe.pairs[0]
    config = StrategyConfig(target_volatility=None)
    backtester = PairBacktester()
    baseline = backtester.run(universe.prices, pair, config)

    changed_prices = universe.prices.copy()
    changed_prices.loc[changed_prices.index[500] :, pair.y] *= 1.7
    changed = backtester.run(changed_prices, pair, config)

    columns = ["beta", "spread", "zscore", "position", "net_return"]
    pd.testing.assert_frame_equal(
        baseline.frame.loc[: baseline.frame.index[499], columns],
        changed.frame.loc[: changed.frame.index[499], columns],
    )


def test_transaction_costs_are_explicit_and_reduce_compounded_return() -> None:
    universe = generate_cross_asset_universe(observations=700, seed=3)
    pair = universe.pairs[1]
    base = StrategyConfig(target_volatility=None, cost_bps=0.0)
    expensive = replace(base, cost_bps=25.0)
    backtester = PairBacktester()

    free_result = backtester.run(universe.prices, pair, base)
    costly_result = backtester.run(universe.prices, pair, expensive)

    assert np.allclose(free_result.frame["gross_return"], costly_result.frame["gross_return"])
    assert costly_result.frame["transaction_cost"].sum() > 0.0
    assert costly_result.metrics["total_return"] < free_result.metrics["total_return"]


def test_walk_forward_folds_have_a_purge_and_disjoint_tests() -> None:
    index = pd.bdate_range("2015-01-01", periods=1_000)
    config = WalkForwardConfig(train_size=400, test_size=100, purge_size=7)
    folds = build_walk_forward_folds(index, config)

    assert len(folds) == 5
    assert all(fold.test_start - fold.train_end == 7 for fold in folds)
    assert all(left.test_end <= right.test_start for left, right in pairwise(folds))


def test_walk_forward_returns_only_held_out_rows() -> None:
    universe = generate_cross_asset_universe(observations=800, seed=19)
    pair = universe.pairs[0]
    schedule = WalkForwardConfig(
        train_size=400,
        test_size=100,
        purge_size=5,
        zscore_lookbacks=(40, 60),
        entry_thresholds=(1.5, 2.0),
    )
    result = run_walk_forward(
        universe.prices,
        pair,
        StrategyConfig(target_volatility=None),
        schedule,
    )

    assert len(result.frame) == len(result.folds) * schedule.test_size
    assert result.frame.index.is_unique
    assert result.frame["fold"].nunique() == len(result.folds)
