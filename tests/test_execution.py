import numpy as np
import pandas as pd

from atlas_rv.backtest.engine import PairBacktester
from atlas_rv.config import StrategyConfig
from atlas_rv.data.synthetic import generate_cross_asset_universe
from atlas_rv.execution.costs import ExecutionCostModel


def test_cost_model_reconciles_every_component() -> None:
    index = pd.bdate_range("2024-01-01", periods=5)
    targets = pd.DataFrame(
        {
            "weight_y": [0.0, 0.5, 0.5, 0.0, 0.0],
            "weight_x": [0.0, -0.5, -0.5, 0.0, 0.0],
        },
        index=index,
    )
    held = targets.shift(1).fillna(0.0)
    result = ExecutionCostModel(
        commission_bps=0.2,
        half_spread_bps=0.8,
        slippage_bps=0.5,
        impact_coefficient_bps=1.0,
        borrow_rate_bps_annual=50.0,
        financing_rate_bps_annual=25.0,
    ).calculate(targets, held, annualization=252)

    components = [
        "legacy_cost",
        "commission_cost",
        "spread_cost",
        "slippage_cost",
        "impact_cost",
        "borrow_cost",
        "financing_cost",
    ]
    assert (result.frame[components] >= 0.0).all(axis=None)
    assert np.allclose(result.total, result.frame[components].sum(axis=1))
    assert result.frame["turnover"].sum() == 2.0


def test_backtest_exposes_cost_attribution_and_reconciles_net_return() -> None:
    universe = generate_cross_asset_universe(observations=500, seed=31)
    pair = universe.pairs[0]
    config = StrategyConfig(
        cost_bps=0.0,
        commission_bps=0.2,
        half_spread_bps=0.8,
        slippage_bps=0.5,
        impact_coefficient_bps=0.5,
        borrow_rate_bps_annual=20.0,
        financing_rate_bps_annual=15.0,
        target_volatility=None,
    )
    result = PairBacktester().run(universe.prices, pair, config)

    assert np.allclose(
        result.frame["net_return"],
        result.frame["gross_return"] - result.frame["transaction_cost"],
    )
    assert result.frame["transaction_cost"].sum() > 0.0
    assert result.metrics["total_transaction_cost"] > 0.0
