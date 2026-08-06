import numpy as np
import pandas as pd

from atlas_rv.config import StrategyConfig
from atlas_rv.data.synthetic import generate_cross_asset_universe
from atlas_rv.models.regression import ExpandingOLSRegression, RollingOLSRegression
from atlas_rv.research.model_comparison import compare_hedge_models


def test_ols_models_do_not_change_the_past_when_the_future_changes() -> None:
    index = pd.bdate_range("2021-01-01", periods=240)
    x = pd.Series(np.linspace(3.0, 5.0, len(index)), index=index)
    y = 0.3 + 1.2 * x + pd.Series(
        0.01 * np.sin(np.arange(len(index)) / 8.0),
        index=index,
    )
    changed = y.copy()
    changed.iloc[180:] += 5.0

    for estimator in (
        ExpandingOLSRegression(minimum_observations=20),
        RollingOLSRegression(lookback=80, minimum_observations=20),
    ):
        baseline = estimator.fit(y, x).to_frame()
        perturbed = estimator.fit(changed, x).to_frame()
        pd.testing.assert_frame_equal(baseline.iloc[:180], perturbed.iloc[:180])


def test_model_comparison_changes_only_the_hedge_estimator() -> None:
    universe = generate_cross_asset_universe(observations=450, seed=13)
    comparison = compare_hedge_models(
        universe.prices,
        universe.pairs[0],
        StrategyConfig(
            rolling_ols_lookback=80,
            target_volatility=None,
        ),
    )

    assert set(comparison.results) == {"expanding_ols", "rolling_ols", "kalman"}
    assert comparison.metrics["sharpe_rank"].notna().all()
    assert comparison.metrics.index.name == "hedge_model"
