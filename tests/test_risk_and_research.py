import numpy as np
import pandas as pd

from atlas_rv.backtest.engine import PairBacktester
from atlas_rv.config import StrategyConfig
from atlas_rv.data.synthetic import generate_cross_asset_universe
from atlas_rv.research.diagnostics import diagnose_pair
from atlas_rv.risk.metrics import drawdown_series, performance_metrics
from atlas_rv.risk.portfolio import combine_pair_results


def test_performance_metrics_and_drawdown() -> None:
    returns = pd.Series([0.10, -0.10, 0.05, -0.02, 0.03])
    metrics = performance_metrics(returns, annualization=5)
    drawdown = drawdown_series(returns)

    assert metrics["observations"] == 5.0
    assert metrics["max_drawdown"] < 0.0
    assert drawdown.iloc[0] == 0.0
    assert np.isfinite(metrics["sharpe"])


def test_diagnostics_reject_the_deliberate_beta_break() -> None:
    universe = generate_cross_asset_universe(observations=1_200, seed=7)
    diagnostics = {pair.name: diagnose_pair(universe.prices, pair) for pair in universe.pairs}

    assert diagnostics["oil_energy"].coint_pvalue < 0.05
    assert diagnostics["oil_energy"].passes_research_gate
    assert not diagnostics["banks_curve"].passes_research_gate
    assert diagnostics["banks_curve"].beta_instability > diagnostics["oil_energy"].beta_instability


def test_inverse_vol_portfolio_is_causal_and_weight_capped() -> None:
    universe = generate_cross_asset_universe(observations=700, seed=5)
    backtester = PairBacktester()
    strategy = StrategyConfig(target_volatility=None)
    results = {
        pair.name: backtester.run(universe.prices, pair, strategy)
        for pair in universe.pairs[:3]
    }
    portfolio = combine_pair_results(results, max_weight=0.50, volatility_lookback=42)

    assert portfolio.weights.max(axis=None) <= 0.50 + 1e-12
    active_sums = portfolio.weights.sum(axis=1)
    assert (active_sums <= 1.0 + 1e-12).all()
    assert np.isclose(active_sums.max(), 1.0)
    assert np.isfinite(portfolio.frame["portfolio_return"]).all()
