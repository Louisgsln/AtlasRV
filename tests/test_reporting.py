import json
from dataclasses import replace
from pathlib import Path

from atlas_rv.backtest.engine import PairBacktester
from atlas_rv.config import StrategyConfig
from atlas_rv.data.synthetic import generate_cross_asset_universe
from atlas_rv.reporting.report import write_research_bundle
from atlas_rv.research.diagnostics import diagnose_pair
from atlas_rv.risk.portfolio import combine_pair_results


def test_bundle_serializes_non_finite_diagnostics_as_null(tmp_path: Path) -> None:
    universe = generate_cross_asset_universe(observations=500, seed=23)
    pairs = universe.pairs[:3]
    backtester = PairBacktester()
    strategy = StrategyConfig(target_volatility=None)
    results = {
        pair.name: backtester.run(universe.prices, pair, strategy) for pair in pairs
    }
    diagnostics = {pair.name: diagnose_pair(universe.prices, pair) for pair in pairs}
    first_name = pairs[0].name
    diagnostics[first_name] = replace(diagnostics[first_name], half_life_days=float("inf"))
    portfolio = combine_pair_results(results, volatility_lookback=42)

    report = write_research_bundle(
        tmp_path,
        prices=universe.prices,
        pair_results=results,
        diagnostics=diagnostics,
        portfolio=portfolio,
    )
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    assert report.exists()
    assert summary["diagnostics"][first_name]["half_life_days"] is None
