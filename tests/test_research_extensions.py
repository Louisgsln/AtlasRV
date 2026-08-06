import numpy as np
import pandas as pd

from atlas_rv.data.synthetic import generate_cross_asset_universe
from atlas_rv.research.diagnostics import diagnose_pair
from atlas_rv.research.multiple_testing import (
    apply_false_discovery_control,
    benjamini_hochberg,
)
from atlas_rv.research.regimes import classify_market_regimes, performance_by_regime


def test_benjamini_hochberg_is_monotone_in_ranked_pvalues() -> None:
    adjusted = benjamini_hochberg(
        {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.20}
    )

    assert np.isclose(adjusted["a"], 0.04)
    assert np.isclose(adjusted["c"], 0.04 * 4.0 / 3.0)
    assert np.isclose(adjusted["b"], adjusted["c"])
    assert np.isclose(adjusted["d"], 0.20)


def test_false_discovery_control_never_weakens_the_pair_gate() -> None:
    universe = generate_cross_asset_universe(observations=800, seed=7)
    raw = {
        pair.name: diagnose_pair(universe.prices, pair)
        for pair in universe.pairs
    }
    controlled = apply_false_discovery_control(raw, alpha=0.05)

    assert set(controlled) == set(raw)
    assert all(item.coint_qvalue >= item.coint_pvalue for item in controlled.values())
    assert all(
        not controlled[name].passes_research_gate
        for name, item in raw.items()
        if not item.passes_research_gate
    )


def test_regime_labels_are_causal_and_support_attribution() -> None:
    generator = np.random.default_rng(41)
    index = pd.bdate_range("2018-01-01", periods=800)
    benchmark = pd.Series(generator.normal(0.0002, 0.01, len(index)), index=index)
    changed = benchmark.copy()
    changed.iloc[700:] += 0.10

    baseline = classify_market_regimes(benchmark)
    perturbed = classify_market_regimes(changed)
    pd.testing.assert_frame_equal(baseline.iloc[:700], perturbed.iloc[:700])

    strategy = 0.2 * benchmark + pd.Series(
        generator.normal(0.0, 0.002, len(index)),
        index=index,
    )
    metrics = performance_by_regime(strategy, baseline)
    assert not metrics.empty
    assert {"sharpe", "max_drawdown"}.issubset(metrics.columns)
