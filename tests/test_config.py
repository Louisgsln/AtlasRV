from pathlib import Path

import pytest

from atlas_rv.config import PairConfig, PortfolioConfig, StrategyConfig, load_config


def test_loads_example_universe_and_v02_controls() -> None:
    config = load_config(Path("configs/universe.yml"))

    assert len(config.pairs) == 5
    assert config.pairs[0].asset_classes == ("equity", "commodity")
    assert config.strategy.entry_z == 2.0
    assert config.strategy.hedge_model == "kalman"
    assert config.walk_forward.purge_size == 5
    assert config.portfolio.correlation_penalty == 1.0
    assert config.fdr_alpha == 0.05


def test_loads_fred_multi_asset_study() -> None:
    config = load_config(Path("configs/fred_market_study.yml"))

    assert len(config.pairs) == 6
    assert {asset for pair in config.pairs for asset in pair.asset_classes} == {
        "commodity",
        "credit",
        "crypto",
        "equity",
        "fx",
        "rates",
        "volatility",
    }
    assert config.portfolio.max_weight == 0.25


def test_rejects_invalid_pair_thresholds_model_and_portfolio() -> None:
    with pytest.raises(ValueError, match="different"):
        PairConfig(name="bad", x="SAME", y="SAME")

    with pytest.raises(ValueError, match="Thresholds"):
        StrategyConfig(entry_z=0.5, exit_z=1.0)

    with pytest.raises(ValueError, match="hedge_model"):
        StrategyConfig(hedge_model="oracle")

    with pytest.raises(ValueError, match="max_weight"):
        PortfolioConfig(max_weight=1.1)
