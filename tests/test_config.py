from pathlib import Path

import pytest

from atlas_rv.config import PairConfig, StrategyConfig, load_config


def test_loads_example_universe() -> None:
    config = load_config(Path("configs/universe.yml"))

    assert len(config.pairs) == 5
    assert config.pairs[0].asset_classes == ("equity", "commodity")
    assert config.strategy.entry_z == 2.0
    assert config.walk_forward.purge_size == 5


def test_rejects_invalid_pair_and_thresholds() -> None:
    with pytest.raises(ValueError, match="different"):
        PairConfig(name="bad", x="SAME", y="SAME")

    with pytest.raises(ValueError, match="Thresholds"):
        StrategyConfig(entry_z=0.5, exit_z=1.0)

