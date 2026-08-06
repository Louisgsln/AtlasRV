"""Typed configuration objects and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PairConfig:
    """Definition and economic rationale for a tradable relationship."""

    name: str
    y: str
    x: str
    asset_classes: tuple[str, ...] = ()
    thesis: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.x or not self.y:
            raise ValueError("Pair name, x, and y must be non-empty")
        if self.x == self.y:
            raise ValueError("A pair must contain two different instruments")


@dataclass(frozen=True)
class StrategyConfig:
    """Parameters that control signal formation, execution, and risk."""

    zscore_lookback: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    kalman_delta: float = 1e-6
    observation_variance: float = 1e-3
    cost_bps: float = 2.0
    target_volatility: float | None = 0.10
    volatility_lookback: int = 42
    max_leverage: float = 2.0

    def __post_init__(self) -> None:
        if self.zscore_lookback < 10:
            raise ValueError("zscore_lookback must be at least 10")
        if not 0 <= self.exit_z < self.entry_z < self.stop_z:
            raise ValueError("Thresholds must satisfy 0 <= exit < entry < stop")
        if self.kalman_delta <= 0 or self.observation_variance <= 0:
            raise ValueError("Kalman variances must be strictly positive")
        if self.cost_bps < 0:
            raise ValueError("cost_bps cannot be negative")
        if self.target_volatility is not None and self.target_volatility <= 0:
            raise ValueError("target_volatility must be positive or None")
        if self.volatility_lookback < 10:
            raise ValueError("volatility_lookback must be at least 10")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be strictly positive")


@dataclass(frozen=True)
class WalkForwardConfig:
    """Leakage-aware train/purge/test schedule and parameter grid."""

    train_size: int = 504
    test_size: int = 126
    purge_size: int = 5
    zscore_lookbacks: tuple[int, ...] = (40, 60, 90)
    entry_thresholds: tuple[float, ...] = (1.5, 2.0, 2.5)

    def __post_init__(self) -> None:
        if min(self.train_size, self.test_size) <= 0 or self.purge_size < 0:
            raise ValueError("Walk-forward sizes must be positive and purge non-negative")
        if not self.zscore_lookbacks or not self.entry_thresholds:
            raise ValueError("Walk-forward parameter grids cannot be empty")


@dataclass(frozen=True)
class UniverseConfig:
    """Complete research universe."""

    pairs: tuple[PairConfig, ...]
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    annualization: int = 252

    def __post_init__(self) -> None:
        if not self.pairs:
            raise ValueError("The universe must contain at least one pair")
        names = [pair.name for pair in self.pairs]
        if len(names) != len(set(names)):
            raise ValueError("Pair names must be unique")
        if self.annualization <= 0:
            raise ValueError("annualization must be strictly positive")


def _tuple_values(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    result = dict(payload)
    for key in keys:
        if key in result:
            result[key] = tuple(result[key])
    return result


def load_config(path: str | Path) -> UniverseConfig:
    """Load and validate a universe YAML file."""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")

    pair_payloads = raw.get("pairs", [])
    pairs = tuple(
        PairConfig(**_tuple_values(pair, ("asset_classes",))) for pair in pair_payloads
    )
    strategy = StrategyConfig(**raw.get("strategy", {}))
    walk_forward = WalkForwardConfig(
        **_tuple_values(
            raw.get("walk_forward", {}), ("zscore_lookbacks", "entry_thresholds")
        )
    )
    return UniverseConfig(
        pairs=pairs,
        strategy=strategy,
        walk_forward=walk_forward,
        annualization=int(raw.get("annualization", 252)),
    )
