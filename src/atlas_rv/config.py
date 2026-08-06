"""Typed configuration objects and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_HEDGE_MODELS = frozenset({"kalman", "rolling_ols", "expanding_ols"})


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
    """Parameters controlling the hedge model, signal, execution, and sleeve risk."""

    hedge_model: str = "kalman"
    rolling_ols_lookback: int = 126
    zscore_lookback: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    kalman_delta: float = 1e-6
    observation_variance: float = 1e-3

    # cost_bps is retained as a backwards-compatible all-in linear cost.
    # The explicit components below can instead be calibrated independently.
    cost_bps: float = 2.0
    commission_bps: float = 0.0
    half_spread_bps: float = 0.0
    slippage_bps: float = 0.0
    impact_coefficient_bps: float = 0.0
    borrow_rate_bps_annual: float = 0.0
    financing_rate_bps_annual: float = 0.0

    target_volatility: float | None = 0.10
    volatility_lookback: int = 42
    max_leverage: float = 2.0

    def __post_init__(self) -> None:
        if self.hedge_model not in _HEDGE_MODELS:
            choices = ", ".join(sorted(_HEDGE_MODELS))
            raise ValueError(f"hedge_model must be one of: {choices}")
        if self.rolling_ols_lookback < 20:
            raise ValueError("rolling_ols_lookback must be at least 20")
        if self.zscore_lookback < 10:
            raise ValueError("zscore_lookback must be at least 10")
        if not 0 <= self.exit_z < self.entry_z < self.stop_z:
            raise ValueError("Thresholds must satisfy 0 <= exit < entry < stop")
        if self.kalman_delta <= 0 or self.observation_variance <= 0:
            raise ValueError("Kalman variances must be strictly positive")
        cost_fields = (
            self.cost_bps,
            self.commission_bps,
            self.half_spread_bps,
            self.slippage_bps,
            self.impact_coefficient_bps,
            self.borrow_rate_bps_annual,
            self.financing_rate_bps_annual,
        )
        if any(value < 0 for value in cost_fields):
            raise ValueError("Execution-cost inputs cannot be negative")
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
class PortfolioConfig:
    """Causal cross-sleeve allocation and portfolio-risk settings."""

    volatility_lookback: int = 63
    correlation_lookback: int = 126
    correlation_penalty: float = 1.0
    rebalance_frequency: int = 21
    max_weight: float = 0.35
    allocation_cost_bps: float = 1.0
    target_volatility: float | None = None
    max_leverage: float = 1.5

    def __post_init__(self) -> None:
        if min(self.volatility_lookback, self.correlation_lookback) < 10:
            raise ValueError("Portfolio lookbacks must be at least 10")
        if self.correlation_penalty < 0:
            raise ValueError("correlation_penalty cannot be negative")
        if self.rebalance_frequency < 1:
            raise ValueError("rebalance_frequency must be positive")
        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must lie in (0, 1]")
        if self.allocation_cost_bps < 0:
            raise ValueError("allocation_cost_bps cannot be negative")
        if self.target_volatility is not None and self.target_volatility <= 0:
            raise ValueError("target_volatility must be positive or None")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be strictly positive")


@dataclass(frozen=True)
class UniverseConfig:
    """Complete cross-asset research universe."""

    pairs: tuple[PairConfig, ...]
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    annualization: int = 252
    fdr_alpha: float = 0.05

    def __post_init__(self) -> None:
        if not self.pairs:
            raise ValueError("The universe must contain at least one pair")
        names = [pair.name for pair in self.pairs]
        if len(names) != len(set(names)):
            raise ValueError("Pair names must be unique")
        if self.annualization <= 0:
            raise ValueError("annualization must be strictly positive")
        if not 0 < self.fdr_alpha < 1:
            raise ValueError("fdr_alpha must lie in (0, 1)")


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
    portfolio = PortfolioConfig(**raw.get("portfolio", {}))
    return UniverseConfig(
        pairs=pairs,
        strategy=strategy,
        walk_forward=walk_forward,
        portfolio=portfolio,
        annualization=int(raw.get("annualization", 252)),
        fdr_alpha=float(raw.get("fdr_alpha", 0.05)),
    )
