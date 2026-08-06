"""AtlasRV: causal cross-asset relative-value research."""

from atlas_rv.config import (
    PairConfig,
    PortfolioConfig,
    StrategyConfig,
    UniverseConfig,
    load_config,
)

__all__ = [
    "PairConfig",
    "PortfolioConfig",
    "StrategyConfig",
    "UniverseConfig",
    "load_config",
]
__version__ = "0.2.0"
