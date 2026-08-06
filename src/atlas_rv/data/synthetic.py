"""Deterministic cross-asset data for reproducible demos and tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from atlas_rv.config import PairConfig


@dataclass(frozen=True)
class SyntheticUniverse:
    """Generated prices plus the relationships used to create them."""

    prices: pd.DataFrame
    pairs: tuple[PairConfig, ...]
    true_betas: dict[str, pd.Series]


@dataclass(frozen=True)
class _PairSpec:
    name: str
    y: str
    x: str
    asset_classes: tuple[str, str]
    x_start: float
    y_start: float
    beta: float
    x_volatility: float
    spread_speed: float
    spread_volatility: float
    thesis: str
    beta_shift: float = 0.0


_SPECS = (
    _PairSpec(
        "oil_energy",
        "ENERGY_EQ",
        "WTI",
        ("equity", "commodity"),
        72.0,
        85.0,
        0.72,
        0.012,
        0.08,
        0.018,
        "Energy equities and crude oil share a common cash-flow driver.",
    ),
    _PairSpec(
        "copper_miners",
        "MINERS",
        "COPPER",
        ("equity", "commodity"),
        4.1,
        38.0,
        1.28,
        0.011,
        0.06,
        0.020,
        "Copper miners embed operating leverage to the underlying metal.",
    ),
    _PairSpec(
        "gold_tips",
        "GOLD",
        "TIPS",
        ("commodity", "rates"),
        110.0,
        1850.0,
        0.48,
        0.006,
        0.05,
        0.012,
        "Gold and inflation-protected bonds share real-rate and inflation exposure.",
    ),
    _PairSpec(
        "credit_equity",
        "HY_CREDIT",
        "EQUITY",
        ("credit", "equity"),
        4200.0,
        78.0,
        0.32,
        0.009,
        0.10,
        0.010,
        "High-yield credit and equities are claims on the same balance sheets.",
    ),
    _PairSpec(
        "banks_curve",
        "BANKS",
        "CURVE_PROXY",
        ("equity", "rates"),
        100.0,
        48.0,
        0.86,
        0.007,
        0.04,
        0.018,
        "Bank valuations react to changes in the expected profitability of maturity transformation.",
        beta_shift=0.28,
    ),
)


def _ou_process(
    rng: np.random.Generator,
    observations: int,
    speed: float,
    volatility: float,
) -> np.ndarray:
    values = np.zeros(observations, dtype=float)
    shocks = rng.normal(0.0, volatility, observations)
    for index in range(1, observations):
        values[index] = (1.0 - speed) * values[index - 1] + shocks[index]
    return values


def _beta_path(observations: int, beta: float, shift: float) -> np.ndarray:
    path = np.full(observations, beta, dtype=float)
    if shift:
        start = int(observations * 0.68)
        end = min(observations, start + 126)
        path[start:end] += np.linspace(0.0, shift, end - start)
        path[end:] += shift
    return path


def generate_cross_asset_universe(
    observations: int = 1_500,
    start: str = "2018-01-02",
    seed: int = 7,
) -> SyntheticUniverse:
    """Create five economically labelled, partially cointegrated relationships.

    The final pair undergoes a slow beta shift. That deliberate failure mode
    prevents the demo from presenting every relationship as permanently stable.
    """

    if observations < 300:
        raise ValueError("At least 300 observations are required")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=observations)
    series: dict[str, pd.Series] = {}
    pairs: list[PairConfig] = []
    betas: dict[str, pd.Series] = {}

    for spec in _SPECS:
        x_shocks = rng.normal(0.00015, spec.x_volatility, observations)
        x_log = np.log(spec.x_start) + np.cumsum(x_shocks)
        beta_path = _beta_path(observations, spec.beta, spec.beta_shift)
        spread = _ou_process(
            rng,
            observations=observations,
            speed=spec.spread_speed,
            volatility=spec.spread_volatility,
        )
        intercept = np.log(spec.y_start) - spec.beta * np.log(spec.x_start)
        y_log = intercept + beta_path * x_log + spread

        series[spec.x] = pd.Series(np.exp(x_log), index=dates, name=spec.x)
        series[spec.y] = pd.Series(np.exp(y_log), index=dates, name=spec.y)
        pairs.append(
            PairConfig(
                name=spec.name,
                y=spec.y,
                x=spec.x,
                asset_classes=spec.asset_classes,
                thesis=spec.thesis,
            )
        )
        betas[spec.name] = pd.Series(beta_path, index=dates, name="true_beta")

    prices = pd.DataFrame(series, index=dates).sort_index(axis=1)
    return SyntheticUniverse(prices=prices, pairs=tuple(pairs), true_betas=betas)
