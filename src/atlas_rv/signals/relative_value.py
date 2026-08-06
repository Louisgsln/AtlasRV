"""Causal relative-value signal construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore(
    values: pd.Series,
    lookback: int,
    minimum_periods: int | None = None,
) -> pd.Series:
    """Standardize today's value against a trailing window ending yesterday."""

    if lookback < 2:
        raise ValueError("lookback must be at least two")
    minimum = lookback if minimum_periods is None else minimum_periods
    if not 2 <= minimum <= lookback:
        raise ValueError("minimum_periods must be between two and lookback")
    trailing_mean = values.rolling(lookback, min_periods=minimum).mean().shift(1)
    trailing_std = values.rolling(lookback, min_periods=minimum).std(ddof=1).shift(1)
    zscore = (values - trailing_mean) / trailing_std.replace(0.0, np.nan)
    return zscore.replace([np.inf, -np.inf], np.nan).rename("zscore")


def generate_positions(
    zscore: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
) -> pd.Series:
    """Create a stateful long/flat/short spread position.

    A stop activates a cooldown: the strategy cannot re-enter until the spread
    has first moved back inside the entry band. This avoids immediate repeated
    bets during a structural break.
    """

    if not 0 <= exit_z < entry_z < stop_z:
        raise ValueError("Thresholds must satisfy 0 <= exit < entry < stop")

    positions = np.zeros(len(zscore), dtype=np.int8)
    state = 0
    cooldown = False
    for index, value in enumerate(zscore.to_numpy(dtype=float)):
        if not np.isfinite(value):
            positions[index] = state
            continue

        absolute = abs(value)
        if cooldown:
            if absolute < entry_z:
                cooldown = False
            positions[index] = 0
            continue

        if state == 0:
            if entry_z <= value < stop_z:
                state = -1
            elif -stop_z < value <= -entry_z:
                state = 1
        elif absolute >= stop_z:
            state = 0
            cooldown = True
        elif absolute <= exit_z:
            state = 0

        positions[index] = state

    return pd.Series(positions, index=zscore.index, name="position", dtype="int8")
