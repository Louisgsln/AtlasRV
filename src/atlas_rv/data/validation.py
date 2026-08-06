"""Data-quality gates used before any statistical test or backtest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataQualityReport:
    rows: int
    columns: int
    start: pd.Timestamp
    end: pd.Timestamp
    missing_by_symbol: dict[str, int]
    stale_fraction_by_symbol: dict[str, float]
    duplicate_timestamps: int

    @property
    def has_critical_issue(self) -> bool:
        return self.rows < 100 or self.duplicate_timestamps > 0


def validate_prices(prices: pd.DataFrame, minimum_rows: int = 100) -> DataQualityReport:
    """Validate the structural invariants expected by the research engine."""

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("Price index must be a pandas DatetimeIndex")
    if prices.empty or prices.shape[1] == 0:
        raise ValueError("Price frame cannot be empty")
    if len(prices) < minimum_rows:
        raise ValueError(f"At least {minimum_rows} observations are required")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Price index must be sorted in ascending order")
    duplicate_count = int(prices.index.duplicated().sum())
    if duplicate_count:
        raise ValueError("Price index contains duplicate timestamps")
    numeric = prices.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("Price frame contains infinite values")

    stale = numeric.diff().eq(0).mean()
    return DataQualityReport(
        rows=len(numeric),
        columns=numeric.shape[1],
        start=pd.Timestamp(numeric.index[0]),
        end=pd.Timestamp(numeric.index[-1]),
        missing_by_symbol={column: int(numeric[column].isna().sum()) for column in numeric},
        stale_fraction_by_symbol={column: float(stale[column]) for column in numeric},
        duplicate_timestamps=duplicate_count,
    )


def clean_prices(prices: pd.DataFrame, max_forward_fill: int = 3) -> pd.DataFrame:
    """Coerce, sort, and conservatively fill short data gaps.

    Longer gaps remain missing and are removed only when all requested assets
    have been aligned. This prevents silently carrying stale prices for weeks.
    """

    if max_forward_fill < 0:
        raise ValueError("max_forward_fill cannot be negative")
    cleaned = prices.copy()
    cleaned.index = pd.DatetimeIndex(cleaned.index).tz_localize(None)
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")].sort_index()
    cleaned = cleaned.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if max_forward_fill:
        cleaned = cleaned.ffill(limit=max_forward_fill)
    cleaned = cleaned.dropna(how="all")
    result: pd.DataFrame = cleaned.astype(float)
    return result
