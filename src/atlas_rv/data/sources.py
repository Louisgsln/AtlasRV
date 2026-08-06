"""Interchangeable market-data sources.

The research engine consumes a simple wide price frame. Providers live behind
one protocol so data access can change without contaminating strategy logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd


class MarketDataSource(Protocol):
    """Protocol implemented by all price sources."""

    def load(
        self,
        symbols: Sequence[str],
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return adjusted price levels indexed by timestamp."""


@dataclass(frozen=True)
class CsvSource:
    """Load a wide CSV with one date column and one column per symbol."""

    path: str | Path
    date_column: str = "date"

    def load(
        self,
        symbols: Sequence[str],
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        frame = pd.read_csv(self.path, parse_dates=[self.date_column])
        frame = frame.set_index(self.date_column).sort_index()
        missing = sorted(set(symbols) - set(frame.columns))
        if missing:
            raise KeyError(f"CSV is missing symbols: {', '.join(missing)}")
        result = frame.loc[:, list(symbols)].astype(float)
        if start is not None:
            result = result.loc[pd.Timestamp(start) :]
        if end is not None:
            result = result.loc[: pd.Timestamp(end)]
        return result


@dataclass(frozen=True)
class FredSource:
    """Download public FRED series without an API key.

    FRED observations are macro or rate levels, not necessarily tradable prices.
    They are intended for features and explanatory variables.
    """

    base_url: str = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def load(
        self,
        symbols: Sequence[str],
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            url = f"{self.base_url}?id={symbol}"
            series = pd.read_csv(url, na_values=".")
            date_column = next(
                (candidate for candidate in ("observation_date", "DATE") if candidate in series),
                None,
            )
            if date_column is None:
                raise KeyError("FRED response does not contain an observation date column")
            series[date_column] = pd.to_datetime(series[date_column])
            series = series.rename(columns={date_column: "date", symbol: symbol}).set_index("date")
            frames.append(series[[symbol]])
        result = pd.concat(frames, axis=1).sort_index().astype(float)
        if start is not None:
            result = result.loc[pd.Timestamp(start) :]
        if end is not None:
            result = result.loc[: pd.Timestamp(end)]
        return result


@dataclass(frozen=True)
class YahooFinanceSource:
    """Optional adjusted-price adapter powered by ``yfinance``."""

    auto_adjust: bool = True

    def load(
        self,
        symbols: Sequence[str],
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "YahooFinanceSource requires the optional data dependency: "
                "pip install 'atlas-rv[data]'"
            ) from exc

        raw = yf.download(
            list(symbols),
            start=None if start is None else str(pd.Timestamp(start).date()),
            end=None if end is None else str(pd.Timestamp(end).date()),
            auto_adjust=self.auto_adjust,
            progress=False,
            threads=True,
        )
        if raw.empty:
            raise ValueError("Yahoo Finance returned no observations")

        if isinstance(raw.columns, pd.MultiIndex):
            field = "Close"
            if field not in raw.columns.get_level_values(0):
                raise KeyError("Downloaded data does not contain a Close field")
            result = raw[field]
        else:
            result = raw[["Close"]].rename(columns={"Close": symbols[0]})
        result = result.reindex(columns=list(symbols))
        result.index = pd.DatetimeIndex(result.index).tz_localize(None)
        output: pd.DataFrame = result.astype(float)
        return output
