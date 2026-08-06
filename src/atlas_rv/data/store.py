"""Small local cache with an auditable CSV fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class LocalDataStore:
    """Persist timestamp-indexed frames without coupling research to a database."""

    root: str | Path

    @property
    def root_path(self) -> Path:
        path = Path(self.root)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write(self, name: str, frame: pd.DataFrame) -> Path:
        safe_name = self._safe_name(name)
        parquet_path = self.root_path / f"{safe_name}.parquet"
        try:
            frame.to_parquet(parquet_path)
            return parquet_path
        except (ImportError, ValueError):
            csv_path = self.root_path / f"{safe_name}.csv.gz"
            frame.to_csv(csv_path, index_label="date", compression="gzip")
            return csv_path

    def read(self, name: str) -> pd.DataFrame:
        safe_name = self._safe_name(name)
        parquet_path = self.root_path / f"{safe_name}.parquet"
        csv_path = self.root_path / f"{safe_name}.csv.gz"
        if parquet_path.exists():
            frame = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            frame = pd.read_csv(csv_path, index_col="date", parse_dates=True)
        else:
            raise FileNotFoundError(f"No cached dataset named {safe_name!r}")
        frame.index = pd.DatetimeIndex(frame.index)
        frame.index.name = None
        return frame.sort_index()

    @staticmethod
    def _safe_name(name: str) -> str:
        if not name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in name.lower()):
            raise ValueError("Dataset name may contain only letters, digits, underscores, and hyphens")
        return name.lower()
