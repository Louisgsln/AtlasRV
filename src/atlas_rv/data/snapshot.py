"""Immutable, checksum-verified research data snapshots."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SnapshotManifest:
    dataset_file: str
    sha256: str
    rows: int
    columns: int
    start: str
    end: str
    symbols: tuple[str, ...]


def _canonical_csv(frame: pd.DataFrame) -> str:
    canonical = frame.copy().sort_index().sort_index(axis=1)
    if not isinstance(canonical.index, pd.DatetimeIndex):
        raise TypeError("Snapshot index must be a DatetimeIndex")
    return canonical.to_csv(
        index_label="date",
        date_format="%Y-%m-%dT%H:%M:%S",
        float_format="%.12g",
        lineterminator="\n",
    )


def write_snapshot(
    directory: str | Path,
    frame: pd.DataFrame,
    *,
    name: str = "prices",
) -> Path:
    """Write canonical CSV bytes and a deterministic integrity manifest."""

    if frame.empty or frame.shape[1] == 0:
        raise ValueError("Cannot snapshot an empty frame")
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    csv_name = f"{name}.csv"
    csv_path = output / csv_name
    payload = _canonical_csv(frame).encode("utf-8")
    csv_path.write_bytes(payload)

    manifest = SnapshotManifest(
        dataset_file=csv_name,
        sha256=hashlib.sha256(payload).hexdigest(),
        rows=len(frame),
        columns=frame.shape[1],
        start=str(pd.Timestamp(frame.index.min())),
        end=str(pd.Timestamp(frame.index.max())),
        symbols=tuple(sorted(str(column) for column in frame.columns)),
    )
    manifest_path = output / f"{name}.manifest.json"
    manifest_path.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def read_snapshot(manifest_path: str | Path) -> pd.DataFrame:
    """Read a snapshot and fail closed when its bytes no longer match."""

    path = Path(manifest_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    data_path = path.parent / str(raw["dataset_file"])
    payload = data_path.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != raw["sha256"]:
        raise ValueError("Snapshot checksum mismatch")
    frame = pd.read_csv(data_path, index_col="date", parse_dates=True)
    if len(frame) != int(raw["rows"]) or frame.shape[1] != int(raw["columns"]):
        raise ValueError("Snapshot dimensions do not match the manifest")
    return frame.astype(float)
