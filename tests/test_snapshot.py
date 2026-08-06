import json
from pathlib import Path

import pandas as pd
import pytest

from atlas_rv.data.snapshot import read_snapshot, write_snapshot


def test_snapshot_round_trip_and_manifest_are_deterministic(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"B": [3.0, 4.0], "A": [1.0, 2.0]},
        index=pd.bdate_range("2024-01-01", periods=2),
    )
    first = write_snapshot(tmp_path / "one", frame)
    second = write_snapshot(tmp_path / "two", frame)
    first_manifest = json.loads(first.read_text(encoding="utf-8"))
    second_manifest = json.loads(second.read_text(encoding="utf-8"))

    assert first_manifest["sha256"] == second_manifest["sha256"]
    restored = read_snapshot(first)
    pd.testing.assert_frame_equal(
        restored,
        frame.sort_index(axis=1),
        check_freq=False,
    )


def test_snapshot_fails_closed_after_tampering(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0]},
        index=pd.bdate_range("2024-01-01", periods=3),
    )
    manifest = write_snapshot(tmp_path, frame)
    data_path = tmp_path / "prices.csv"
    data_path.write_text(data_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        read_snapshot(manifest)
