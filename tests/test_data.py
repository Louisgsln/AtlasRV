import numpy as np
import pandas as pd
import pytest

from atlas_rv.data.store import LocalDataStore
from atlas_rv.data.synthetic import generate_cross_asset_universe
from atlas_rv.data.validation import clean_prices, validate_prices


def test_synthetic_universe_is_deterministic() -> None:
    first = generate_cross_asset_universe(observations=350, seed=42)
    second = generate_cross_asset_universe(observations=350, seed=42)

    pd.testing.assert_frame_equal(first.prices, second.prices)
    assert len(first.pairs) == 5
    assert np.isfinite(first.prices.to_numpy()).all()
    assert (first.prices > 0.0).all(axis=None)


def test_clean_and_validate_short_gaps() -> None:
    universe = generate_cross_asset_universe(observations=350)
    dirty = universe.prices.copy()
    dirty.iloc[10:12, 0] = np.nan

    cleaned = clean_prices(dirty, max_forward_fill=2)
    report = validate_prices(cleaned)

    assert cleaned.iloc[11, 0] == cleaned.iloc[9, 0]
    assert report.rows == 350
    assert report.duplicate_timestamps == 0


def test_local_store_round_trip(tmp_path: pytest.TempPathFactory) -> None:
    prices = generate_cross_asset_universe(observations=300).prices.iloc[:, :2]
    store = LocalDataStore(tmp_path)
    path = store.write("sample_prices", prices)

    assert path.exists()
    restored = store.read("sample_prices")
    pd.testing.assert_frame_equal(restored, prices, check_freq=False)

