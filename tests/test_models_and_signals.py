import numpy as np
import pandas as pd

from atlas_rv.models.kalman import KalmanDynamicRegression
from atlas_rv.signals.relative_value import generate_positions, rolling_zscore


def test_kalman_tracks_a_slowly_changing_beta() -> None:
    index = pd.bdate_range("2020-01-01", periods=600)
    x = pd.Series(np.linspace(3.5, 5.0, len(index)), index=index)
    true_beta = np.linspace(0.8, 1.3, len(index))
    noise = 0.003 * np.sin(np.arange(len(index)) / 7.0)
    y = pd.Series(0.2 + true_beta * x.to_numpy() + noise, index=index)

    result = KalmanDynamicRegression(delta=1e-3, observation_variance=1e-4).fit(y, x)

    assert abs(result.beta.iloc[-1] - true_beta[-1]) < 0.08
    assert (result.innovation_variance > 0.0).all()


def test_rolling_zscore_does_not_use_future_values() -> None:
    values = pd.Series(np.sin(np.arange(200) / 8.0))
    changed = values.copy()
    changed.iloc[150:] += 100.0

    original_zscore = rolling_zscore(values, 30)
    changed_zscore = rolling_zscore(changed, 30)

    pd.testing.assert_series_equal(original_zscore.iloc[:150], changed_zscore.iloc[:150])


def test_position_state_machine_and_stop_cooldown() -> None:
    zscore = pd.Series([0.0, -2.1, -2.5, -0.4, 2.2, 4.2, 3.0, 1.0, 2.2, 0.3])

    positions = generate_positions(zscore, entry_z=2.0, exit_z=0.5, stop_z=4.0)

    assert positions.tolist() == [0, 1, 1, 0, -1, 0, 0, 0, -1, 0]

