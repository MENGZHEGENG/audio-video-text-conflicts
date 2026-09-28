import numpy as np

from conflictbench.controls import select_control_thresholds
from conflictbench.core import Dataset


def test_calibration_uses_validation_only_and_returns_grid_member():
    data = Dataset(
        x=np.array([[0.2, 0.2, -1.0], [-0.2, -0.2, 1.0]]),
        y=np.array([1, 0]), ambiguous=np.array([False, False]),
        mechanism=np.array(['real', 'real']),
    )
    chosen, records = select_control_thresholds(data, [0.1, 0.35])
    assert chosen['active_diagnostic'] == 0.1
    assert chosen['pair_selective'] == 0.1
    assert len(records) == 8
    assert set(chosen.values()) <= {0.1, 0.35}
