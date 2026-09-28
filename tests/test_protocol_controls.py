import numpy as np

from conflictbench.core import policy_actions
from conflictbench.controls import protocol_controls


def test_information_schedule_and_shared_decision():
    x = np.array([[1., 1., -3.], [1., -1., 0.1], [0.1, 1., 1.], [-1., -1., 3.]])
    outputs = protocol_controls(x)
    active = policy_actions('active_diagnostic', x)
    np.testing.assert_array_equal(outputs['always_query_selective'].action, active.action)
    assert outputs['always_query_selective'].query.all()
    assert not outputs['pair_selective'].query.any()
    np.testing.assert_array_equal(outputs['query_majority'].query, active.query)
    changed = x.copy()
    changed[:, 2] *= -10
    np.testing.assert_array_equal(protocol_controls(changed)['pair_selective'].action,
                                  outputs['pair_selective'].action)
