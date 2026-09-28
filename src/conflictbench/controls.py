"""Ablations sharing the active policy's initial pair and request cost."""

import numpy as np

from .core import ActionOutput, evaluate_actions, policy_actions


def protocol_controls(x, threshold=0.35):
    active = policy_actions('active_diagnostic', x, threshold=threshold)
    pair_action = active.action.copy()
    pair_action[active.query] = 2
    strong = np.abs(x) >= threshold
    positive = np.sum(strong & (x >= 0), axis=1) >= 2
    negative = np.sum(strong & (x < 0), axis=1) >= 2
    full_action = np.full(len(x), 2, dtype=np.int64)
    full_action[positive] = 1
    full_action[negative] = 0
    majority = policy_actions('majority', x).action
    forced = active.action.copy()
    forced[active.query] = majority[active.query]
    return {
        'active_diagnostic': active,
        'pair_selective': ActionOutput(pair_action, np.zeros(len(x), dtype=bool)),
        'always_query_selective': ActionOutput(full_action, np.ones(len(x), dtype=bool)),
        'query_majority': ActionOutput(forced, active.query.copy()),
    }


def select_control_thresholds(validation, thresholds):
    """Maximize validation utility per control; ties favor the smaller threshold."""
    grid = sorted(set(float(value) for value in thresholds))
    if not grid or any(not np.isfinite(value) or value <= 0 for value in grid):
        raise ValueError('thresholds must be finite and positive')
    best, selected, records = {}, {}, []
    for threshold in grid:
        for method, output in protocol_controls(validation.x, threshold).items():
            metrics = evaluate_actions(output, validation, initial_threshold=threshold)['metrics']
            records.append(dict(method=method, threshold=threshold, metrics=metrics))
            if method not in best or metrics['utility'] > best[method]:
                best[method] = metrics['utility']
                selected[method] = threshold
    return selected, records
