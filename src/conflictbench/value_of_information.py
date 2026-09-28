"""Leakage-safe helpers for value-of-information experiments."""

from __future__ import annotations

import numpy as np


def masked_observations(
    observations: np.ndarray,
    observed_mask: np.ndarray,
    *,
    fill_value: float = 0.0,
    append_mask: bool = True,
) -> np.ndarray:
    """Return features that cannot depend on values hidden by ``observed_mask``.

    Inputs use ``[example, modality]`` layout. By default the returned feature
    vector appends the mask so that equal fill values remain distinguishable
    from genuinely observed values.
    """

    values = np.asarray(observations)
    mask = np.asarray(observed_mask)
    if values.ndim != 2:
        raise ValueError("observations must have shape [n_examples, n_modalities]")
    if mask.shape != values.shape or mask.dtype.kind != "b":
        raise ValueError("observed_mask must be a boolean array matching observations")
    if not np.isfinite(values[mask]).all():
        raise ValueError("observed values must be finite")
    if not np.isfinite(fill_value):
        raise ValueError("fill_value must be finite")

    masked = np.where(mask, values, fill_value)
    if append_mask:
        return np.concatenate((masked, mask.astype(masked.dtype)), axis=1)
    return masked


def realized_acquisition_value(
    loss_before: np.ndarray,
    loss_after: np.ndarray,
    observed_mask: np.ndarray,
) -> np.ndarray:
    """Measure per-candidate loss reduction before applying any query cost.

    Entries for modalities that are already observed are ``NaN`` because they
    are not feasible acquisition actions.
    """

    before = np.asarray(loss_before, dtype=float)
    after = np.asarray(loss_after, dtype=float)
    mask = np.asarray(observed_mask)
    if before.ndim != 1 or after.ndim != 2 or after.shape[0] != before.shape[0]:
        raise ValueError("loss arrays must have shapes [n_examples] and [n_examples, n_modalities]")
    if mask.shape != after.shape or mask.dtype.kind != "b":
        raise ValueError("observed_mask must be a boolean array matching loss_after")
    if not np.isfinite(before).all() or not np.isfinite(after[~mask]).all():
        raise ValueError("feasible loss values must be finite")

    value = before[:, None] - after
    return np.where(mask, np.nan, value)


def select_exact_budget(
    scores: np.ndarray,
    observed_mask: np.ndarray,
    *,
    query_count: int,
) -> np.ndarray:
    """Select exactly ``query_count`` example-modality actions.

    At most one missing modality is selected for each example. Ties prefer the
    lower example index and then the lower modality index.
    """

    candidate_scores = np.asarray(scores, dtype=float)
    mask = np.asarray(observed_mask)
    if candidate_scores.ndim != 2:
        raise ValueError("scores must have shape [n_examples, n_modalities]")
    if mask.shape != candidate_scores.shape or mask.dtype.kind != "b":
        raise ValueError("observed_mask must be a boolean array matching scores")
    if not np.isfinite(candidate_scores[~mask]).all():
        raise ValueError("scores for missing modalities must be finite")
    if isinstance(query_count, (bool, np.bool_)) or int(query_count) != query_count:
        raise ValueError("query_count must be an integer")
    count = int(query_count)
    if count < 0:
        raise ValueError("query_count cannot be negative")

    selectable = ~mask
    eligible = np.flatnonzero(np.any(selectable, axis=1))
    if count > len(eligible):
        raise ValueError("query_count exceeds the number of eligible examples")

    selected = np.zeros_like(mask, dtype=bool)
    if count == 0:
        return selected

    masked_scores = np.where(selectable, candidate_scores, -np.inf)
    best_modality = np.argmax(masked_scores, axis=1)
    best_score = masked_scores[np.arange(len(masked_scores)), best_modality]
    order = np.lexsort((eligible, -best_score[eligible]))
    selected_examples = eligible[order[:count]]
    selected[selected_examples, best_modality[selected_examples]] = True
    return selected


def random_matched_selection(
    reference_selection: np.ndarray,
    observed_mask: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Draw a random feasible policy with the reference policy's query count."""

    reference = np.asarray(reference_selection)
    mask = np.asarray(observed_mask)
    if reference.ndim != 2 or reference.dtype.kind != "b":
        raise ValueError("reference_selection must be a two-dimensional boolean array")
    if mask.shape != reference.shape or mask.dtype.kind != "b":
        raise ValueError("observed_mask must be a boolean array matching reference_selection")
    if np.any(reference & mask) or np.any(reference.sum(axis=1) > 1):
        raise ValueError("reference_selection must select at most one missing modality per example")

    query_count = int(reference.sum())
    selectable = ~mask
    eligible = np.flatnonzero(np.any(selectable, axis=1))
    if query_count > len(eligible):
        raise ValueError("reference query count exceeds the number of eligible examples")

    rng = np.random.default_rng(seed)
    selected = np.zeros_like(reference, dtype=bool)
    if query_count == 0:
        return selected
    selected_examples = rng.choice(eligible, size=query_count, replace=False)
    for example in selected_examples.tolist():
        modalities = np.flatnonzero(selectable[example])
        selected[example, int(rng.choice(modalities))] = True
    return selected


def source_group_split(
    group_ids: np.ndarray,
    *,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Create deterministic train/dev/test indices without splitting a group."""

    groups = np.asarray(group_ids)
    weights = np.asarray(fractions, dtype=float)
    if groups.ndim != 1 or len(groups) == 0:
        raise ValueError("group_ids must be a non-empty one-dimensional array")
    if weights.shape != (3,) or not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("fractions must contain three finite non-negative values")
    if not np.isclose(float(weights.sum()), 1.0):
        raise ValueError("fractions must sum to one")

    unique_groups = np.unique(groups)
    positive = weights > 0.0
    if len(unique_groups) < int(positive.sum()):
        raise ValueError("too few groups to populate every positive-fraction split")

    counts = positive.astype(np.int64)
    targets = weights * len(unique_groups)
    for _ in range(len(unique_groups) - int(counts.sum())):
        deficit = np.where(positive, targets - counts, -np.inf)
        counts[int(np.argmax(deficit))] += 1

    rng = np.random.default_rng(seed)
    shuffled = unique_groups[rng.permutation(len(unique_groups))]
    first = int(counts[0])
    second = first + int(counts[1])
    assignments = {
        "train": shuffled[:first],
        "dev": shuffled[first:second],
        "test": shuffled[second:],
    }
    return {
        name: np.flatnonzero(np.isin(groups, assigned_groups))
        for name, assigned_groups in assignments.items()
    }


def grouped_bootstrap_indices(
    group_ids: np.ndarray,
    *,
    n_resamples: int,
    seed: int = 0,
) -> tuple[np.ndarray, ...]:
    """Return paired bootstrap indices formed by sampling whole groups."""

    groups = np.asarray(group_ids)
    if groups.ndim != 1 or len(groups) == 0:
        raise ValueError("group_ids must be a non-empty one-dimensional array")
    if isinstance(n_resamples, (bool, np.bool_)) or int(n_resamples) != n_resamples:
        raise ValueError("n_resamples must be an integer")
    count = int(n_resamples)
    if count <= 0:
        raise ValueError("n_resamples must be positive")

    unique_groups = np.unique(groups)
    members = tuple(np.flatnonzero(groups == group) for group in unique_groups)
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    for _ in range(count):
        selected_groups = rng.integers(0, len(unique_groups), size=len(unique_groups))
        samples.append(np.concatenate([members[index] for index in selected_groups]))
    return tuple(samples)
