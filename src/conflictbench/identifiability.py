"""Controlled acquisition-value simulations for pre-query identifiability tests.

The simulator is intentionally separate from the historical score-conflict
generator. It gives every example one useful candidate modality and one harmful
candidate modality. A router sees only a noisy observable cue that names the
useful candidate; the oracle reference sees the hidden useful-candidate field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ControlledValueSimulation:
    """One fixed full-information simulation used to evaluate acquisition rules."""

    candidate_values: np.ndarray
    observable_candidate: np.ndarray
    helpful_candidate: np.ndarray
    candidate_costs: np.ndarray
    cue_strength: float

    @property
    def samples(self) -> int:
        return int(self.candidate_values.shape[0])


@dataclass(frozen=True)
class AcquisitionActions:
    """Exact-budget candidate choices, with oracle access made explicit."""

    query_mask: np.ndarray
    chosen_candidate: np.ndarray
    uses_hidden_outcome: bool


@dataclass(frozen=True)
class CandidateValueCalibration:
    """Calibration-only lower estimates of each observable candidate's value."""

    lower_expected_gain: np.ndarray
    samples_per_candidate: np.ndarray


def generate_controlled_value_simulation(
    *,
    seed: int,
    samples: int,
    oracle_headroom: float,
    cue_strength: float,
    candidate_costs: tuple[float, float] = (0.0, 0.0),
) -> ControlledValueSimulation:
    """Generate balanced candidate values with a controllable observable cue.

    ``cue_strength`` is the probability-scale correlation between the observed
    candidate cue and the hidden useful candidate: zero yields a chance-level
    cue and one yields a perfect cue. The oracle headroom is the absolute value
    of the beneficial candidate's loss reduction.
    """

    if samples <= 0 or samples % 2:
        raise ValueError("samples must be a positive even integer")
    if oracle_headroom <= 0:
        raise ValueError("oracle_headroom must be positive")
    if not 0.0 <= cue_strength <= 1.0:
        raise ValueError("cue_strength must be between 0 and 1")
    costs = np.asarray(candidate_costs, dtype=np.float64)
    if costs.shape != (2,) or (costs < 0).any():
        raise ValueError("candidate costs must be two non-negative values")

    rng = np.random.default_rng(seed)
    helpful = np.repeat(np.asarray([0, 1], dtype=np.int64), samples // 2)
    rng.shuffle(helpful)
    values = np.full((samples, 2), -float(oracle_headroom), dtype=np.float64)
    values[np.arange(samples), helpful] = float(oracle_headroom)

    matches = np.zeros(samples, dtype=bool)
    match_count = int(round(((1.0 + cue_strength) / 2.0) * samples))
    matches[rng.choice(samples, size=match_count, replace=False)] = True
    observed = np.where(matches, helpful, 1 - helpful).astype(np.int64)
    return ControlledValueSimulation(values, observed, helpful, costs, cue_strength)


def _exact_query_mask(samples: int, budget: float, seed: int) -> np.ndarray:
    if not 0.0 <= budget <= 1.0:
        raise ValueError("budget must be between 0 and 1")
    count = int(round(samples * budget))
    if not np.isclose(count / samples, budget, atol=1e-12):
        raise ValueError("budget must correspond to an exact number of samples")
    rng = np.random.default_rng(seed)
    mask = np.zeros(samples, dtype=bool)
    mask[rng.choice(samples, size=count, replace=False)] = True
    return mask


def candidate_prior_actions(
    simulation: ControlledValueSimulation, *, budget: float, seed: int
) -> AcquisitionActions:
    """Query a fixed candidate on an exact random subset of examples."""

    return AcquisitionActions(
        _exact_query_mask(simulation.samples, budget, seed),
        np.zeros(simulation.samples, dtype=np.int64),
        False,
    )


def random_actions(simulation: ControlledValueSimulation, *, budget: float, seed: int) -> AcquisitionActions:
    """Choose a candidate uniformly at random on the same exact-budget subset."""

    rng = np.random.default_rng(seed + 1)
    return AcquisitionActions(
        _exact_query_mask(simulation.samples, budget, seed),
        rng.integers(0, 2, size=simulation.samples, dtype=np.int64),
        False,
    )


def cue_policy_actions(simulation: ControlledValueSimulation, *, budget: float, seed: int) -> AcquisitionActions:
    """Choose the candidate named by the permitted observable cue."""

    return AcquisitionActions(
        _exact_query_mask(simulation.samples, budget, seed),
        simulation.observable_candidate.copy(),
        False,
    )


def oracle_actions(simulation: ControlledValueSimulation, *, budget: float, seed: int) -> AcquisitionActions:
    """Select the hidden useful candidate; this is an explicit upper reference."""

    return AcquisitionActions(
        _exact_query_mask(simulation.samples, budget, seed),
        simulation.helpful_candidate.copy(),
        True,
    )


def risk_aware_actions(simulation: ControlledValueSimulation) -> AcquisitionActions:
    """Acquire only when calibrated observable value exceeds candidate cost."""

    expected_raw_gain = simulation.cue_strength * abs(simulation.candidate_values[0, 0])
    choices = simulation.observable_candidate.copy()
    query_mask = expected_raw_gain > simulation.candidate_costs[choices]
    return AcquisitionActions(query_mask, choices, False)


def calibrate_candidate_values(
    calibration: ControlledValueSimulation, *, confidence_z: float = 1.96
) -> CandidateValueCalibration:
    """Estimate conservative candidate values using a disjoint calibration simulation."""

    if confidence_z <= 0:
        raise ValueError("confidence_z must be positive")
    lower = np.empty(2, dtype=np.float64)
    counts = np.empty(2, dtype=np.int64)
    indices = np.arange(calibration.samples)
    for candidate in range(2):
        mask = calibration.observable_candidate == candidate
        values = calibration.candidate_values[indices[mask], candidate]
        if values.size < 2:
            raise ValueError("calibration requires at least two samples per candidate")
        counts[candidate] = values.size
        lower[candidate] = values.mean() - confidence_z * values.std(ddof=1) / np.sqrt(values.size)
    return CandidateValueCalibration(lower, counts)


def calibrated_risk_aware_actions(
    simulation: ControlledValueSimulation, calibration: CandidateValueCalibration
) -> AcquisitionActions:
    """Acquire only when a disjoint calibration lower estimate clears source cost."""

    if calibration.lower_expected_gain.shape != (2,) or calibration.samples_per_candidate.shape != (2,):
        raise ValueError("calibration must provide two candidate estimates")
    choices = simulation.observable_candidate.copy()
    query_mask = calibration.lower_expected_gain[choices] > simulation.candidate_costs[choices]
    return AcquisitionActions(query_mask, choices, False)


def evaluate_gain(simulation: ControlledValueSimulation, actions: AcquisitionActions) -> dict[str, float | bool]:
    """Return acquisition value and realized request rate without hiding oracle access."""

    if actions.query_mask.shape != (simulation.samples,):
        raise ValueError("query mask shape does not match simulation")
    if actions.chosen_candidate.shape != (simulation.samples,):
        raise ValueError("candidate choices shape does not match simulation")
    if not np.isin(actions.chosen_candidate, [0, 1]).all():
        raise ValueError("candidate choices must be 0 or 1")
    gain = np.zeros(simulation.samples, dtype=np.float64)
    indices = np.flatnonzero(actions.query_mask)
    gain[indices] = (
        simulation.candidate_values[indices, actions.chosen_candidate[indices]]
        - simulation.candidate_costs[actions.chosen_candidate[indices]]
    )
    return {
        "mean_gain": float(gain.mean()),
        "query_rate": float(actions.query_mask.mean()),
        "abstention_rate": float((~actions.query_mask).mean()),
        "uses_hidden_outcome": actions.uses_hidden_outcome,
    }


def evaluate_source_recovery(
    simulation: ControlledValueSimulation, actions: AcquisitionActions
) -> dict[str, dict[str, int | float | None]]:
    """Summarize held-out post-query usefulness separately for each source."""

    if actions.query_mask.shape != (simulation.samples,) or actions.chosen_candidate.shape != (simulation.samples,):
        raise ValueError("action shapes do not match simulation")
    summary: dict[str, dict[str, int | float | None]] = {}
    indices = np.arange(simulation.samples)
    for candidate in range(2):
        queried = actions.query_mask & (actions.chosen_candidate == candidate)
        values = simulation.candidate_values[indices[queried], candidate]
        helpful_count = int(np.count_nonzero(values > 0))
        summary[str(candidate)] = {
            "query_count": int(values.size),
            "post_query_helpful_count": helpful_count,
            "post_query_helpful_rate": float(helpful_count / values.size) if values.size else None,
        }
    return summary
