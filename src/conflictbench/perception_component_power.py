"""Training-derived, component-level pre-study power approximation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import platform
import re
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np


class PowerValidationError(ValueError):
    """Raised when a power input or result violates its frozen contract."""


CONFIG_SCHEMA = "conflictbench.perception-component-power-config.v2"
SUMMARY_SCHEMA = "conflictbench.perception-component-event-summary.v2"
OUTPUT_SCHEMA = "conflictbench.perception-component-power-report.v2"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMPONENT_ID_RE = re.compile(r"[0-9a-f]{24}\Z")
_CONFIG_KEYS = {"schema", "analysis", "scope", "simulation", "paired_analysis"}
_SUMMARY_KEYS = {"schema", "derivation", "counts", "exclusions", "components"}
_EVENT_KEYS = {
    "both_correct",
    "baseline_error_only",
    "proposed_error_only",
    "both_error",
}
_EVENT_ORDER = (
    "both_correct",
    "baseline_error_only",
    "proposed_error_only",
    "both_error",
)
_OUTPUT_KEYS = {
    "schema",
    "analysis",
    "configuration",
    "component_event_summary",
    "implementation_files",
    "input_digests",
    "runtime",
    "design",
    "simulation",
    "payload_sha256",
}
_MINIMUM_COMPONENTS = 20


def _duplicate_safe_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PowerValidationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise PowerValidationError(f"non-finite JSON number is not allowed: {value}")


def _strict_json(data: bytes, field: str) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PowerValidationError(f"{field} is not strict UTF-8 JSON") from exc


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PowerValidationError(f"{field} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        raise PowerValidationError(
            f"{field} keys differ; missing={missing}, unknown={unknown}"
        )


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PowerValidationError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PowerValidationError(f"{field} must be a nonnegative integer")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PowerValidationError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PowerValidationError(f"{field} must be finite")
    return result


def _plain_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PowerValidationError(f"{field} must be a nonempty trimmed string")
    return value


def _sha256_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise PowerValidationError(f"{field} must be a lowercase SHA-256 value")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_configuration(value: Any) -> dict[str, Any]:
    config = _mapping(value, "configuration")
    _exact_keys(config, _CONFIG_KEYS, "configuration")
    if config["schema"] != CONFIG_SCHEMA:
        raise PowerValidationError("configuration schema is unexpected")
    if config["analysis"] != "paired_component_sign_flip_power_approximation":
        raise PowerValidationError("configuration analysis is unexpected")

    scope = _mapping(config["scope"], "configuration.scope")
    _exact_keys(
        scope,
        {
            "dataset",
            "derivation_partition",
            "derivation_split_id",
            "evaluation_outcomes_access",
            "policy_id",
            "acquisition_budget",
        },
        "configuration.scope",
    )
    expected_scope = {
        "dataset": "Perception Test",
        "derivation_partition": "training",
        "derivation_split_id": "perception_test_train_v1",
        "evaluation_outcomes_access": "forbidden",
        "policy_id": "value_of_evidence_policy_v1",
        "acquisition_budget": 0.50,
    }
    if dict(scope) != expected_scope:
        raise PowerValidationError(
            "configuration scope must lock training-only inputs and a 0.50 budget"
        )

    simulation = _mapping(config["simulation"], "configuration.simulation")
    _exact_keys(
        simulation,
        {
            "repetitions",
            "randomization_draws",
            "seed",
            "minimum_detectable_error_reduction",
            "two_sided_alpha",
            "target_power",
            "decision_confidence_level",
            "null_calibration_max_rejection_rate",
        },
        "configuration.simulation",
    )
    if _positive_int(simulation["repetitions"], "simulation.repetitions") != 10000:
        raise PowerValidationError("simulation.repetitions must equal 10000")
    if (
        _positive_int(
            simulation["randomization_draws"], "simulation.randomization_draws"
        )
        != 999
    ):
        raise PowerValidationError("simulation.randomization_draws must equal 999")
    _nonnegative_int(simulation["seed"], "simulation.seed")
    expected_numbers = {
        "minimum_detectable_error_reduction": 0.01,
        "two_sided_alpha": 0.05,
        "target_power": 0.80,
        "decision_confidence_level": 0.95,
        "null_calibration_max_rejection_rate": 0.06,
    }
    for name, expected in expected_numbers.items():
        if _finite_number(simulation[name], f"simulation.{name}") != expected:
            raise PowerValidationError(f"simulation.{name} must equal {expected}")

    paired = _mapping(config["paired_analysis"], "configuration.paired_analysis")
    _exact_keys(
        paired,
        {
            "estimand",
            "cluster_unit",
            "test",
            "reference_distribution",
            "outcome_model",
            "component_size_model",
            "within_component_dependence",
            "finite_sample_requirement",
            "final_analysis_alignment",
        },
        "configuration.paired_analysis",
    )
    expected_paired = {
        "estimand": "event_weighted_absolute_error_reduction",
        "cluster_unit": "target_donor_connected_component",
        "test": "monte_carlo_component_sign_flip",
        "reference_distribution": "component_rademacher_sign_flips",
        "outcome_model": (
            "rademacher_orient_whole_observed_component_scores_to_global_mde"
        ),
        "component_size_model": "fixed_observed",
        "within_component_dependence": "preserve_whole_observed_component_aggregates",
        "finite_sample_requirement": "minimum_20_components",
        "final_analysis_alignment": "pre_study_approximation_only",
    }
    if dict(paired) != expected_paired:
        raise PowerValidationError("configuration paired analysis is unexpected")
    return json.loads(json.dumps(config, ensure_ascii=False, sort_keys=True))


def load_power_configuration(
    path: pathlib.Path, expected_sha256: str
) -> dict[str, Any]:
    """Load strict configuration only after its external digest matches."""

    expected = _sha256_text(expected_sha256, "expected configuration SHA-256")
    try:
        data = pathlib.Path(path).read_bytes()
    except OSError as exc:
        raise PowerValidationError("configuration could not be read") from exc
    if hashlib.sha256(data).hexdigest() != expected:
        raise PowerValidationError("configuration SHA-256 does not match")
    return _validate_configuration(_strict_json(data, "configuration"))


def _validate_component_event_summary(value: Any) -> dict[str, Any]:
    summary = _mapping(value, "component event summary")
    _exact_keys(summary, _SUMMARY_KEYS, "component event summary")
    if summary["schema"] != SUMMARY_SCHEMA:
        raise PowerValidationError("component event summary schema is unexpected")

    derivation = _mapping(summary["derivation"], "summary.derivation")
    derivation_keys = {
        "dataset",
        "partition",
        "split_id",
        "frozen_before_evaluation",
        "evaluation_outcomes_access",
        "event_unit",
        "paired_outcomes",
        "policy_id",
        "acquisition_budget",
        "source_data_sha256",
        "policy_configuration_sha256",
        "summary_generator_sha256",
    }
    _exact_keys(derivation, derivation_keys, "summary.derivation")
    if derivation["frozen_before_evaluation"] is not True:
        raise PowerValidationError(
            "component event summary must be frozen before evaluation"
        )
    expected_core = {
        "dataset": "Perception Test",
        "partition": "training",
        "evaluation_outcomes_access": "forbidden",
        "event_unit": "target_question",
        "paired_outcomes": ["baseline_error", "proposed_error"],
    }
    if any(derivation[name] != expected for name, expected in expected_core.items()):
        raise PowerValidationError(
            "component event summary must be frozen and training-derived"
        )
    split_id = _plain_string(derivation["split_id"], "summary.derivation.split_id")
    _plain_string(derivation["policy_id"], "summary.derivation.policy_id")
    if (
        _finite_number(
            derivation["acquisition_budget"], "summary.derivation.acquisition_budget"
        )
        != 0.50
    ):
        raise PowerValidationError("summary acquisition budget must equal 0.50")
    for digest_name in (
        "source_data_sha256",
        "policy_configuration_sha256",
        "summary_generator_sha256",
    ):
        _sha256_text(derivation[digest_name], f"summary.derivation.{digest_name}")

    counts = _mapping(summary["counts"], "summary.counts")
    _exact_keys(
        counts,
        {
            "eligible_target_count",
            "unique_source_count",
            "connected_component_count",
            "exclusion_count",
        },
        "summary.counts",
    )
    eligible_target_count = _positive_int(
        counts["eligible_target_count"], "summary eligible_target_count"
    )
    unique_source_count = _positive_int(
        counts["unique_source_count"], "summary unique_source_count"
    )
    component_count = _positive_int(
        counts["connected_component_count"], "summary connected_component_count"
    )
    if component_count < _MINIMUM_COMPONENTS:
        raise PowerValidationError(
            f"at least {_MINIMUM_COMPONENTS} connected components are required"
        )
    exclusion_count = _nonnegative_int(
        counts["exclusion_count"], "summary exclusion_count"
    )

    exclusions_value = summary["exclusions"]
    if not isinstance(exclusions_value, list):
        raise PowerValidationError("summary exclusions must be a list")
    exclusions: list[dict[str, str]] = []
    exclusion_ids: set[str] = set()
    for index, raw_exclusion in enumerate(exclusions_value):
        exclusion = _mapping(raw_exclusion, f"summary.exclusions[{index}]")
        _exact_keys(
            exclusion,
            {"record_id", "source_id", "split_id", "reason"},
            f"summary.exclusions[{index}]",
        )
        record_id = _plain_string(
            exclusion["record_id"], f"summary.exclusions[{index}].record_id"
        )
        if record_id in exclusion_ids:
            raise PowerValidationError("exclusion record IDs must be unique")
        exclusion_ids.add(record_id)
        exclusion_split = _plain_string(
            exclusion["split_id"], f"summary.exclusions[{index}].split_id"
        )
        if exclusion_split != split_id:
            raise PowerValidationError(
                "every exclusion must name the training split ID"
            )
        exclusions.append(
            {
                "record_id": record_id,
                "source_id": _plain_string(
                    exclusion["source_id"],
                    f"summary.exclusions[{index}].source_id",
                ),
                "split_id": exclusion_split,
                "reason": _plain_string(
                    exclusion["reason"], f"summary.exclusions[{index}].reason"
                ),
            }
        )
    if exclusions != sorted(exclusions, key=lambda item: item["record_id"]):
        raise PowerValidationError("summary exclusions must be in record-ID order")
    if exclusion_count != len(exclusions):
        raise PowerValidationError("exclusion count differs from exclusion records")

    components_value = summary["components"]
    if not isinstance(components_value, list) or not components_value:
        raise PowerValidationError("summary components must be a nonempty list")
    components: list[dict[str, Any]] = []
    seen_component_ids: set[str] = set()
    seen_source_ids: set[str] = set()
    total_events = 0
    for index, raw_component in enumerate(components_value):
        component = _mapping(raw_component, f"summary.components[{index}]")
        _exact_keys(
            component,
            {"component_id", "split_id", "member_source_ids", "event_counts"},
            f"summary.components[{index}]",
        )
        component_id = component["component_id"]
        if (
            not isinstance(component_id, str)
            or not _COMPONENT_ID_RE.fullmatch(component_id)
            or component_id in seen_component_ids
        ):
            raise PowerValidationError(
                "component IDs must be unique 24-character lowercase hex values"
            )
        seen_component_ids.add(component_id)
        component_split = _plain_string(
            component["split_id"], f"summary.components[{index}].split_id"
        )
        if component_split != split_id:
            raise PowerValidationError(
                "every component must name the training split ID"
            )
        raw_members = component["member_source_ids"]
        if not isinstance(raw_members, list) or not raw_members:
            raise PowerValidationError(
                "every component must list at least one member source ID"
            )
        members = [
            _plain_string(item, f"summary.components[{index}].member_source_ids")
            for item in raw_members
        ]
        if members != sorted(set(members)):
            raise PowerValidationError(
                "component member source IDs must be unique and sorted"
            )
        if seen_source_ids.intersection(members):
            raise PowerValidationError("each source ID must belong to one component")
        seen_source_ids.update(members)

        event_counts = _mapping(
            component["event_counts"], f"summary.components[{index}].event_counts"
        )
        _exact_keys(
            event_counts, _EVENT_KEYS, f"summary.components[{index}].event_counts"
        )
        normalized_counts = {
            name: _nonnegative_int(
                event_counts[name],
                f"summary.components[{index}].event_counts.{name}",
            )
            for name in sorted(_EVENT_KEYS)
        }
        component_total = sum(normalized_counts.values())
        if component_total <= 0:
            raise PowerValidationError(
                "every component must contain at least one event"
            )
        total_events += component_total
        components.append(
            {
                "component_id": component_id,
                "split_id": component_split,
                "member_source_ids": members,
                "event_counts": normalized_counts,
            }
        )
    if components != sorted(components, key=lambda item: item["component_id"]):
        raise PowerValidationError("summary components must be in component-ID order")
    if component_count != len(components):
        raise PowerValidationError("connected component count differs from components")
    if unique_source_count != len(seen_source_ids):
        raise PowerValidationError(
            "unique source count differs from component membership"
        )
    if eligible_target_count != total_events:
        raise PowerValidationError("eligible target count differs from event counts")
    return {
        "schema": SUMMARY_SCHEMA,
        "derivation": dict(derivation),
        "counts": dict(counts),
        "exclusions": exclusions,
        "components": components,
    }


def load_component_event_summary(
    path: pathlib.Path, expected_sha256: str
) -> dict[str, Any]:
    """Load a frozen training summary after checking its external digest."""

    expected = _sha256_text(expected_sha256, "expected summary SHA-256")
    try:
        data = pathlib.Path(path).read_bytes()
    except OSError as exc:
        raise PowerValidationError("component event summary could not be read") from exc
    if hashlib.sha256(data).hexdigest() != expected:
        raise PowerValidationError("component event summary SHA-256 does not match")
    return _validate_component_event_summary(
        _strict_json(data, "component event summary")
    )


def build_power_design(
    summary: Mapping[str, Any], minimum_detectable_effect: float
) -> dict[str, Any]:
    """Orient whole observed component aggregates to reproduce a global MDE."""

    normalized = _validate_component_event_summary(summary)
    mde = _finite_number(minimum_detectable_effect, "minimum detectable effect")
    if not 0.0 < mde < 1.0:
        raise PowerValidationError(
            "minimum detectable effect must be between zero and one"
        )

    total_counts = {name: 0 for name in _EVENT_ORDER}
    raw_components: list[tuple[dict[str, Any], int, int, int]] = []
    for component in normalized["components"]:
        counts = component["event_counts"]
        size = sum(counts.values())
        score = counts["baseline_error_only"] - counts["proposed_error_only"]
        discordant = counts["baseline_error_only"] + counts["proposed_error_only"]
        raw_components.append((component, size, score, discordant))
        for name in _EVENT_ORDER:
            total_counts[name] += counts[name]
    event_count = sum(total_counts.values())
    observed_score = sum(item[2] for item in raw_components)
    observed_effect = observed_score / event_count
    absolute_score_sum = sum(abs(item[2]) for item in raw_components)
    absolute_score_rate = absolute_score_sum / event_count
    if absolute_score_rate + 1e-15 < mde:
        raise PowerValidationError(
            "observed component score heterogeneity cannot support the requested effect"
        )
    directional_bias = mde / absolute_score_rate
    positive_orientation_probability = (1.0 + directional_bias) / 2.0
    components: list[dict[str, Any]] = []
    for component, size, score, discordant in raw_components:
        absolute_score = abs(score)
        if absolute_score > discordant:
            raise PowerValidationError("component score exceeds its discordant count")
        components.append(
            {
                "component_id": component["component_id"],
                "split_id": component["split_id"],
                "member_source_ids": component["member_source_ids"],
                "event_count": size,
                "observed_event_counts": component["event_counts"],
                "observed_error_reduction": score / size,
                "absolute_score": absolute_score,
                "absolute_error_reduction": absolute_score / size,
            }
        )
    expected_effect = directional_bias * absolute_score_sum / event_count
    if not math.isclose(expected_effect, mde, rel_tol=0.0, abs_tol=1e-12):
        raise PowerValidationError(
            "power design does not reproduce the requested effect"
        )
    discordant_count = (
        total_counts["baseline_error_only"] + total_counts["proposed_error_only"]
    )
    return {
        "component_count": len(components),
        "event_count": event_count,
        "observed_event_counts": total_counts,
        "observed_error_reduction": observed_effect,
        "observed_discordance_rate": discordant_count / event_count,
        "observed_absolute_score_rate": absolute_score_rate,
        "directional_orientation_bias": directional_bias,
        "alternative_positive_orientation_probability": (
            positive_orientation_probability
        ),
        "expected_error_reduction": expected_effect,
        "simulation_unit": "whole_component_aggregate",
        "assumptions": [
            "training component profile represents the planned study design",
            "method labels are exchangeable within each component under the null",
            "component orientations are independent across connected components",
            "observed membership sizes and paired-event aggregates stay fixed",
            "the alternative changes only the probability that a whole component favors the proposed policy",
            "no event-level independence or multinomial redraw is assumed",
        ],
        "components": components,
    }


def paired_component_sign_flip_test(
    component_scores: np.ndarray,
    component_sizes: np.ndarray,
    *,
    reference_signs: np.ndarray,
    two_sided_alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a two-sided Monte Carlo sign-flip test to whole component scores."""

    scores = np.asarray(component_scores, dtype=np.float64)
    sizes = np.asarray(component_sizes, dtype=np.float64)
    signs = np.asarray(reference_signs, dtype=np.float64)
    if scores.ndim != 2 or sizes.ndim != 1 or scores.shape[1] != sizes.shape[0]:
        raise PowerValidationError("component score and size shapes are incompatible")
    if scores.shape[0] == 0 or sizes.shape[0] < _MINIMUM_COMPONENTS:
        raise PowerValidationError(
            f"sign-flip analysis requires at least {_MINIMUM_COMPONENTS} components"
        )
    if signs.ndim != 2 or signs.shape[1] != sizes.shape[0] or signs.shape[0] == 0:
        raise PowerValidationError("reference sign shape is incompatible")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(sizes)):
        raise PowerValidationError("component scores and sizes must be finite")
    if np.any(sizes <= 0):
        raise PowerValidationError("component sizes must be positive")
    if not np.all(np.logical_or(signs == -1.0, signs == 1.0)):
        raise PowerValidationError("reference signs must equal minus one or one")
    alpha = _finite_number(two_sided_alpha, "two-sided alpha")
    if not 0.0 < alpha < 1.0:
        raise PowerValidationError("two-sided alpha must be between zero and one")

    event_count = float(np.sum(sizes))
    estimates = np.sum(scores, axis=1) / event_count
    observed_statistics = np.abs(np.sum(scores, axis=1))
    p_values = np.empty(scores.shape[0], dtype=np.float64)
    batch_size = 256
    for start in range(0, scores.shape[0], batch_size):
        stop = min(start + batch_size, scores.shape[0])
        randomization_statistics = np.abs(scores[start:stop] @ signs.T)
        exceedances = np.count_nonzero(
            randomization_statistics >= observed_statistics[start:stop, None] - 1e-12,
            axis=1,
        )
        p_values[start:stop] = (1.0 + exceedances) / (1.0 + signs.shape[0])
    return estimates, p_values, p_values <= alpha


def _wilson_interval(
    successes: int, repetitions: int, confidence_level: float
) -> dict[str, Any]:
    if successes < 0 or successes > repetitions:
        raise PowerValidationError("Wilson successes must be within repetitions")
    confidence = _finite_number(confidence_level, "Wilson confidence level")
    if not 0.0 < confidence < 1.0:
        raise PowerValidationError(
            "Wilson confidence level must be between zero and one"
        )
    estimate = successes / repetitions
    critical = statistics.NormalDist().inv_cdf(0.5 + confidence / 2.0)
    denominator = 1.0 + critical * critical / repetitions
    center = (estimate + critical * critical / (2.0 * repetitions)) / denominator
    half_width = (
        critical
        * math.sqrt(
            estimate * (1.0 - estimate) / repetitions
            + critical * critical / (4.0 * repetitions * repetitions)
        )
        / denominator
    )
    return {
        "method": "wilson_score",
        "confidence_level": confidence,
        "lower": max(0.0, center - half_width),
        "upper": min(1.0, center + half_width),
    }


def wilson_decision(
    *,
    successes: int,
    repetitions: int,
    confidence_level: float,
    threshold: float,
    direction: Literal["lower_at_least", "upper_at_most"],
) -> dict[str, Any]:
    """Make a threshold decision from a two-sided Wilson interval endpoint."""

    success_count = _nonnegative_int(successes, "successes")
    repetition_count = _positive_int(repetitions, "repetitions")
    if success_count > repetition_count:
        raise PowerValidationError("successes cannot exceed repetitions")
    limit = _finite_number(threshold, "decision threshold")
    if not 0.0 <= limit <= 1.0:
        raise PowerValidationError("decision threshold must be between zero and one")
    interval = _wilson_interval(success_count, repetition_count, confidence_level)
    if direction == "lower_at_least":
        passes = interval["lower"] >= limit
        rule = "wilson_lower_bound_gte_threshold"
    elif direction == "upper_at_most":
        passes = interval["upper"] <= limit
        rule = "wilson_upper_bound_lte_threshold"
    else:
        raise PowerValidationError("Wilson decision direction is unexpected")
    return {
        "estimate": success_count / repetition_count,
        "interval": interval,
        "threshold": limit,
        "decision_rule": rule,
        "passes": passes,
    }


def _simulate_power(
    config: Mapping[str, Any], design: Mapping[str, Any]
) -> dict[str, Any]:
    simulation = config["simulation"]
    repetitions = int(simulation["repetitions"])
    randomization_draws = int(simulation["randomization_draws"])
    seed = int(simulation["seed"])
    alpha = float(simulation["two_sided_alpha"])
    confidence = float(simulation["decision_confidence_level"])
    target_power = float(simulation["target_power"])
    calibration_limit = float(simulation["null_calibration_max_rejection_rate"])
    seed_sequences = np.random.SeedSequence(seed).spawn(3)
    null_rng = np.random.Generator(np.random.PCG64(seed_sequences[0]))
    alternative_rng = np.random.Generator(np.random.PCG64(seed_sequences[1]))
    reference_rng = np.random.Generator(np.random.PCG64(seed_sequences[2]))

    component_sizes = np.asarray(
        [component["event_count"] for component in design["components"]],
        dtype=np.float64,
    )
    absolute_scores = np.asarray(
        [component["absolute_score"] for component in design["components"]],
        dtype=np.float64,
    )
    positive_probability = float(design["alternative_positive_orientation_probability"])
    reference_signs = reference_rng.choice(
        np.asarray([-1.0, 1.0]),
        size=(randomization_draws, component_sizes.shape[0]),
        replace=True,
    )
    null_signs = null_rng.choice(
        np.asarray([-1.0, 1.0]),
        size=(repetitions, component_sizes.shape[0]),
        replace=True,
    )
    alternative_signs = np.where(
        alternative_rng.random(size=(repetitions, component_sizes.shape[0]))
        < positive_probability,
        1.0,
        -1.0,
    )
    null_scores = null_signs * absolute_scores[None, :]
    alternative_scores = alternative_signs * absolute_scores[None, :]
    null_estimates, _, null_rejected = paired_component_sign_flip_test(
        null_scores,
        component_sizes,
        reference_signs=reference_signs,
        two_sided_alpha=alpha,
    )
    alternative_estimates, _, alternative_rejected = paired_component_sign_flip_test(
        alternative_scores,
        component_sizes,
        reference_signs=reference_signs,
        two_sided_alpha=alpha,
    )
    null_rejection_count = int(np.count_nonzero(null_rejected))
    power_rejection_count = int(np.count_nonzero(alternative_rejected))
    null_decision = wilson_decision(
        successes=null_rejection_count,
        repetitions=repetitions,
        confidence_level=confidence,
        threshold=calibration_limit,
        direction="upper_at_most",
    )
    power_decision = wilson_decision(
        successes=power_rejection_count,
        repetitions=repetitions,
        confidence_level=confidence,
        threshold=target_power,
        direction="lower_at_least",
    )
    null_rate = null_rejection_count / repetitions
    power_rate = power_rejection_count / repetitions
    null_calibration = {
        "rejection_count": null_rejection_count,
        "estimated_type_i_error": null_rate,
        "monte_carlo_standard_error": math.sqrt(
            null_rate * (1.0 - null_rate) / repetitions
        ),
        **null_decision,
    }
    power_result = {
        "rejection_count": power_rejection_count,
        "estimated_power": power_rate,
        "monte_carlo_standard_error": math.sqrt(
            power_rate * (1.0 - power_rate) / repetitions
        ),
        **power_decision,
        "meets_target_power": power_decision["passes"],
    }
    if not null_decision["passes"]:
        status = "null_calibration_failed"
    elif power_decision["passes"]:
        status = "adequately_powered"
    else:
        status = "underpowered"
    return {
        "repetitions": repetitions,
        "randomization_draws": randomization_draws,
        "seed": seed,
        "minimum_detectable_error_reduction": float(
            simulation["minimum_detectable_error_reduction"]
        ),
        "two_sided_alpha": alpha,
        "decision_confidence_level": confidence,
        "mean_null_error_reduction": float(np.mean(null_estimates)),
        "mean_simulated_error_reduction": float(np.mean(alternative_estimates)),
        "null_calibration": null_calibration,
        "power": power_result,
        "status": status,
    }


def _analysis_description() -> dict[str, Any]:
    return {
        "role": "pre_study_power_approximation",
        "paired_outcome": "baseline_error_minus_proposed_error",
        "estimand": "event_weighted_absolute_error_reduction",
        "cluster_unit": "target_donor_connected_component",
        "test": "monte_carlo_component_sign_flip",
        "reference_distribution": "component_rademacher_sign_flips",
        "final_analysis_note": (
            "This is a training-derived pre-study approximation. Its sign-exchangeability "
            "and common-effect assumptions must not be described as the final analysis."
        ),
    }


def _runtime_record() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def _implementation_inventory(
    source_paths: Mapping[str, pathlib.Path],
) -> list[dict[str, Any]]:
    if not isinstance(source_paths, Mapping) or not source_paths:
        raise PowerValidationError("at least one implementation source is required")
    files: list[dict[str, Any]] = []
    filenames: set[str] = set()
    for role in sorted(source_paths):
        if (
            not isinstance(role, str)
            or not role
            or pathlib.PurePosixPath(role).name != role
        ):
            raise PowerValidationError("implementation roles must be plain filenames")
        path = pathlib.Path(source_paths[role])
        if path.name in filenames:
            raise PowerValidationError("implementation filenames must be unique")
        filenames.add(path.name)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise PowerValidationError(
                "implementation source could not be read"
            ) from exc
        files.append(
            {
                "role": role,
                "filename": path.name,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return files


def _training_provenance_digests(summary: Mapping[str, Any]) -> dict[str, str]:
    derivation = summary["derivation"]
    membership = [
        {
            "component_id": component["component_id"],
            "split_id": component["split_id"],
            "member_source_ids": component["member_source_ids"],
        }
        for component in summary["components"]
    ]
    source_split_ids = sorted(
        [
            {"source_id": source_id, "split_id": component["split_id"]}
            for component in summary["components"]
            for source_id in component["member_source_ids"]
        ]
        + [
            {
                "source_id": exclusion["source_id"],
                "split_id": exclusion["split_id"],
            }
            for exclusion in summary["exclusions"]
        ],
        key=lambda item: (item["split_id"], item["source_id"]),
    )
    return {
        "source_data_sha256": derivation["source_data_sha256"],
        "policy_configuration_sha256": derivation["policy_configuration_sha256"],
        "summary_generator_sha256": derivation["summary_generator_sha256"],
        "component_membership_sha256": _canonical_sha256(membership),
        "exclusions_sha256": _canonical_sha256(summary["exclusions"]),
        "source_split_ids_sha256": _canonical_sha256(source_split_ids),
    }


def run_power_simulation(
    *,
    config_path: pathlib.Path,
    expected_config_sha256: str,
    summary_path: pathlib.Path,
    expected_summary_sha256: str,
    source_paths: Mapping[str, pathlib.Path],
) -> dict[str, Any]:
    """Run the deterministic approximation from externally hash-bound inputs."""

    config = load_power_configuration(config_path, expected_config_sha256)
    summary = load_component_event_summary(summary_path, expected_summary_sha256)
    scope = config["scope"]
    derivation = summary["derivation"]
    cross_checks = {
        "dataset": (scope["dataset"], derivation["dataset"]),
        "partition": (scope["derivation_partition"], derivation["partition"]),
        "split ID": (scope["derivation_split_id"], derivation["split_id"]),
        "evaluation access": (
            scope["evaluation_outcomes_access"],
            derivation["evaluation_outcomes_access"],
        ),
        "policy ID": (scope["policy_id"], derivation["policy_id"]),
        "acquisition budget": (
            scope["acquisition_budget"],
            derivation["acquisition_budget"],
        ),
    }
    for field, (expected, observed) in cross_checks.items():
        if expected != observed:
            raise PowerValidationError(f"configuration and summary {field} differ")
    config_digest = _sha256_text(
        expected_config_sha256, "expected configuration SHA-256"
    )
    summary_digest = _sha256_text(expected_summary_sha256, "expected summary SHA-256")
    implementation_files = _implementation_inventory(source_paths)
    implementation_digest = _canonical_sha256(implementation_files)
    design = build_power_design(
        summary,
        minimum_detectable_effect=float(
            config["simulation"]["minimum_detectable_error_reduction"]
        ),
    )
    digests = {
        "configuration_sha256": config_digest,
        "configuration_value_sha256": _canonical_sha256(config),
        "component_event_summary_sha256": summary_digest,
        "component_event_summary_value_sha256": _canonical_sha256(summary),
        "implementation_bundle_sha256": implementation_digest,
        **_training_provenance_digests(summary),
    }
    digests["run_input_sha256"] = _canonical_sha256(digests)
    output: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "analysis": _analysis_description(),
        "configuration": config,
        "component_event_summary": summary,
        "implementation_files": implementation_files,
        "input_digests": digests,
        "runtime": _runtime_record(),
        "design": design,
        "simulation": _simulate_power(config, design),
    }
    output["payload_sha256"] = _canonical_sha256(output)
    validate_power_output(
        output,
        expected_config_sha256=config_digest,
        expected_summary_sha256=summary_digest,
        expected_implementation_bundle_sha256=implementation_digest,
    )
    return output


def _validate_implementation_files(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise PowerValidationError("implementation_files must be a nonempty list")
    normalized: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    seen_filenames: set[str] = set()
    for index, raw_file in enumerate(value):
        file_record = _mapping(raw_file, f"implementation_files[{index}]")
        _exact_keys(
            file_record,
            {"role", "filename", "size_bytes", "sha256"},
            f"implementation_files[{index}]",
        )
        role = _plain_string(file_record["role"], f"implementation_files[{index}].role")
        filename = _plain_string(
            file_record["filename"], f"implementation_files[{index}].filename"
        )
        if pathlib.PurePosixPath(role).name != role or role in seen_roles:
            raise PowerValidationError(
                "implementation roles must be unique plain names"
            )
        if (
            pathlib.PurePosixPath(filename).name != filename
            or filename in seen_filenames
        ):
            raise PowerValidationError(
                "implementation filenames must be unique plain names"
            )
        seen_roles.add(role)
        seen_filenames.add(filename)
        normalized.append(
            {
                "role": role,
                "filename": filename,
                "size_bytes": _positive_int(
                    file_record["size_bytes"],
                    f"implementation_files[{index}].size_bytes",
                ),
                "sha256": _sha256_text(
                    file_record["sha256"], f"implementation_files[{index}].sha256"
                ),
            }
        )
    if normalized != sorted(normalized, key=lambda item: item["role"]):
        raise PowerValidationError("implementation files must be in role order")
    return normalized


def _validate_runtime(value: Any) -> dict[str, str]:
    runtime = _mapping(value, "runtime")
    _exact_keys(runtime, {"python_version", "numpy_version"}, "runtime")
    return {
        "python_version": _plain_string(
            runtime["python_version"], "runtime.python_version"
        ),
        "numpy_version": _plain_string(
            runtime["numpy_version"], "runtime.numpy_version"
        ),
    }


def validate_power_output(
    value: Any,
    *,
    expected_config_sha256: str | None = None,
    expected_summary_sha256: str | None = None,
    expected_implementation_bundle_sha256: str | None = None,
) -> None:
    """Validate report structure, provenance digests, and deterministic replay."""

    output = _mapping(value, "power output")
    _exact_keys(output, _OUTPUT_KEYS, "power output")
    payload_sha256 = _sha256_text(output["payload_sha256"], "payload_sha256")
    unhashed = {key: output[key] for key in output if key != "payload_sha256"}
    if _canonical_sha256(unhashed) != payload_sha256:
        raise PowerValidationError("payload SHA-256 does not match")
    if output["schema"] != OUTPUT_SCHEMA:
        raise PowerValidationError("power output schema is unexpected")
    if output["analysis"] != _analysis_description():
        raise PowerValidationError("power output analysis is unexpected")

    config = _validate_configuration(output["configuration"])
    summary = _validate_component_event_summary(output["component_event_summary"])
    implementation_files = _validate_implementation_files(
        output["implementation_files"]
    )
    _validate_runtime(output["runtime"])
    digests = _mapping(output["input_digests"], "input_digests")
    provenance_names = set(_training_provenance_digests(summary))
    digest_keys = {
        "configuration_sha256",
        "configuration_value_sha256",
        "component_event_summary_sha256",
        "component_event_summary_value_sha256",
        "implementation_bundle_sha256",
        "run_input_sha256",
        *provenance_names,
    }
    _exact_keys(digests, digest_keys, "input_digests")
    for name in digest_keys:
        _sha256_text(digests[name], f"input_digests.{name}")
    if _canonical_sha256(config) != digests["configuration_value_sha256"]:
        raise PowerValidationError("configuration value SHA-256 does not match")
    if _canonical_sha256(summary) != digests["component_event_summary_value_sha256"]:
        raise PowerValidationError(
            "component event summary value SHA-256 does not match"
        )
    if (
        _canonical_sha256(implementation_files)
        != digests["implementation_bundle_sha256"]
    ):
        raise PowerValidationError("implementation bundle SHA-256 does not match")
    expected_provenance = _training_provenance_digests(summary)
    for name, expected in expected_provenance.items():
        if digests[name] != expected:
            raise PowerValidationError(
                f"training provenance digest {name} does not match"
            )
    run_inputs = {
        name: digests[name] for name in digest_keys if name != "run_input_sha256"
    }
    if _canonical_sha256(run_inputs) != digests["run_input_sha256"]:
        raise PowerValidationError("run input SHA-256 does not match")
    if expected_config_sha256 is not None and digests[
        "configuration_sha256"
    ] != _sha256_text(expected_config_sha256, "expected configuration SHA-256"):
        raise PowerValidationError("output configuration SHA-256 differs from expected")
    if expected_summary_sha256 is not None and digests[
        "component_event_summary_sha256"
    ] != _sha256_text(expected_summary_sha256, "expected summary SHA-256"):
        raise PowerValidationError("output summary SHA-256 differs from expected")
    if expected_implementation_bundle_sha256 is not None and digests[
        "implementation_bundle_sha256"
    ] != _sha256_text(
        expected_implementation_bundle_sha256,
        "expected implementation bundle SHA-256",
    ):
        raise PowerValidationError(
            "output implementation SHA-256 differs from expected"
        )

    expected_design = build_power_design(
        summary,
        minimum_detectable_effect=float(
            config["simulation"]["minimum_detectable_error_reduction"]
        ),
    )
    if output["design"] != expected_design:
        raise PowerValidationError("power design differs from the component summary")
    expected_simulation = _simulate_power(config, expected_design)
    if output["simulation"] != expected_simulation:
        raise PowerValidationError("power simulation differs from deterministic replay")


def write_power_output(path: pathlib.Path, output: Mapping[str, Any]) -> None:
    """Atomically write one validated report without replacing an existing file."""

    validate_power_output(output)
    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                output,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, destination)
        except FileExistsError as exc:
            raise PowerValidationError("output path already exists") from exc
        os.unlink(temporary_name)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
