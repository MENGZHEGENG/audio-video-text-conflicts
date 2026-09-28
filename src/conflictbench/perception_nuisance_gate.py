"""Leakage-resistant edit-role diagnostics for Perception Test media pairs.

The detector is intentionally limited to container and stream engineering
measurements.  It never receives question text, options, answers, semantic
embeddings, source-answer scores, or source-sufficiency outcomes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pathlib
import shutil
import stat
import statistics
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

FEATURE_NAMES = (
    "duration_relative_difference",
    "byte_rate_relative_difference",
    "target_av_duration_gap_fraction",
    "donor_av_duration_gap_fraction",
    "target_video_donor_audio_gap_fraction",
    "target_audio_donor_video_gap_fraction",
    "width_relative_difference",
    "height_relative_difference",
    "frame_rate_relative_difference",
    "sample_rate_relative_difference",
    "channel_count_relative_difference",
    "video_codec_mismatch",
    "audio_codec_mismatch",
    "pixel_format_mismatch",
    "sample_format_mismatch",
)

IMPLEMENTATION_SOURCE_ROLES = (
    "__init__.py",
    "runner.py",
    "perception_nuisance_gate.py",
    "perception_media_pilot.py",
    "perception_candidate_audit.py",
    "perception_omni_gate.py",
)

SHORTCUT_OUTPUT_SCHEMA_V2 = "conflictbench.perception-shortcut-output.v2"
QUESTION_BLIND_REQUEST_SCHEMA_V2 = (
    "conflictbench.perception-question-blind-av-mismatch-request.v2"
)
QUESTION_BLIND_OUTPUT_SCHEMA_V2 = (
    "conflictbench.perception-question-blind-av-mismatch-output.v2"
)
CONTROL_REQUEST_SCHEMA_V2 = "conflictbench.perception-control-request.v2"
RELATION_OUTPUT_SCHEMA_V2 = "conflictbench.perception-conflict-relation-output.v2"
CHOICE_OUTPUT_SCHEMA_V2 = "conflictbench.perception-source-choice-output.v2"
REPLAY_REQUEST_SCHEMA_V2 = "conflictbench.perception-replay-request.v2"
REPLAY_RESPONSE_SCHEMA_V2 = "conflictbench.perception-replay-worker-response.v2"
SHORTCUT_PREREGISTRATION_SCHEMA_V2 = (
    "conflictbench.perception-shortcut-preregistration.v2"
)
SHORTCUT_POWER_SCHEMA_V2 = "conflictbench.perception-shortcut-power.v2"
VERIFIED_DEPENDENCY_SCHEMA_V2 = "conflictbench.perception-verified-dependency.v2"
DEPENDENCY_SUBJECT_PROJECTION_SCHEMA_V2 = (
    "conflictbench.perception-dependency-subject-projection.v2"
)
DEPENDENCY_TRANSCRIPT_SCHEMA_V2 = (
    "conflictbench.perception-dependency-verification-transcript.v2"
)
DEPENDENCY_VERIFIER_REQUEST_SCHEMA_V2 = (
    "conflictbench.perception-dependency-verifier-request.v2"
)
DEPENDENCY_VERIFIER_RESPONSE_SCHEMA_V2 = (
    "conflictbench.perception-dependency-verifier-response.v2"
)
SHORTCUT_POWER_DGP_SCHEMA_V2 = "conflictbench.perception-shortcut-power-dgp.v2"
POWER_SIMULATOR_REQUEST_SCHEMA_V2 = (
    "conflictbench.perception-power-simulator-request.v2"
)
POWER_SIMULATOR_RESPONSE_SCHEMA_V2 = (
    "conflictbench.perception-power-simulator-response.v2"
)
POWER_SIMULATION_ALGORITHM_V2 = "sha256_counter_endpoint_gaussian_bounds_v1"

_V2_DEPENDENCY_DIGEST_KEYS = {
    "raw_conflict_index_sha256",
    "raw_conflict_index_verification_sha256",
    "source_only_output_sha256",
    "source_only_replay_verification_sha256",
    "semantic_gate_output_sha256",
    "semantic_gate_replay_verification_sha256",
    "derived_media_v4_index_sha256",
    "derived_media_v4_verification_sha256",
    "power_output_sha256",
}
_V2_WORKER_DIGEST_KEYS = {
    "implementation_sha256",
    "configuration_sha256",
    "model_sha256",
}
_V2_DEPENDENCY_ROLES = {
    "raw_conflict_index": (
        "raw_conflict_index_sha256",
        "raw_conflict_index_verification_sha256",
    ),
    "source_only": (
        "source_only_output_sha256",
        "source_only_replay_verification_sha256",
    ),
    "semantic_gate": (
        "semantic_gate_output_sha256",
        "semantic_gate_replay_verification_sha256",
    ),
    "derived_media_v4": (
        "derived_media_v4_index_sha256",
        "derived_media_v4_verification_sha256",
    ),
}
_V2_POWER_ENDPOINTS = (
    "metadata_balanced_accuracy",
    "question_only_balanced_accuracy",
    "question_blind_balanced_accuracy",
    "remux_accuracy_difference",
    "remux_p_gold_difference",
    "orientation_difference",
    "option_permutation_0_1_difference",
    "option_permutation_0_2_difference",
    "option_permutation_1_2_difference",
    "shuffled_question_balanced_accuracy",
    "aligned_minus_shuffled_difference",
)
_V2_POWER_ENDPOINT_RULES = {
    "metadata_balanced_accuracy": {
        "type": "upper_confidence_bound_below",
        "upper_boundary": 0.55,
    },
    "question_only_balanced_accuracy": {
        "type": "upper_confidence_bound_below",
        "upper_boundary": 0.55,
    },
    "question_blind_balanced_accuracy": {
        "type": "upper_confidence_bound_below",
        "upper_boundary": 0.55,
    },
    "remux_accuracy_difference": {
        "type": "lower_confidence_bound_above",
        "lower_boundary": -0.02,
    },
    "remux_p_gold_difference": {
        "type": "lower_confidence_bound_above",
        "lower_boundary": -0.03,
    },
    "orientation_difference": {
        "type": "confidence_interval_within",
        "lower_boundary": -0.05,
        "upper_boundary": 0.05,
    },
    "option_permutation_0_1_difference": {
        "type": "confidence_interval_within",
        "lower_boundary": -0.05,
        "upper_boundary": 0.05,
    },
    "option_permutation_0_2_difference": {
        "type": "confidence_interval_within",
        "lower_boundary": -0.05,
        "upper_boundary": 0.05,
    },
    "option_permutation_1_2_difference": {
        "type": "confidence_interval_within",
        "lower_boundary": -0.05,
        "upper_boundary": 0.05,
    },
    "shuffled_question_balanced_accuracy": {
        "type": "upper_confidence_bound_below",
        "upper_boundary": 0.55,
    },
    "aligned_minus_shuffled_difference": {
        "type": "lower_confidence_bound_above",
        "lower_boundary": 0.10,
    },
}


@dataclass(frozen=True)
class SubprocessReplayWorkerSpec:
    """Data-only identity for an isolated, hash-pinned replay executable."""

    executable_path: pathlib.Path
    executable_sha256: str
    configuration_path: pathlib.Path
    configuration_sha256: str
    model_path: pathlib.Path
    model_sha256: str


@dataclass(frozen=True)
class SubprocessDependencyVerifierSpec:
    """Pinned executable and configuration for one dependency replay verifier."""

    executable_path: pathlib.Path
    executable_sha256: str
    configuration_path: pathlib.Path
    configuration_sha256: str


@dataclass(frozen=True)
class SubprocessPowerSimulatorSpec:
    """Pinned executable and configuration for the frozen power simulator."""

    executable_path: pathlib.Path
    executable_sha256: str
    configuration_path: pathlib.Path
    configuration_sha256: str


class NuisanceValidationError(ValueError):
    """Raised when an input or output violates the frozen detector protocol."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NuisanceValidationError(f"{label} must be an object")
    return value


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise NuisanceValidationError(f"{label} must be a positive finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise NuisanceValidationError(
            f"{label} must be a positive finite number"
        ) from exc
    if not math.isfinite(number) or number <= 0.0:
        raise NuisanceValidationError(f"{label} must be a positive finite number")
    return number


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise NuisanceValidationError(f"{label} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise NuisanceValidationError(f"{label} must be a positive integer") from exc
    if number <= 0 or str(value).strip() not in {str(number), f"{number}.0"}:
        raise NuisanceValidationError(f"{label} must be a positive integer")
    return number


def _frame_rate(value: Any) -> float:
    if not isinstance(value, str) or not value:
        raise NuisanceValidationError("video frame rate must be a nonempty ratio")
    pieces = value.split("/")
    if len(pieces) == 1:
        return _finite_positive(pieces[0], "video frame rate")
    if len(pieces) != 2:
        raise NuisanceValidationError("video frame rate must be a valid ratio")
    numerator = _finite_positive(pieces[0], "video frame-rate numerator")
    denominator = _finite_positive(pieces[1], "video frame-rate denominator")
    return numerator / denominator


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NuisanceValidationError(f"{label} must be nonempty text")
    return value.strip()


def parse_ffprobe_record(payload: Any, *, size_bytes: int) -> dict[str, Any]:
    """Normalize one strict ``ffprobe -show_format -show_streams`` response."""

    record = _mapping(payload, "ffprobe response")
    if set(record) != {"format", "streams"}:
        raise NuisanceValidationError("ffprobe response fields are unexpected")
    streams = record["streams"]
    if not isinstance(streams, list):
        raise NuisanceValidationError("ffprobe streams must be a list")
    audio = [
        stream
        for stream in streams
        if isinstance(stream, Mapping) and stream.get("codec_type") == "audio"
    ]
    video = [
        stream
        for stream in streams
        if isinstance(stream, Mapping) and stream.get("codec_type") == "video"
    ]
    if len(audio) != 1 or len(video) != 1:
        raise NuisanceValidationError(
            "media must expose exactly one audio and one video stream"
        )
    audio_stream = audio[0]
    video_stream = video[0]
    duration = _finite_positive(
        _mapping(record["format"], "ffprobe format").get("duration"),
        "format duration",
    )
    file_size = _positive_integer(size_bytes, "media size")
    result = {
        "duration_seconds": duration,
        "size_bytes": file_size,
        "byte_rate": file_size / duration,
        "video": {
            "duration_seconds": _finite_positive(
                video_stream.get("duration"), "video duration"
            ),
            "width": _positive_integer(video_stream.get("width"), "video width"),
            "height": _positive_integer(video_stream.get("height"), "video height"),
            "frame_rate": _frame_rate(video_stream.get("avg_frame_rate")),
            "codec": _nonempty_text(video_stream.get("codec_name"), "video codec"),
            "pixel_format": _nonempty_text(
                video_stream.get("pix_fmt"), "video pixel format"
            ),
        },
        "audio": {
            "duration_seconds": _finite_positive(
                audio_stream.get("duration"), "audio duration"
            ),
            "sample_rate": _positive_integer(
                audio_stream.get("sample_rate"), "audio sample rate"
            ),
            "channels": _positive_integer(
                audio_stream.get("channels"), "audio channel count"
            ),
            "codec": _nonempty_text(audio_stream.get("codec_name"), "audio codec"),
            "sample_format": _nonempty_text(
                audio_stream.get("sample_fmt"), "audio sample format"
            ),
        },
    }
    return result


def _relative_difference(first: float, second: float) -> float:
    return abs(first - second) / max(abs(first), abs(second))


def _mismatch(first: str, second: str) -> float:
    return float(first != second)


def extract_metadata_features(
    target: Mapping[str, Any], donor: Mapping[str, Any]
) -> np.ndarray:
    """Return the frozen, non-semantic feature vector for one media pair."""

    target_video = _mapping(target.get("video"), "target video diagnostics")
    target_audio = _mapping(target.get("audio"), "target audio diagnostics")
    donor_video = _mapping(donor.get("video"), "donor video diagnostics")
    donor_audio = _mapping(donor.get("audio"), "donor audio diagnostics")
    target_duration = _finite_positive(
        target.get("duration_seconds"), "target duration"
    )
    donor_duration = _finite_positive(donor.get("duration_seconds"), "donor duration")
    values = np.asarray(
        [
            _relative_difference(target_duration, donor_duration),
            _relative_difference(
                _finite_positive(target.get("byte_rate"), "target byte rate"),
                _finite_positive(donor.get("byte_rate"), "donor byte rate"),
            ),
            abs(
                _finite_positive(
                    target_video.get("duration_seconds"), "target video duration"
                )
                - _finite_positive(
                    target_audio.get("duration_seconds"), "target audio duration"
                )
            )
            / target_duration,
            abs(
                _finite_positive(
                    donor_video.get("duration_seconds"), "donor video duration"
                )
                - _finite_positive(
                    donor_audio.get("duration_seconds"), "donor audio duration"
                )
            )
            / donor_duration,
            _relative_difference(
                _finite_positive(
                    target_video.get("duration_seconds"), "target video duration"
                ),
                _finite_positive(
                    donor_audio.get("duration_seconds"), "donor audio duration"
                ),
            ),
            _relative_difference(
                _finite_positive(
                    target_audio.get("duration_seconds"), "target audio duration"
                ),
                _finite_positive(
                    donor_video.get("duration_seconds"), "donor video duration"
                ),
            ),
            _relative_difference(
                _finite_positive(target_video.get("width"), "target video width"),
                _finite_positive(donor_video.get("width"), "donor video width"),
            ),
            _relative_difference(
                _finite_positive(target_video.get("height"), "target video height"),
                _finite_positive(donor_video.get("height"), "donor video height"),
            ),
            _relative_difference(
                _finite_positive(target_video.get("frame_rate"), "target frame rate"),
                _finite_positive(donor_video.get("frame_rate"), "donor frame rate"),
            ),
            _relative_difference(
                _finite_positive(target_audio.get("sample_rate"), "target sample rate"),
                _finite_positive(donor_audio.get("sample_rate"), "donor sample rate"),
            ),
            _relative_difference(
                _finite_positive(target_audio.get("channels"), "target channel count"),
                _finite_positive(donor_audio.get("channels"), "donor channel count"),
            ),
            _mismatch(
                _nonempty_text(target_video.get("codec"), "target video codec"),
                _nonempty_text(donor_video.get("codec"), "donor video codec"),
            ),
            _mismatch(
                _nonempty_text(target_audio.get("codec"), "target audio codec"),
                _nonempty_text(donor_audio.get("codec"), "donor audio codec"),
            ),
            _mismatch(
                _nonempty_text(target_video.get("pixel_format"), "target pixel format"),
                _nonempty_text(donor_video.get("pixel_format"), "donor pixel format"),
            ),
            _mismatch(
                _nonempty_text(
                    target_audio.get("sample_format"), "target sample format"
                ),
                _nonempty_text(donor_audio.get("sample_format"), "donor sample format"),
            ),
        ],
        dtype=np.float64,
    )
    if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
        raise NuisanceValidationError("nuisance feature vector is invalid")
    return values


_PARTITIONS = ("scorer_fit", "threshold_calibration", "pilot_gate")
_ROLES = ("same_answer_nuisance", "opposite_answer_candidate")
_CONFIG_KEYS = {
    "schema_version",
    "runner",
    "feature_set",
    "model",
    "protocol",
    "uncertainty",
    "permutation",
    "gate",
}
_CONFIG_V2_KEYS = _CONFIG_KEYS | {"shortcut_protocol", "shortcut_gate"}


def _canonical_digest(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NuisanceValidationError(f"value is not strict JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise NuisanceValidationError(f"{label} fields are unexpected")


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NuisanceValidationError(f"{label} must be a nonnegative integer")
    return value


def _probability(value: Any, label: str, *, open_interval: bool = False) -> float:
    if isinstance(value, bool):
        raise NuisanceValidationError(f"{label} must be a probability")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise NuisanceValidationError(f"{label} must be a probability") from exc
    valid = 0.0 < number < 1.0 if open_interval else 0.0 <= number <= 1.0
    if not math.isfinite(number) or not valid:
        raise NuisanceValidationError(f"{label} must be a probability")
    return number


def _exact_float(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise NuisanceValidationError(f"{label} must be a finite JSON float")
    if minimum is not None and value < minimum:
        raise NuisanceValidationError(f"{label} is below its minimum")
    if maximum is not None and value > maximum:
        raise NuisanceValidationError(f"{label} exceeds its maximum")
    return value


def _validate_shortcut_config_v2(config: Mapping[str, Any]) -> None:
    protocol = _mapping(config["shortcut_protocol"], "shortcut_protocol")
    _exact_keys(
        protocol,
        {
            "preregistration_schema",
            "output_schema",
            "report_schema",
            "result_schema",
            "power_schema",
            "verified_dependency_schema",
            "replay_worker_request_schema",
            "replay_worker_response_schema",
            "question_blind_request_schema",
            "question_blind_output_schema",
            "question_blind_allowed_request_fields",
            "question_blind_forbidden_inputs",
            "replay_verification",
            "question_only_replay_verification",
            "dependency_roles",
            "derived_media_authentication",
            "chronology",
            "power_simulations",
            "uncertainty_unit",
            "uncertainty_method",
            "confidence_level",
            "bootstrap_repetitions",
            "bootstrap_seed",
        },
        "shortcut_protocol",
    )
    expected_text = {
        "preregistration_schema": SHORTCUT_PREREGISTRATION_SCHEMA_V2,
        "output_schema": "conflictbench.perception-shortcut-output.v2",
        "report_schema": "conflictbench.perception-shortcut-report.v2",
        "result_schema": "conflictbench.perception-shortcut-result.v2",
        "power_schema": SHORTCUT_POWER_SCHEMA_V2,
        "verified_dependency_schema": VERIFIED_DEPENDENCY_SCHEMA_V2,
        "replay_worker_request_schema": REPLAY_REQUEST_SCHEMA_V2,
        "replay_worker_response_schema": REPLAY_RESPONSE_SCHEMA_V2,
        "question_blind_request_schema": QUESTION_BLIND_REQUEST_SCHEMA_V2,
        "question_blind_output_schema": QUESTION_BLIND_OUTPUT_SCHEMA_V2,
        "replay_verification": "two_fresh_exact_replays",
        "question_only_replay_verification": "two_fresh_exact_replays",
        "derived_media_authentication": "exact_inventory_before_and_after_replay",
        "chronology": "preregistration_bound_before_completed_output",
        "uncertainty_unit": "connected_analysis_component",
        "uncertainty_method": (
            "one_sided_component_cluster_percentile_bonferroni_familywise"
        ),
    }
    for key, expected in expected_text.items():
        if protocol.get(key) != expected:
            raise NuisanceValidationError(f"shortcut protocol {key} differs")
    if protocol.get("dependency_roles") != list(_V2_DEPENDENCY_ROLES):
        raise NuisanceValidationError("shortcut dependency roles differ")
    if protocol.get("power_simulations") != 10_000:
        raise NuisanceValidationError("shortcut power simulations must equal 10000")
    if protocol.get("question_blind_allowed_request_fields") != [
        "schema",
        "audio",
        "video",
    ]:
        raise NuisanceValidationError("question-blind request allowlist differs")
    if protocol.get("question_blind_forbidden_inputs") != [
        "question",
        "options",
        "answer",
        "condition",
        "orientation",
        "option_permutation",
        "component_id",
        "question_key_sha256",
    ]:
        raise NuisanceValidationError("question-blind forbidden-input list differs")
    if (
        _exact_float(
            protocol.get("confidence_level"),
            "shortcut confidence level",
            minimum=0.0,
            maximum=1.0,
        )
        != 0.95
    ):
        raise NuisanceValidationError("shortcut confidence level must be 0.95")
    shortcut_repetitions = protocol.get("bootstrap_repetitions")
    if type(shortcut_repetitions) is not int or shortcut_repetitions < 10_000:
        raise NuisanceValidationError(
            "shortcut bootstrap repetitions must be at least 10000"
        )
    _nonnegative_integer(protocol.get("bootstrap_seed"), "shortcut bootstrap seed")

    gate = _mapping(config["shortcut_gate"], "shortcut_gate")
    _exact_keys(
        gate,
        {
            "maximum_metadata_balanced_accuracy_ucb_exclusive",
            "maximum_question_only_balanced_accuracy_ucb_exclusive",
            "maximum_question_blind_balanced_accuracy_ucb_exclusive",
            "minimum_remux_accuracy_difference_exclusive",
            "remux_p_gold_minimum",
            "equivalence_interval",
            "maximum_shuffled_question_balanced_accuracy_ucb_exclusive",
            "minimum_aligned_minus_shuffled_lcb_inclusive",
        },
        "shortcut_gate",
    )
    fixed_values = {
        "maximum_metadata_balanced_accuracy_ucb_exclusive": 0.55,
        "maximum_question_only_balanced_accuracy_ucb_exclusive": 0.55,
        "maximum_question_blind_balanced_accuracy_ucb_exclusive": 0.55,
        "minimum_remux_accuracy_difference_exclusive": -0.02,
        "maximum_shuffled_question_balanced_accuracy_ucb_exclusive": 0.55,
        "minimum_aligned_minus_shuffled_lcb_inclusive": 0.10,
    }
    for key, expected in fixed_values.items():
        if _exact_float(gate.get(key), f"shortcut gate {key}") != expected:
            raise NuisanceValidationError(f"shortcut gate {key} differs")
    if gate.get("remux_p_gold_minimum") != "negative_semantic_fit_epsilon":
        raise NuisanceValidationError("remux p_gold margin source differs")
    interval = gate.get("equivalence_interval")
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or any(type(value) is not float for value in interval)
        or interval != [-0.05, 0.05]
    ):
        raise NuisanceValidationError("shortcut equivalence interval differs")


def _validate_config(config_value: Any) -> Mapping[str, Any]:
    config = _mapping(config_value, "configuration")
    schema_version = config.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, 2}:
        raise NuisanceValidationError("configuration schema version differs")
    _exact_keys(
        config,
        _CONFIG_KEYS if schema_version == 1 else _CONFIG_V2_KEYS,
        "configuration",
    )
    expected_runner = (
        "perception_edit_role_nuisance_detector"
        if schema_version == 1
        else "perception_edit_role_shortcut_gate"
    )
    if config.get("runner") != expected_runner:
        raise NuisanceValidationError("configuration runner differs")

    feature_set = _mapping(config["feature_set"], "feature_set")
    _exact_keys(feature_set, {"name", "names", "forbidden_inputs"}, "feature_set")
    if feature_set.get("name") != "container_and_stream_diagnostics_v1":
        raise NuisanceValidationError("feature-set identity differs")
    if feature_set.get("names") != list(FEATURE_NAMES):
        raise NuisanceValidationError("feature names or order differ")
    forbidden = feature_set.get("forbidden_inputs")
    required_forbidden = {
        "question",
        "options",
        "answer",
        "semantic_embedding",
        "source_answer_score",
        "source_sufficiency_outcome",
    }
    if not isinstance(forbidden, list) or set(forbidden) != required_forbidden:
        raise NuisanceValidationError("forbidden-input contract differs")

    model = _mapping(config["model"], "model")
    _exact_keys(model, {"families", "ridge_l2", "selection"}, "model")
    if model.get("families") != [
        "ridge_linear",
        "univariate_linear",
        "pairwise_interaction",
    ]:
        raise NuisanceValidationError("model families or order differ")
    _finite_positive(model.get("ridge_l2"), "model ridge l2")
    if model.get("selection") != "fit_only_max_absolute_covariance":
        raise NuisanceValidationError("model selection protocol differs")

    protocol = _mapping(config["protocol"], "protocol")
    _exact_keys(
        protocol,
        {
            "fit_partition",
            "threshold_partition",
            "evaluation_partition",
            "positive_role",
            "negative_role",
            "threshold_selection",
            "question_key_sensitivity",
            "minimum_unseen_question_components",
            "human_evaluation",
            "outcome_based_sample_selection",
        },
        "protocol",
    )
    expected_protocol = {
        "fit_partition": "scorer_fit",
        "threshold_partition": "threshold_calibration",
        "evaluation_partition": "pilot_gate",
        "positive_role": "opposite_answer_candidate",
        "negative_role": "same_answer_nuisance",
        "threshold_selection": "maximum_balanced_accuracy_then_closest_to_zero",
        "question_key_sensitivity": "evaluation_keys_unseen_in_fit_or_threshold",
        "human_evaluation": "forbidden",
        "outcome_based_sample_selection": "forbidden",
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise NuisanceValidationError(f"protocol {key} differs")
    if (
        _positive_integer(
            protocol.get("minimum_unseen_question_components"),
            "minimum unseen-question component count",
        )
        < 2
    ):
        raise NuisanceValidationError(
            "minimum unseen-question component count must be at least two"
        )

    uncertainty = _mapping(config["uncertainty"], "uncertainty")
    _exact_keys(
        uncertainty,
        {
            "method",
            "confidence_level",
            "bootstrap_repetitions",
            "bootstrap_seed",
        },
        "uncertainty",
    )
    expected_uncertainty_method = (
        "component_cluster_percentile_wilson_and_max_statistic"
        if schema_version == 1
        else "component_cluster_percentile_and_max_statistic"
    )
    if uncertainty.get("method") != expected_uncertainty_method:
        raise NuisanceValidationError("uncertainty method differs")
    if (
        _probability(
            uncertainty.get("confidence_level"),
            "uncertainty confidence level",
            open_interval=True,
        )
        != 0.95
    ):
        raise NuisanceValidationError("uncertainty confidence level must be 0.95")
    _positive_integer(uncertainty.get("bootstrap_repetitions"), "bootstrap repetitions")
    _nonnegative_integer(uncertainty.get("bootstrap_seed"), "bootstrap seed")

    permutation = _mapping(config["permutation"], "permutation")
    _exact_keys(permutation, {"method", "repetitions", "seed"}, "permutation")
    if permutation.get("method") != "within_component_role_swap_with_refit":
        raise NuisanceValidationError("permutation method differs")
    _positive_integer(permutation.get("repetitions"), "permutation repetitions")
    _nonnegative_integer(permutation.get("seed"), "permutation seed")

    gate = _mapping(config["gate"], "gate")
    _exact_keys(
        gate,
        {
            "maximum_primary_upper_confidence_bound",
            "maximum_unseen_question_upper_confidence_bound",
            "maximum_point_lift_over_structural_chance",
            "maximum_structural_chance_balanced_accuracy",
            "minimum_permutation_p_value_exclusive",
        },
        "gate",
    )
    for key in gate:
        _probability(gate[key], f"gate {key}")
    if schema_version == 2:
        if uncertainty["bootstrap_repetitions"] < 10_000:
            raise NuisanceValidationError(
                "v2 bootstrap repetitions must be at least 10000"
            )
        if permutation["repetitions"] < 10_000:
            raise NuisanceValidationError(
                "v2 permutation repetitions must be at least 10000"
            )
        if (
            gate["maximum_primary_upper_confidence_bound"] != 0.55
            or gate["maximum_unseen_question_upper_confidence_bound"] != 0.55
        ):
            raise NuisanceValidationError(
                "v2 metadata upper-confidence limits must equal 0.55"
            )
        _validate_shortcut_config_v2(config)
    return config


def _validate_records(
    records_value: Sequence[Mapping[str, Any]], *, require_all_partitions: bool = True
) -> list[dict[str, Any]]:
    if not isinstance(records_value, Sequence) or isinstance(
        records_value, (str, bytes)
    ):
        raise NuisanceValidationError("feature records must be a sequence")
    records: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    target_groups: dict[str, list[dict[str, Any]]] = {}
    component_partitions: dict[str, str] = {}
    video_partitions: dict[str, str] = {}
    for index, raw in enumerate(records_value):
        record = _mapping(raw, f"feature record {index}")
        _exact_keys(
            record,
            {
                "pair_id",
                "target_id",
                "component_id",
                "partition",
                "key_sha256",
                "target_video_id",
                "donor_video_id",
                "role",
                "features",
            },
            f"feature record {index}",
        )
        normalized: dict[str, Any] = {}
        for key in (
            "pair_id",
            "target_id",
            "component_id",
            "key_sha256",
            "target_video_id",
            "donor_video_id",
        ):
            normalized[key] = _nonempty_text(record.get(key), f"feature record {key}")
        if normalized["pair_id"] in pair_ids:
            raise NuisanceValidationError("pair IDs must be unique")
        pair_ids.add(normalized["pair_id"])
        partition = record.get("partition")
        if partition not in _PARTITIONS:
            raise NuisanceValidationError("feature-record partition differs")
        normalized["partition"] = partition
        role = record.get("role")
        if role not in _ROLES:
            raise NuisanceValidationError("feature-record role differs")
        normalized["role"] = role
        features = np.asarray(record.get("features"), dtype=np.float64)
        if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
            raise NuisanceValidationError("feature record has an invalid vector")
        normalized["features"] = features
        prior_partition = component_partitions.setdefault(
            normalized["component_id"], partition
        )
        if prior_partition != partition:
            raise NuisanceValidationError("component crosses partitions")
        for video_id in (normalized["target_video_id"], normalized["donor_video_id"]):
            video_partition = video_partitions.setdefault(video_id, partition)
            if video_partition != partition:
                raise NuisanceValidationError("video crosses partitions")
        target_groups.setdefault(normalized["target_id"], []).append(normalized)
        records.append(normalized)

    if not records:
        raise NuisanceValidationError("feature records are empty")
    if require_all_partitions and {record["partition"] for record in records} != set(
        _PARTITIONS
    ):
        raise NuisanceValidationError("all three partitions must be nonempty")
    for target_id, target_records in target_groups.items():
        if len(target_records) != 2 or {
            record["role"] for record in target_records
        } != set(_ROLES):
            raise NuisanceValidationError(
                f"target {target_id} must have exactly one record for each role"
            )
        for key in ("component_id", "partition", "key_sha256", "target_video_id"):
            if len({record[key] for record in target_records}) != 1:
                raise NuisanceValidationError(
                    f"target {target_id} has inconsistent {key}"
                )
    return records


def _labels(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [float(record["role"] == "opposite_answer_candidate") for record in records],
        dtype=np.float64,
    )


def _matrix(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.stack(
        [np.asarray(record["features"], dtype=np.float64) for record in records]
    )


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    values = []
    for label in (0.0, 1.0):
        mask = labels == label
        if not np.any(mask):
            raise NuisanceValidationError("balanced accuracy needs both roles")
        values.append(float(np.mean(predictions[mask] == label)))
    return float(np.mean(values))


def _fit_ridge(records: Sequence[Mapping[str, Any]], l2: float) -> dict[str, Any]:
    matrix = _matrix(records)
    labels = 2.0 * _labels(records) - 1.0
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardized = (matrix - means) / scales
    design = np.column_stack((np.ones(len(matrix)), standardized))
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(l2)
    penalty[0, 0] = 0.0
    try:
        coefficients = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ labels,
        )
    except np.linalg.LinAlgError as exc:
        raise NuisanceValidationError("ridge detector fit failed") from exc
    if not np.isfinite(coefficients).all():
        raise NuisanceValidationError("ridge detector coefficients are invalid")
    return {
        "feature_means": means,
        "feature_scales": scales,
        "coefficients": coefficients,
    }


def _score(
    model: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    matrix = _matrix(records)
    standardized = (matrix - model["feature_means"]) / model["feature_scales"]
    return model["coefficients"][0] + standardized @ model["coefficients"][1:]


def _fit_covariance_selector(
    records: Sequence[Mapping[str, Any]], *, interaction: bool
) -> dict[str, Any]:
    """Fit a deterministic one-feature detector using only the fit partition."""

    matrix = _matrix(records)
    labels = 2.0 * _labels(records) - 1.0
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardized = (matrix - means) / scales
    if interaction:
        pairs = [
            (first, second)
            for first in range(standardized.shape[1])
            for second in range(first + 1, standardized.shape[1])
        ]
        candidates = np.column_stack(
            [
                standardized[:, first] * standardized[:, second]
                for first, second in pairs
            ]
        )
        labels_by_candidate = [
            f"{FEATURE_NAMES[first]}__x__{FEATURE_NAMES[second]}"
            for first, second in pairs
        ]
    else:
        pairs = [(index, index) for index in range(standardized.shape[1])]
        candidates = standardized
        labels_by_candidate = list(FEATURE_NAMES)
    covariances = np.asarray(
        [
            float(np.mean(candidates[:, index] * labels))
            for index in range(candidates.shape[1])
        ]
    )
    selected = int(np.argmax(np.abs(covariances)))
    orientation = 1.0 if covariances[selected] >= 0.0 else -1.0
    return {
        "feature_means": means,
        "feature_scales": scales,
        "selected_indices": pairs[selected],
        "selected_feature": labels_by_candidate[selected],
        "orientation": orientation,
        "fit_covariance": float(covariances[selected]),
        "interaction": interaction,
    }


def _score_frozen(
    frozen: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    family = frozen.get("family")
    matrix = _matrix(records)
    means = np.asarray(frozen["feature_means"], dtype=np.float64)
    scales = np.asarray(frozen["feature_scales"], dtype=np.float64)
    standardized = (matrix - means) / scales
    if family == "ridge_linear":
        coefficients = np.asarray(frozen["coefficients"], dtype=np.float64)
        return coefficients[0] + standardized @ coefficients[1:]
    if family in {"univariate_linear", "pairwise_interaction"}:
        indices = frozen.get("selected_indices")
        if (
            not isinstance(indices, list)
            or len(indices) != 2
            or not all(isinstance(index, int) for index in indices)
        ):
            raise NuisanceValidationError("frozen detector selector is invalid")
        first, second = indices
        values = standardized[:, first]
        if family == "pairwise_interaction":
            values = values * standardized[:, second]
        return float(frozen["orientation"]) * values
    raise NuisanceValidationError("frozen detector family differs")


def _select_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    unique = np.unique(scores)
    if len(unique) == 1:
        candidates = np.asarray([float(unique[0])], dtype=np.float64)
    else:
        mids = (unique[:-1] + unique[1:]) / 2.0
        padding = max(1.0, float(np.ptp(unique)))
        candidates = np.concatenate(
            ([unique[0] - padding], mids, [unique[-1] + padding])
        )
    ranked = []
    for threshold in candidates.tolist():
        accuracy = _balanced_accuracy(labels, (scores >= threshold).astype(float))
        ranked.append((-accuracy, abs(threshold), threshold))
    _, _, selected = min(ranked)
    return float(selected), float(-min(ranked)[0])


def freeze_detector(
    records_value: Sequence[Mapping[str, Any]], config_value: Mapping[str, Any]
) -> dict[str, Any]:
    """Fit on ``scorer_fit`` and freeze a threshold on calibration only."""

    config = _validate_config(config_value)
    records = _validate_records(records_value)
    fit = [record for record in records if record["partition"] == "scorer_fit"]
    calibration = [
        record for record in records if record["partition"] == "threshold_calibration"
    ]
    model = _fit_ridge(fit, float(config["model"]["ridge_l2"]))
    calibration_scores = _score(model, calibration)
    threshold, calibration_accuracy = _select_threshold(
        calibration_scores, _labels(calibration)
    )
    frozen = {
        "family": "ridge_linear",
        "feature_names": list(FEATURE_NAMES),
        "feature_means": model["feature_means"].tolist(),
        "feature_scales": model["feature_scales"].tolist(),
        "coefficients": model["coefficients"].tolist(),
        "threshold": threshold,
        "calibration_balanced_accuracy": calibration_accuracy,
        "fit_record_ids": sorted(record["pair_id"] for record in fit),
        "threshold_record_ids": sorted(record["pair_id"] for record in calibration),
    }
    frozen["threshold_record_sha256"] = _canonical_digest(frozen)
    return frozen


def freeze_detector_suite(
    records_value: Sequence[Mapping[str, Any]], config_value: Mapping[str, Any]
) -> dict[str, Any]:
    """Freeze every predeclared detector before inspecting the pilot partition."""

    config = _validate_config(config_value)
    records = _validate_records(records_value)
    fit = [record for record in records if record["partition"] == "scorer_fit"]
    calibration = [
        record for record in records if record["partition"] == "threshold_calibration"
    ]
    fit_ids = sorted(record["pair_id"] for record in fit)
    calibration_ids = sorted(record["pair_id"] for record in calibration)
    models: dict[str, dict[str, Any]] = {}
    for family in config["model"]["families"]:
        if family == "ridge_linear":
            fitted = _fit_ridge(fit, float(config["model"]["ridge_l2"]))
            frozen: dict[str, Any] = {
                "family": family,
                "feature_names": list(FEATURE_NAMES),
                "feature_means": fitted["feature_means"].tolist(),
                "feature_scales": fitted["feature_scales"].tolist(),
                "coefficients": fitted["coefficients"].tolist(),
            }
        else:
            fitted = _fit_covariance_selector(
                fit, interaction=family == "pairwise_interaction"
            )
            frozen = {
                "family": family,
                "feature_names": list(FEATURE_NAMES),
                "feature_means": fitted["feature_means"].tolist(),
                "feature_scales": fitted["feature_scales"].tolist(),
                "selected_indices": list(fitted["selected_indices"]),
                "selected_feature": fitted["selected_feature"],
                "orientation": fitted["orientation"],
                "fit_covariance": fitted["fit_covariance"],
            }
        calibration_scores = _score_frozen(frozen, calibration)
        threshold, calibration_accuracy = _select_threshold(
            calibration_scores, _labels(calibration)
        )
        frozen.update(
            {
                "threshold": threshold,
                "calibration_balanced_accuracy": calibration_accuracy,
                "fit_record_ids": fit_ids,
                "threshold_record_ids": calibration_ids,
            }
        )
        frozen["threshold_record_sha256"] = _canonical_digest(frozen)
        models[family] = frozen
    suite = {
        "families": list(config["model"]["families"]),
        "selection": config["model"]["selection"],
        "models": models,
    }
    suite["detector_suite_sha256"] = _canonical_digest(suite)
    return suite


def _model_arrays(frozen: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        "feature_means": np.asarray(frozen["feature_means"], dtype=np.float64),
        "feature_scales": np.asarray(frozen["feature_scales"], dtype=np.float64),
        "coefficients": np.asarray(frozen["coefficients"], dtype=np.float64),
    }


def _ranking_credits(
    records: Sequence[Mapping[str, Any]], scores: np.ndarray
) -> list[dict[str, Any]]:
    by_target: dict[str, list[tuple[Mapping[str, Any], float]]] = {}
    for record, score in zip(records, scores.tolist()):
        by_target.setdefault(record["target_id"], []).append((record, score))
    credits = []
    for target_id in sorted(by_target):
        entries = by_target[target_id]
        if len(entries) != 2:
            raise NuisanceValidationError("target ranking inputs are incomplete")
        by_role = {record["role"]: score for record, score in entries}
        difference = (
            by_role["opposite_answer_candidate"] - by_role["same_answer_nuisance"]
        )
        credit = 1.0 if difference > 0.0 else (0.0 if difference < 0.0 else 0.5)
        sample = entries[0][0]
        credits.append(
            {
                "target_id": target_id,
                "component_id": sample["component_id"],
                "key_sha256": sample["key_sha256"],
                "same_answer_score": float(by_role["same_answer_nuisance"]),
                "opposite_answer_score": float(by_role["opposite_answer_candidate"]),
                "score_difference": float(difference),
                "credit": credit,
            }
        )
    return credits


def _wilson_interval(
    successes: float, count: int, *, confidence_level: float = 0.95
) -> tuple[float, float]:
    if count <= 0 or successes < 0.0 or successes > count:
        raise NuisanceValidationError("Wilson interval inputs are invalid")
    confidence = _probability(
        confidence_level, "Wilson confidence level", open_interval=True
    )
    z = statistics.NormalDist().inv_cdf(0.5 + confidence / 2.0)
    rate = successes / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _cluster_interval(
    credits: Sequence[Mapping[str, Any]],
    *,
    repetitions: int,
    seed: int,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    component_values: dict[str, list[float]] = {}
    for item in credits:
        component_values.setdefault(item["component_id"], []).append(
            float(item["credit"])
        )
    components = sorted(component_values)
    if len(components) < 2:
        raise NuisanceValidationError("cluster interval needs at least two components")
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for draw_index in range(repetitions):
        sampled = rng.integers(0, len(components), size=len(components))
        values = [
            credit
            for component_index in sampled.tolist()
            for credit in component_values[components[component_index]]
        ]
        draws[draw_index] = float(np.mean(values))
    confidence = _probability(
        confidence_level, "cluster confidence level", open_interval=True
    )
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, (tail, 1.0 - tail))
    return float(low), float(high)


def _ranking_summary(
    records: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    credits = _ranking_credits(records, scores)
    accuracy = float(np.mean([item["credit"] for item in credits]))
    cluster = _cluster_interval(
        credits,
        repetitions=repetitions,
        seed=seed,
        confidence_level=confidence_level,
    )
    component_outcomes = []
    for component_id in sorted({item["component_id"] for item in credits}):
        component_items = [
            item for item in credits if item["component_id"] == component_id
        ]
        component_outcomes.append(
            {
                "component_id": component_id,
                "target_count": len(component_items),
                "correct": sum(item["credit"] == 1.0 for item in component_items),
                "incorrect": sum(item["credit"] == 0.0 for item in component_items),
                "ties": sum(item["credit"] == 0.5 for item in component_items),
                "mean_pair_ranking_credit": float(
                    np.mean([item["credit"] for item in component_items])
                ),
                "target_ids": sorted(item["target_id"] for item in component_items),
            }
        )
    return {
        "target_count": len(credits),
        "component_count": len({item["component_id"] for item in credits}),
        "pair_ranking_accuracy": accuracy,
        "exact_outcome_counts": {
            "correct": sum(item["credit"] == 1.0 for item in credits),
            "incorrect": sum(item["credit"] == 0.0 for item in credits),
            "ties": sum(item["credit"] == 0.5 for item in credits),
        },
        "exact_target_outcomes": credits,
        "exact_component_outcomes": component_outcomes,
        "component_bootstrap_interval": list(cluster),
        "interval_method": "component_cluster_percentile",
        "interval_confidence_level": confidence_level,
        "conservative_upper_confidence_bound": cluster[1],
    }


def _classification_summary(
    records: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    *,
    threshold: float,
    repetitions: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    labels = _labels(records)
    predictions = (scores >= threshold).astype(float)
    outcomes = []
    for record, score, label, prediction in zip(
        records,
        scores.tolist(),
        labels.tolist(),
        predictions.tolist(),
    ):
        outcomes.append(
            {
                "pair_id": record["pair_id"],
                "target_id": record["target_id"],
                "component_id": record["component_id"],
                "role": record["role"],
                "score": float(score),
                "predicted_role": _ROLES[int(prediction)],
                "correct": bool(prediction == label),
                "credit": float(prediction == label),
            }
        )
    correct_count = sum(item["correct"] for item in outcomes)
    wilson = _wilson_interval(
        correct_count, len(outcomes), confidence_level=confidence_level
    )
    component_interval = _cluster_interval(
        outcomes,
        repetitions=repetitions,
        seed=seed,
        confidence_level=confidence_level,
    )
    return {
        "threshold": float(threshold),
        "balanced_accuracy": _balanced_accuracy(labels, predictions),
        "correct_pair_count": correct_count,
        "incorrect_pair_count": len(outcomes) - correct_count,
        "pair_count": len(outcomes),
        "exact_pair_outcomes": outcomes,
        "wilson_interval": list(wilson),
        "component_bootstrap_interval": list(component_interval),
        "interval_confidence_level": confidence_level,
        "conservative_upper_confidence_bound": max(wilson[1], component_interval[1]),
    }


def _combine_role_recovery(
    ranking: Mapping[str, Any], classification: Mapping[str, Any]
) -> dict[str, float]:
    return {
        "role_recovery_point_estimate": max(
            float(ranking["pair_ranking_accuracy"]),
            float(classification["balanced_accuracy"]),
        ),
        "conservative_primary_upper_confidence_bound": max(
            float(ranking["conservative_upper_confidence_bound"]),
            float(classification["conservative_upper_confidence_bound"]),
        ),
    }


def _structural_chance(evaluation: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the exact chance reference implied by one pair per role and target."""

    scores = np.full(len(evaluation), 0.5, dtype=np.float64)
    labels = _labels(evaluation)
    return {
        "method": "paired_role_structure_fixed_probability",
        "fixed_probability": 0.5,
        "balanced_accuracy": _balanced_accuracy(labels, (scores >= 0.5).astype(float)),
        "scores": scores,
    }


def _permuted_records(
    records: Sequence[Mapping[str, Any]], rng: np.random.Generator
) -> list[dict[str, Any]]:
    swaps = {
        component_id: bool(rng.integers(0, 2))
        for component_id in sorted({record["component_id"] for record in records})
    }
    permuted = []
    for record in records:
        copy_record = dict(record)
        copy_record["features"] = np.asarray(record["features"], dtype=float).tolist()
        if swaps[record["component_id"]]:
            copy_record["role"] = (
                "same_answer_nuisance"
                if record["role"] == "opposite_answer_candidate"
                else "opposite_answer_candidate"
            )
        permuted.append(copy_record)
    return permuted


def _permutation_summary(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    observed_statistic: float,
) -> dict[str, Any]:
    repetitions = int(config["permutation"]["repetitions"])
    rng = np.random.default_rng(int(config["permutation"]["seed"]))
    draws = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        permuted = _permuted_records(records, rng)
        frozen_suite = freeze_detector_suite(permuted, config)
        evaluation = [
            record for record in permuted if record["partition"] == "pilot_gate"
        ]
        model_statistics = []
        for frozen in frozen_suite["models"].values():
            scores = _score_frozen(frozen, evaluation)
            credits = _ranking_credits(evaluation, scores)
            ranking_accuracy = float(np.mean([item["credit"] for item in credits]))
            labels = _labels(evaluation)
            predictions = (scores >= float(frozen["threshold"])).astype(float)
            model_statistics.extend(
                [ranking_accuracy, _balanced_accuracy(labels, predictions)]
            )
        draws[index] = max(model_statistics)
    exceedances = int(np.count_nonzero(draws >= observed_statistic - 1e-12))
    return {
        "method": "within_component_role_swap_with_refit",
        "statistic": (
            "maximum_over_predeclared_detectors_pair_ranking_and_"
            "thresholded_balanced_accuracy"
        ),
        "multiplicity_control": "single_max_statistic_permutation_family",
        "repetitions": repetitions,
        "seed": int(config["permutation"]["seed"]),
        "null_mean": float(draws.mean()),
        "null_interval_95": [
            float(value) for value in np.quantile(draws, (0.025, 0.975))
        ],
        "p_value": float((exceedances + 1) / (repetitions + 1)),
    }


def _evaluate_frozen_model(
    records: Sequence[Mapping[str, Any]],
    frozen: Mapping[str, Any],
    *,
    repetitions: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    scores = _score_frozen(frozen, records)
    ranking = _ranking_summary(
        records,
        scores,
        repetitions=repetitions,
        seed=seed,
        confidence_level=confidence_level,
    )
    classification = _classification_summary(
        records,
        scores,
        threshold=float(frozen["threshold"]),
        repetitions=repetitions,
        seed=seed + 1,
        confidence_level=confidence_level,
    )
    return {
        **ranking,
        "balanced_accuracy": classification["balanced_accuracy"],
        "thresholded_classification": classification,
        **_combine_role_recovery(ranking, classification),
    }


def _aggregate_detector_summaries(
    models: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not models:
        raise NuisanceValidationError("detector suite is empty")
    point_family = max(
        models,
        key=lambda family: (
            float(models[family]["role_recovery_point_estimate"]),
            -list(models).index(family),
        ),
    )
    upper_family = max(
        models,
        key=lambda family: (
            float(models[family]["conservative_primary_upper_confidence_bound"]),
            -list(models).index(family),
        ),
    )
    aggregate = dict(models[point_family])
    aggregate.update(
        {
            "selected_model_for_point_estimate": point_family,
            "selected_model_for_upper_bound": upper_family,
            "role_recovery_point_estimate": max(
                float(model["role_recovery_point_estimate"])
                for model in models.values()
            ),
            "conservative_primary_upper_confidence_bound": max(
                float(model["conservative_primary_upper_confidence_bound"])
                for model in models.values()
            ),
        }
    )
    return aggregate


def _decision_reason_codes(
    *,
    detector: Mapping[str, Any],
    structural_chance: Mapping[str, Any],
    question_sensitivity: Mapping[str, Any],
    permutation: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> list[str]:
    reasons = []
    if float(detector["conservative_primary_upper_confidence_bound"]) > float(
        gate["maximum_primary_upper_confidence_bound"]
    ):
        reasons.append("primary_upper_bound_exceeds_ceiling")
    if question_sensitivity.get("status") != "evaluated":
        reasons.append("question_key_sensitivity_too_small")
    elif float(
        question_sensitivity["conservative_primary_upper_confidence_bound"]
    ) > float(gate["maximum_unseen_question_upper_confidence_bound"]):
        reasons.append("unseen_question_upper_bound_exceeds_ceiling")
    lift_ceiling = float(gate["maximum_point_lift_over_structural_chance"])
    if (
        float(detector["pair_ranking_accuracy"])
        - float(structural_chance["pair_ranking_accuracy"])
        > lift_ceiling
        or float(detector["role_recovery_point_estimate"])
        - float(structural_chance["role_recovery_point_estimate"])
        > lift_ceiling
    ):
        reasons.append("point_lift_over_structural_chance_exceeds_ceiling")
    if float(structural_chance["balanced_accuracy"]) > float(
        gate["maximum_structural_chance_balanced_accuracy"]
    ):
        reasons.append("structural_chance_exceeds_ceiling")
    if float(permutation["p_value"]) <= float(
        gate["minimum_permutation_p_value_exclusive"]
    ):
        reasons.append("permutation_test_detects_role_signal")
    return reasons


def evaluate_nuisance_records(
    records_value: Sequence[Mapping[str, Any]], config_value: Mapping[str, Any]
) -> dict[str, Any]:
    """Fit, calibrate, evaluate, and apply the locked fail-closed decision."""

    config = _validate_config(config_value)
    if config["schema_version"] == 2:
        raise NuisanceValidationError(
            "configuration v2 requires the dedicated shortcut evaluator"
        )
    records = _validate_records(records_value)
    frozen_suite = freeze_detector_suite(records, config)
    evaluation = [record for record in records if record["partition"] == "pilot_gate"]
    uncertainty = config["uncertainty"]
    familywise_confidence = float(uncertainty["confidence_level"])
    comparison_count = len(frozen_suite["models"]) * 2
    simultaneous_confidence = familywise_confidence
    model_summaries = {
        family: _evaluate_frozen_model(
            evaluation,
            frozen,
            repetitions=int(uncertainty["bootstrap_repetitions"]),
            seed=int(uncertainty["bootstrap_seed"]) + index * 20,
            confidence_level=simultaneous_confidence,
        )
        for index, (family, frozen) in enumerate(frozen_suite["models"].items())
    }
    detector = _aggregate_detector_summaries(model_summaries)
    chance = _structural_chance(evaluation)
    chance_scores = chance.pop("scores")
    chance_ranking = _ranking_summary(
        evaluation,
        chance_scores,
        repetitions=int(uncertainty["bootstrap_repetitions"]),
        seed=int(uncertainty["bootstrap_seed"]) + 1,
        confidence_level=familywise_confidence,
    )
    chance_classification = _classification_summary(
        evaluation,
        chance_scores,
        threshold=0.5,
        repetitions=int(uncertainty["bootstrap_repetitions"]),
        seed=int(uncertainty["bootstrap_seed"]) + 4,
        confidence_level=familywise_confidence,
    )
    chance.update(
        {
            **chance_ranking,
            "thresholded_classification": chance_classification,
            **_combine_role_recovery(chance_ranking, chance_classification),
        }
    )

    seen_keys = {
        record["key_sha256"]
        for record in records
        if record["partition"] in {"scorer_fit", "threshold_calibration"}
    }
    unseen_evaluation = [
        record for record in evaluation if record["key_sha256"] not in seen_keys
    ]
    unseen_components = {record["component_id"] for record in unseen_evaluation}
    minimum_unseen = int(config["protocol"]["minimum_unseen_question_components"])
    if len(unseen_components) >= minimum_unseen:
        unseen_models = {
            family: _evaluate_frozen_model(
                unseen_evaluation,
                frozen,
                repetitions=int(uncertainty["bootstrap_repetitions"]),
                seed=int(uncertainty["bootstrap_seed"]) + 10 + index * 20,
                confidence_level=simultaneous_confidence,
            )
            for index, (family, frozen) in enumerate(frozen_suite["models"].items())
        }
        unseen_aggregate = _aggregate_detector_summaries(unseen_models)
        question_sensitivity = {
            "status": "evaluated",
            **unseen_aggregate,
            "models": unseen_models,
        }
    else:
        question_sensitivity = {
            "status": "insufficient_components",
            "component_count": len(unseen_components),
            "minimum_component_count": minimum_unseen,
        }

    permutation = _permutation_summary(
        records, config, detector["role_recovery_point_estimate"]
    )
    reasons = _decision_reason_codes(
        detector=detector,
        structural_chance=chance,
        question_sensitivity=question_sensitivity,
        permutation=permutation,
        gate=config["gate"],
    )

    report = {
        "schema": "conflictbench.perception-nuisance-report.v1",
        "human_evaluation_used": False,
        "outcome_based_sample_selection": "forbidden",
        "feature_contract": {
            "name": config["feature_set"]["name"],
            "feature_names": list(FEATURE_NAMES),
            "forbidden_inputs": list(config["feature_set"]["forbidden_inputs"]),
            "scope": "container_and_stream_engineering_diagnostics_only",
        },
        "partition_integrity": {
            "fit": "scorer_fit",
            "threshold": "threshold_calibration",
            "evaluation": "pilot_gate",
            "component_isolation_verified": True,
            "media_isolation_verified": True,
        },
        "multiplicity_control": {
            "familywise_confidence_level": familywise_confidence,
            "comparison_count": comparison_count,
            "per_interval_confidence_level": simultaneous_confidence,
            "interval_method": "per_detector_then_maximum_upper_bound",
            "permutation_method": "single_max_statistic",
        },
        "frozen_detector": frozen_suite["models"]["ridge_linear"],
        "frozen_detector_suite": frozen_suite,
        "evaluation": {
            "detector": detector,
            "detector_suite": {"models": model_summaries, **detector},
            "structural_chance": chance,
            "permutation": permutation,
        },
        "question_key_sensitivity": question_sensitivity,
        "decision": {
            "nuisance_detection_status": "pass" if not reasons else "fail",
            "reason_codes": reasons,
        },
    }
    report["attestation_sha256"] = _canonical_digest(report)
    return report


def build_feature_records(
    pairs_value: Sequence[Mapping[str, Any]],
    media_diagnostics: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build outcome-isolated detector records from candidate identifiers."""

    if not isinstance(pairs_value, Sequence) or isinstance(pairs_value, (str, bytes)):
        raise NuisanceValidationError("candidate pairs must be a sequence")
    diagnostics = _mapping(media_diagnostics, "media diagnostics")
    records = []
    for index, pair_value in enumerate(pairs_value):
        pair = _mapping(pair_value, f"candidate pair {index}")
        target = _mapping(pair.get("target"), f"candidate pair {index} target")
        donor = _mapping(pair.get("donor"), f"candidate pair {index} donor")
        target_video_id = _nonempty_text(
            target.get("video_id"), f"candidate pair {index} target video ID"
        )
        donor_video_id = _nonempty_text(
            donor.get("video_id"), f"candidate pair {index} donor video ID"
        )
        if target_video_id not in diagnostics or donor_video_id not in diagnostics:
            raise NuisanceValidationError(
                "candidate pair media diagnostics are incomplete"
            )
        question_id = target.get("question_id")
        if isinstance(question_id, bool) or not isinstance(question_id, (int, str)):
            raise NuisanceValidationError("candidate target question ID is invalid")
        records.append(
            {
                "pair_id": _nonempty_text(pair.get("pair_id"), "candidate pair ID"),
                "target_id": f"{target_video_id}:{question_id}",
                "component_id": _nonempty_text(
                    pair.get("component_id"), "candidate component ID"
                ),
                "partition": pair.get("partition"),
                "key_sha256": _nonempty_text(
                    pair.get("key_sha256"), "candidate question-key digest"
                ),
                "target_video_id": target_video_id,
                "donor_video_id": donor_video_id,
                "role": pair.get("role"),
                "features": extract_metadata_features(
                    diagnostics[target_video_id], diagnostics[donor_video_id]
                ).tolist(),
            }
        )
    validated = _validate_records(records, require_all_partitions=False)
    return [
        {**record, "features": np.asarray(record["features"], dtype=float).tolist()}
        for record in validated
    ]


def build_post_edit_feature_records(
    rows_value: Sequence[Mapping[str, Any]],
    source_diagnostics: Mapping[str, Mapping[str, Any]],
    output_diagnostics: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build automated diagnostics from the final edited output for every pair."""

    if not isinstance(rows_value, Sequence) or isinstance(rows_value, (str, bytes)):
        raise NuisanceValidationError("post-edit rows must be a sequence")
    sources = _mapping(source_diagnostics, "source diagnostics")
    outputs = _mapping(output_diagnostics, "edited-output diagnostics")
    records = []
    seen_outputs: set[str] = set()
    expected_fields = {
        "pair_id",
        "target_id",
        "component_id",
        "partition",
        "key_sha256",
        "target_video_id",
        "output_video_id",
        "role",
    }
    for index, raw in enumerate(rows_value):
        row = _mapping(raw, f"post-edit row {index}")
        _exact_keys(row, expected_fields, f"post-edit row {index}")
        target_video_id = _nonempty_text(
            row.get("target_video_id"), f"post-edit row {index} target video ID"
        )
        output_video_id = _nonempty_text(
            row.get("output_video_id"), f"post-edit row {index} output video ID"
        )
        if output_video_id in seen_outputs:
            raise NuisanceValidationError("post-edit output IDs must be unique")
        seen_outputs.add(output_video_id)
        if target_video_id not in sources or output_video_id not in outputs:
            raise NuisanceValidationError("post-edit diagnostics are incomplete")
        records.append(
            {
                "pair_id": _nonempty_text(row.get("pair_id"), "post-edit pair ID"),
                "target_id": _nonempty_text(
                    row.get("target_id"), "post-edit target ID"
                ),
                "component_id": _nonempty_text(
                    row.get("component_id"), "post-edit component ID"
                ),
                "partition": row.get("partition"),
                "key_sha256": _nonempty_text(
                    row.get("key_sha256"), "post-edit question-key digest"
                ),
                "target_video_id": target_video_id,
                "donor_video_id": output_video_id,
                "role": row.get("role"),
                "features": extract_metadata_features(
                    sources[target_video_id], outputs[output_video_id]
                ).tolist(),
            }
        )
    validated = _validate_records(records)
    return [
        {**record, "features": np.asarray(record["features"], dtype=float).tolist()}
        for record in validated
    ]


def evaluate_post_edit_diagnostics(
    rows_value: Sequence[Mapping[str, Any]],
    source_diagnostics: Mapping[str, Mapping[str, Any]],
    output_diagnostics: Mapping[str, Mapping[str, Any]],
    config_value: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen automated gate to final edited-output diagnostics."""

    records = build_post_edit_feature_records(
        rows_value, source_diagnostics, output_diagnostics
    )
    report = evaluate_nuisance_records(records, config_value)
    report.pop("attestation_sha256")
    report["schema"] = "conflictbench.perception-post-edit-nuisance-report.v1"
    report["diagnostic_stage"] = "final_edited_outputs"
    report["feature_contract"] = {
        **report["feature_contract"],
        "scope": "final_output_container_and_stream_engineering_diagnostics_only",
    }
    report["attestation_sha256"] = _canonical_digest(report)
    return report


def _strict_identifier(value: Any, label: str) -> str:
    text = _nonempty_text(value, label)
    if (
        text != value
        or len(text) > 128
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for character in text
        )
    ):
        raise NuisanceValidationError(f"{label} must be a safe canonical identifier")
    return text


def _public_relative_path(value: Any, label: str) -> str:
    text = _nonempty_text(value, label)
    path = pathlib.PurePosixPath(text)
    if (
        text != value
        or path.is_absolute()
        or path.as_posix() != text
        or ".." in path.parts
        or "." in path.parts
    ):
        raise NuisanceValidationError(f"{label} must be a canonical relative path")
    return text


def _strict_json_output(raw_output: str, label: str) -> Mapping[str, Any]:
    if not isinstance(raw_output, str) or not raw_output:
        raise NuisanceValidationError(f"{label} raw output must be nonempty text")
    try:
        parsed = json.loads(
            raw_output,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise NuisanceValidationError(f"{label} raw output is not strict JSON") from exc
    parsed_mapping = _mapping(parsed, f"{label} parsed raw output")
    if raw_output != json.dumps(
        parsed_mapping,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ):
        raise NuisanceValidationError(f"{label} raw output is not canonical JSON")
    return parsed_mapping


def _run_pinned_json_tool_v2(
    *,
    executable_path: pathlib.Path,
    executable_sha256: str,
    configuration_path: pathlib.Path,
    configuration_sha256: str,
    envelope: Mapping[str, Any],
    label: str,
    timeout_seconds: int = 300,
    maximum_output_bytes: int = 25_000_000,
) -> Mapping[str, Any]:
    """Execute one hash-pinned JSON tool without inheriting mutable Python state."""

    executable = pathlib.Path(executable_path)
    if (
        executable.is_symlink()
        or not executable.is_file()
        or not os.access(executable, os.X_OK)
    ):
        raise NuisanceValidationError(
            f"{label} executable must be an executable non-symlink file"
        )
    expected_executable = _sha256_text(executable_sha256, f"{label} executable digest")
    if _sha256_path(executable) != expected_executable:
        raise NuisanceValidationError(f"{label} executable digest differs")

    configuration = pathlib.Path(configuration_path)
    if configuration.is_symlink() or not configuration.is_file():
        raise NuisanceValidationError(
            f"{label} configuration must be a regular non-symlink file"
        )
    if stat.S_IMODE(configuration.stat().st_mode) & 0o222:
        raise NuisanceValidationError(f"{label} configuration must be read-only")
    expected_configuration = _sha256_text(
        configuration_sha256, f"{label} configuration digest"
    )
    if _sha256_path(configuration) != expected_configuration:
        raise NuisanceValidationError(f"{label} configuration digest differs")

    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
        raise NuisanceValidationError(f"{label} timeout must be an integer")
    if timeout_seconds < 1 or timeout_seconds > 3600:
        raise NuisanceValidationError(f"{label} timeout is outside bounds")
    maximum_size = _positive_integer(
        maximum_output_bytes, f"{label} maximum output bytes"
    )
    try:
        payload = (
            json.dumps(
                envelope,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise NuisanceValidationError(f"{label} request is not strict JSON") from exc
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    try:
        completed = subprocess.run(
            [
                str(executable),
                "--configuration",
                str(configuration.resolve(strict=True)),
            ],
            input=payload,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NuisanceValidationError(f"{label} execution failed") from exc
    if completed.returncode != 0:
        raise NuisanceValidationError(f"{label} returned a failure status")
    if len(completed.stdout.encode("utf-8")) > maximum_size:
        raise NuisanceValidationError(f"{label} response is too large")
    response = _strict_json_output(
        completed.stdout.removesuffix("\n"), f"{label} response"
    )
    if _sha256_path(executable) != expected_executable:
        raise NuisanceValidationError(f"{label} executable changed during execution")
    if _sha256_path(configuration) != expected_configuration:
        raise NuisanceValidationError(f"{label} configuration changed during execution")
    return response


def _validate_replay_request_v2(value: Any) -> dict[str, Any]:
    request = _mapping(value, "replay-worker request")
    _exact_keys(request, {"schema", "task", "inputs"}, "replay-worker request")
    if request.get("schema") != REPLAY_REQUEST_SCHEMA_V2:
        raise NuisanceValidationError("replay-worker request schema differs")
    task = request.get("task")
    allowed_inputs = {
        "question_blind_audiovisual_mismatch": {"audio_path", "video_path"},
        "question_only_conflict": {"question", "options"},
        "source_choice": {"question", "options", "audio_path", "video_path"},
        "conflict_relation": {
            "question",
            "options",
            "audio_path",
            "video_path",
        },
    }
    if task not in allowed_inputs:
        raise NuisanceValidationError("replay-worker task differs")
    inputs = _mapping(request.get("inputs"), "replay-worker inputs")
    _exact_keys(inputs, allowed_inputs[str(task)], "replay-worker inputs")
    if task == "question_only_conflict":
        _nonempty_text(inputs.get("question"), "replay-worker question")
        options = inputs.get("options")
        if (
            not isinstance(options, list)
            or len(options) != 3
            or any(not isinstance(option, str) or not option for option in options)
        ):
            raise NuisanceValidationError(
                "replay-worker options must contain three nonempty strings"
            )
    else:
        for name in ("audio_path", "video_path"):
            path_text = _nonempty_text(inputs.get(name), f"replay-worker {name}")
            if pathlib.PurePath(path_text).name != path_text:
                raise NuisanceValidationError(
                    "replay-worker media paths must be opaque basenames"
                )
        if "question" in inputs:
            _nonempty_text(inputs.get("question"), "replay-worker question")
            options = inputs.get("options")
            if (
                not isinstance(options, list)
                or len(options) != 3
                or any(not isinstance(option, str) or not option for option in options)
            ):
                raise NuisanceValidationError(
                    "replay-worker options must contain three nonempty strings"
                )
    return copy.deepcopy(dict(request))


def run_replay_worker(
    spec: SubprocessReplayWorkerSpec,
    request_value: Mapping[str, Any],
    *,
    timeout_seconds: int = 300,
    working_directory: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Execute one isolated replay request with a pinned worker identity."""

    if type(spec) is not SubprocessReplayWorkerSpec:
        raise NuisanceValidationError("replay-worker specification is required")
    executable = pathlib.Path(spec.executable_path)
    if (
        executable.is_symlink()
        or not executable.is_file()
        or not os.access(executable, os.X_OK)
    ):
        raise NuisanceValidationError(
            "replay-worker executable must be an executable non-symlink file"
        )
    expected_executable = _sha256_text(
        spec.executable_sha256, "replay-worker executable digest"
    )
    if _sha256_path(executable) != expected_executable:
        raise NuisanceValidationError("replay-worker executable digest differs")
    authenticated_assets: dict[str, tuple[pathlib.Path, str]] = {}
    for name, path_value, digest_value in (
        ("configuration", spec.configuration_path, spec.configuration_sha256),
        ("model", spec.model_path, spec.model_sha256),
    ):
        path = pathlib.Path(path_value)
        if path.is_symlink() or not path.is_file():
            raise NuisanceValidationError(
                f"replay-worker {name} must be a regular non-symlink file"
            )
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise NuisanceValidationError(f"replay-worker {name} must be read-only")
        expected = _sha256_text(digest_value, f"replay-worker {name} digest")
        if _sha256_path(path) != expected:
            raise NuisanceValidationError(f"replay-worker {name} digest differs")
        authenticated_assets[name] = (path.resolve(strict=True), expected)
    bindings = {
        "configuration_sha256": authenticated_assets["configuration"][1],
        "implementation_sha256": expected_executable,
        "model_sha256": authenticated_assets["model"][1],
    }
    request = _validate_replay_request_v2(request_value)
    worker_cwd: pathlib.Path | None = None
    if working_directory is not None:
        candidate = pathlib.Path(working_directory)
        if candidate.is_symlink() or not candidate.is_dir():
            raise NuisanceValidationError(
                "replay-worker directory must be a non-symlink directory"
            )
        worker_cwd = candidate.resolve(strict=True)
        if request["task"] == "question_blind_audiovisual_mismatch":
            expected_names = {"audio.bin", "video.bin"}
            observed_names = {path.name for path in worker_cwd.iterdir()}
            if observed_names != expected_names:
                raise NuisanceValidationError(
                    "question-blind replay-worker directory membership differs"
                )
            for name in sorted(expected_names):
                media = worker_cwd / name
                mode = media.stat().st_mode
                if media.is_symlink() or not stat.S_ISREG(mode):
                    raise NuisanceValidationError(
                        "question-blind replay input is not a regular file"
                    )
                if stat.S_IMODE(mode) & 0o222:
                    raise NuisanceValidationError(
                        "question-blind replay input must be read-only"
                    )
    envelope = {
        "schema": REPLAY_REQUEST_SCHEMA_V2,
        "bindings": bindings,
        "request": request,
    }
    payload = (
        json.dumps(
            envelope,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
        raise NuisanceValidationError("replay-worker timeout must be an integer")
    if timeout_seconds < 1 or timeout_seconds > 3600:
        raise NuisanceValidationError("replay-worker timeout is outside bounds")
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
    try:
        completed = subprocess.run(
            [
                str(executable),
                "--configuration",
                str(authenticated_assets["configuration"][0]),
                "--model",
                str(authenticated_assets["model"][0]),
            ],
            input=payload,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=environment,
            cwd=worker_cwd,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NuisanceValidationError("replay-worker execution failed") from exc
    if completed.returncode != 0:
        raise NuisanceValidationError("replay-worker returned a failure status")
    if len(completed.stdout.encode("utf-8")) > 5_000_000:
        raise NuisanceValidationError("replay-worker response is too large")
    raw_response = completed.stdout.removesuffix("\n")
    response = _strict_json_output(raw_response, "replay-worker response")
    _exact_keys(
        response,
        {"schema", "bindings", "output"},
        "replay-worker response",
    )
    if response.get("schema") != REPLAY_RESPONSE_SCHEMA_V2:
        raise NuisanceValidationError("replay-worker response schema differs")
    if response.get("bindings") != bindings:
        raise NuisanceValidationError("replay-worker response binding differs")
    for name, (path, digest) in authenticated_assets.items():
        if _sha256_path(path) != digest:
            raise NuisanceValidationError(
                f"replay-worker {name} changed during execution"
            )
    return copy.deepcopy(dict(_mapping(response.get("output"), "replay-worker output")))


def _run_question_blind_replay_v2(
    spec: SubprocessReplayWorkerSpec,
    request_value: Mapping[str, Any],
    *,
    media_root: pathlib.Path,
) -> dict[str, Any]:
    """Stage anonymous media names before invoking the isolated worker."""

    request = _validate_question_blind_request_v2(
        request_value, "question-blind replay request"
    )
    root = pathlib.Path(media_root).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="conflictbench-replay-") as name:
        stage = pathlib.Path(name)
        for modality in ("audio", "video"):
            source = root.joinpath(
                *pathlib.PurePosixPath(request[modality]["relative_path"]).parts
            )
            try:
                resolved = source.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, ValueError) as exc:
                raise NuisanceValidationError(
                    "question-blind replay media path is unsafe"
                ) from exc
            if source.is_symlink() or not source.is_file():
                raise NuisanceValidationError(
                    "question-blind replay media is not a regular file"
                )
            if _sha256_path(source) != request[modality]["sha256"]:
                raise NuisanceValidationError(
                    "question-blind replay media digest differs"
                )
            destination = stage / f"{modality}.bin"
            shutil.copyfile(source, destination)
            destination.chmod(0o400)
        return run_replay_worker(
            spec,
            {
                "schema": REPLAY_REQUEST_SCHEMA_V2,
                "task": "question_blind_audiovisual_mismatch",
                "inputs": {
                    "audio_path": "audio.bin",
                    "video_path": "video.bin",
                },
            },
            working_directory=stage,
        )


def _validate_control_request_v2(value: Any, label: str) -> dict[str, Any]:
    """Validate one frozen, media-bound request without model outputs."""

    request = _mapping(value, label)
    _exact_keys(
        request,
        {"schema", "task", "question", "options", "audio", "video"},
        label,
    )
    if request.get("schema") != CONTROL_REQUEST_SCHEMA_V2:
        raise NuisanceValidationError(f"{label} schema differs")
    if request.get("task") not in {"source_choice", "conflict_relation"}:
        raise NuisanceValidationError(f"{label} task differs")
    _nonempty_text(request.get("question"), f"{label} question")
    options = request.get("options")
    if (
        not isinstance(options, list)
        or len(options) != 3
        or any(not isinstance(option, str) or not option for option in options)
    ):
        raise NuisanceValidationError(
            f"{label} options must contain three nonempty strings"
        )
    for modality in ("audio", "video"):
        media = _mapping(request.get(modality), f"{label} {modality}")
        _exact_keys(media, {"relative_path", "sha256"}, f"{label} {modality}")
        _public_relative_path(
            media.get("relative_path"), f"{label} {modality} relative path"
        )
        _sha256_text(media.get("sha256"), f"{label} {modality} digest")
    return copy.deepcopy(dict(request))


def _run_control_replay_v2(
    spec: SubprocessReplayWorkerSpec,
    request_value: Mapping[str, Any],
    *,
    media_root: pathlib.Path,
) -> dict[str, Any]:
    """Authenticate and stage one frozen control request under opaque names."""

    request = _validate_control_request_v2(request_value, "control replay request")
    root = pathlib.Path(media_root).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="conflictbench-control-replay-") as name:
        stage = pathlib.Path(name)
        for modality in ("audio", "video"):
            media = request[modality]
            source = root.joinpath(*pathlib.PurePosixPath(media["relative_path"]).parts)
            try:
                resolved = source.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, ValueError) as exc:
                raise NuisanceValidationError(
                    "control replay media path is unsafe"
                ) from exc
            if source.is_symlink() or not source.is_file():
                raise NuisanceValidationError(
                    "control replay media is not a regular file"
                )
            if _sha256_path(source) != media["sha256"]:
                raise NuisanceValidationError("control replay media digest differs")
            destination = stage / f"{modality}.bin"
            shutil.copyfile(source, destination)
            destination.chmod(0o400)
        return run_replay_worker(
            spec,
            {
                "schema": REPLAY_REQUEST_SCHEMA_V2,
                "task": request["task"],
                "inputs": {
                    "question": request["question"],
                    "options": request["options"],
                    "audio_path": "audio.bin",
                    "video_path": "video.bin",
                },
            },
            working_directory=stage,
        )


def _validate_raw_output_binding(
    output_value: Any, *, schema: str, payload_key: str, label: str
) -> tuple[Mapping[str, Any], Any]:
    output = _mapping(output_value, label)
    _exact_keys(
        output,
        {"schema", "raw_output", "raw_output_sha256", payload_key},
        label,
    )
    if output.get("schema") != schema:
        raise NuisanceValidationError(f"{label} schema differs")
    raw_output = output.get("raw_output")
    parsed = _strict_json_output(raw_output, label)
    _exact_keys(parsed, {payload_key}, f"{label} parsed raw output")
    observed_digest = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
    if (
        _sha256_text(output.get("raw_output_sha256"), f"{label} raw-output digest")
        != observed_digest
    ):
        raise NuisanceValidationError(f"{label} raw-output digest differs")
    if parsed[payload_key] != output.get(payload_key):
        raise NuisanceValidationError(f"{label} parsed value differs")
    return output, output[payload_key]


def _validate_mismatch_output_v2(value: Any, label: str) -> dict[str, Any]:
    output, score = _validate_raw_output_binding(
        value,
        schema=QUESTION_BLIND_OUTPUT_SCHEMA_V2,
        payload_key="mismatch_probability",
        label=label,
    )
    _exact_float(score, f"{label} mismatch probability", minimum=0.0, maximum=1.0)
    return copy.deepcopy(dict(output))


def _validate_relation_output_v2(value: Any, label: str) -> dict[str, Any]:
    output, prediction = _validate_raw_output_binding(
        value,
        schema=RELATION_OUTPUT_SCHEMA_V2,
        payload_key="prediction",
        label=label,
    )
    if prediction not in {"AGREE", "CONFLICT"}:
        raise NuisanceValidationError(f"{label} prediction differs")
    return copy.deepcopy(dict(output))


def _validate_choice_output_v2(value: Any, label: str) -> dict[str, Any]:
    output, probabilities_value = _validate_raw_output_binding(
        value,
        schema=CHOICE_OUTPUT_SCHEMA_V2,
        payload_key="choice_probabilities",
        label=label,
    )
    probabilities = _mapping(probabilities_value, f"{label} choice probabilities")
    _exact_keys(probabilities, {"A", "B", "C"}, f"{label} choice probabilities")
    values = [
        _exact_float(
            probabilities[choice],
            f"{label} probability {choice}",
            minimum=0.0,
            maximum=1.0,
        )
        for choice in ("A", "B", "C")
    ]
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise NuisanceValidationError(f"{label} probabilities must sum to one")
    if values.count(max(values)) != 1:
        raise NuisanceValidationError(f"{label} top choice must be unique")
    return copy.deepcopy(dict(output))


def _validate_question_blind_request_v2(value: Any, label: str) -> dict[str, Any]:
    request = _mapping(value, label)
    _exact_keys(request, {"schema", "audio", "video"}, label)
    if request.get("schema") != QUESTION_BLIND_REQUEST_SCHEMA_V2:
        raise NuisanceValidationError(f"{label} schema differs")
    for modality in ("audio", "video"):
        media = _mapping(request.get(modality), f"{label} {modality}")
        _exact_keys(media, {"relative_path", "sha256"}, f"{label} {modality}")
        _public_relative_path(
            media.get("relative_path"), f"{label} {modality} relative path"
        )
        _sha256_text(media.get("sha256"), f"{label} {modality} digest")
    return copy.deepcopy(dict(request))


def _validate_question_blind_records_v2(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise NuisanceValidationError("question-blind records must be a nonempty list")
    records = []
    record_ids: set[str] = set()
    request_digests: set[str] = set()
    component_labels: dict[str, set[str]] = {}
    for index, raw in enumerate(value):
        label = f"question-blind record {index}"
        record = _mapping(raw, label)
        _exact_keys(
            record,
            {
                "record_id",
                "component_id",
                "expected_relation",
                "request",
                "request_sha256",
                "primary_output",
                "replay_output",
            },
            label,
        )
        record_id = _strict_identifier(record.get("record_id"), f"{label} ID")
        if record_id in record_ids:
            raise NuisanceValidationError("question-blind record IDs must be unique")
        record_ids.add(record_id)
        component_id = _strict_identifier(
            record.get("component_id"), f"{label} component ID"
        )
        expected_relation = record.get("expected_relation")
        if expected_relation not in {"matched", "mismatched"}:
            raise NuisanceValidationError(f"{label} expected relation differs")
        component_labels.setdefault(component_id, set()).add(expected_relation)
        request = _validate_question_blind_request_v2(
            record.get("request"), f"{label} request"
        )
        request_sha256 = _sha256_text(
            record.get("request_sha256"), f"{label} request digest"
        )
        if _canonical_digest(request) != request_sha256:
            raise NuisanceValidationError(f"{label} request digest differs")
        if request_sha256 in request_digests:
            raise NuisanceValidationError(
                "question-blind requests must have unique digests"
            )
        request_digests.add(request_sha256)
        primary = _validate_mismatch_output_v2(
            record.get("primary_output"), f"{label} primary output"
        )
        replay = _validate_mismatch_output_v2(
            record.get("replay_output"), f"{label} replay output"
        )
        if primary != replay:
            raise NuisanceValidationError(f"{label} recorded replay differs")
        records.append(copy.deepcopy(dict(record)))
    if len(component_labels) < 2:
        raise NuisanceValidationError(
            "question-blind evaluation needs at least two components"
        )
    if any(labels != {"matched", "mismatched"} for labels in component_labels.values()):
        raise NuisanceValidationError(
            "each question-blind component must contain both relations"
        )
    return records


def _validate_record_population(
    records: Any,
    *,
    name: str,
    required_fields: set[str],
    validator: Callable[[Mapping[str, Any], str], None],
) -> list[dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise NuisanceValidationError(f"{name} records must be a nonempty list")
    normalized = []
    record_ids: set[str] = set()
    for index, raw in enumerate(records):
        label = f"{name} record {index}"
        record = _mapping(raw, label)
        _exact_keys(record, required_fields, label)
        record_id = _strict_identifier(record.get("record_id"), f"{label} ID")
        if record_id in record_ids:
            raise NuisanceValidationError(f"{name} record IDs must be unique")
        record_ids.add(record_id)
        _strict_identifier(record.get("component_id"), f"{label} component ID")
        validator(record, label)
        normalized.append(copy.deepcopy(dict(record)))
    if len({record["component_id"] for record in normalized}) < 2:
        raise NuisanceValidationError(f"{name} needs at least two components")
    return normalized


def _validate_remux_record(record: Mapping[str, Any], label: str) -> None:
    if record.get("gold_answer") not in {"A", "B", "C"}:
        raise NuisanceValidationError(f"{label} gold answer differs")
    for prefix in ("clean", "remux"):
        request = _validate_control_request_v2(
            record.get(f"{prefix}_request"), f"{label} {prefix} request"
        )
        if request["task"] != "source_choice":
            raise NuisanceValidationError(f"{label} {prefix} request task differs")
        if record.get(f"{prefix}_request_sha256") != _canonical_digest(request):
            raise NuisanceValidationError(f"{label} {prefix} request digest differs")
    _validate_choice_output_v2(record.get("clean_output"), f"{label} clean output")
    _validate_choice_output_v2(record.get("remux_output"), f"{label} remux output")


def _validate_question_only_record(record: Mapping[str, Any], label: str) -> None:
    if record.get("gold_relation") not in {"AGREE", "CONFLICT"}:
        raise NuisanceValidationError(f"{label} gold relation differs")
    request = _validate_replay_request_v2(record.get("request"))
    if request["task"] != "question_only_conflict":
        raise NuisanceValidationError(f"{label} request task differs")
    if record.get("request_sha256") != _canonical_digest(request):
        raise NuisanceValidationError(f"{label} request digest differs")
    primary = _validate_relation_output_v2(
        record.get("primary_output"), f"{label} primary output"
    )
    replay = _validate_relation_output_v2(
        record.get("replay_output"), f"{label} replay output"
    )
    if primary != replay:
        raise NuisanceValidationError(f"{label} recorded replay differs")


def _validate_orientation_record(record: Mapping[str, Any], label: str) -> None:
    if record.get("gold_relation") not in {"AGREE", "CONFLICT"}:
        raise NuisanceValidationError(f"{label} gold relation differs")
    for prefix in ("audio_over_video", "video_over_audio"):
        request = _validate_control_request_v2(
            record.get(f"{prefix}_request"), f"{label} {prefix} request"
        )
        if request["task"] != "conflict_relation":
            raise NuisanceValidationError(f"{label} {prefix} request task differs")
        if record.get(f"{prefix}_request_sha256") != _canonical_digest(request):
            raise NuisanceValidationError(f"{label} {prefix} request digest differs")
    _validate_relation_output_v2(
        record.get("audio_over_video_output"), f"{label} audio-over-video output"
    )
    _validate_relation_output_v2(
        record.get("video_over_audio_output"), f"{label} video-over-audio output"
    )


def _validate_permutation_record(record: Mapping[str, Any], label: str) -> None:
    if record.get("gold_relation") not in {"AGREE", "CONFLICT"}:
        raise NuisanceValidationError(f"{label} gold relation differs")
    requests = _mapping(record.get("requests"), f"{label} requests")
    request_sha256s = _mapping(
        record.get("request_sha256s"), f"{label} request digests"
    )
    outputs = _mapping(record.get("outputs"), f"{label} outputs")
    _exact_keys(requests, {"0", "1", "2"}, f"{label} requests")
    _exact_keys(request_sha256s, {"0", "1", "2"}, f"{label} request digests")
    _exact_keys(outputs, {"0", "1", "2"}, f"{label} outputs")
    for permutation in ("0", "1", "2"):
        request = _validate_control_request_v2(
            requests[permutation], f"{label} permutation {permutation} request"
        )
        if request["task"] != "conflict_relation":
            raise NuisanceValidationError(
                f"{label} permutation {permutation} request task differs"
            )
        if request_sha256s[permutation] != _canonical_digest(request):
            raise NuisanceValidationError(
                f"{label} permutation {permutation} request digest differs"
            )
        _validate_relation_output_v2(
            outputs[permutation], f"{label} permutation {permutation} output"
        )


def _validate_shuffled_record(record: Mapping[str, Any], label: str) -> None:
    if record.get("gold_relation") not in {"AGREE", "CONFLICT"}:
        raise NuisanceValidationError(f"{label} gold relation differs")
    for prefix in ("aligned", "shuffled"):
        request = _validate_control_request_v2(
            record.get(f"{prefix}_request"), f"{label} {prefix} request"
        )
        if request["task"] != "conflict_relation":
            raise NuisanceValidationError(f"{label} {prefix} request task differs")
        if record.get(f"{prefix}_request_sha256") != _canonical_digest(request):
            raise NuisanceValidationError(f"{label} {prefix} request digest differs")
    _validate_relation_output_v2(
        record.get("aligned_output"), f"{label} aligned output"
    )
    _validate_relation_output_v2(
        record.get("shuffled_output"), f"{label} shuffled output"
    )


def validate_shortcut_output_v2(
    value: Any,
    *,
    expected_configuration_sha256: str,
    expected_media_set_sha256: str,
    expected_gate_implementation_bundle_sha256: str,
    expected_shortcut_preregistration_sha256: str,
) -> dict[str, Any]:
    """Validate the completed, hash-bound v2 model output without using scores."""

    shortcut = _mapping(value, "shortcut output")
    _exact_keys(
        shortcut,
        {
            "schema",
            "status",
            "shortcut_preregistration_sha256",
            "input_digests",
            "question_only_conflict",
            "question_blind_audiovisual_mismatch",
            "remux_noninferiority",
            "orientation_symmetry",
            "option_permutation_equivalence",
            "shuffled_question",
            "attestation_sha256",
        },
        "shortcut output",
    )
    if shortcut.get("schema") != SHORTCUT_OUTPUT_SCHEMA_V2:
        raise NuisanceValidationError("shortcut output schema differs")
    if shortcut.get("status") != "complete_replay_recorded":
        raise NuisanceValidationError("shortcut output is incomplete")
    if shortcut.get("shortcut_preregistration_sha256") != _sha256_text(
        expected_shortcut_preregistration_sha256,
        "expected shortcut-preregistration digest",
    ):
        raise NuisanceValidationError("shortcut output preregistration binding differs")
    attestation = _sha256_text(
        shortcut.get("attestation_sha256"), "shortcut-output attestation"
    )
    unsigned = dict(shortcut)
    unsigned.pop("attestation_sha256")
    if _canonical_digest(unsigned) != attestation:
        raise NuisanceValidationError("shortcut-output attestation differs")

    digests = _mapping(shortcut.get("input_digests"), "shortcut output digests")
    digest_keys = {
        "configuration_sha256",
        "raw_conflict_index_sha256",
        "media_set_sha256",
        "primary_system_sha256",
        "semantic_gate_sha256",
        "semantic_fit_output_sha256",
        "question_blind_model_sha256",
        "question_blind_implementation_sha256",
        "question_blind_configuration_sha256",
        "gate_implementation_bundle_sha256",
    }
    _exact_keys(digests, digest_keys, "shortcut output digests")
    for key in digest_keys:
        _sha256_text(digests.get(key), f"shortcut output {key}")
    expected_bindings = {
        "configuration_sha256": _sha256_text(
            expected_configuration_sha256, "expected configuration digest"
        ),
        "media_set_sha256": _sha256_text(
            expected_media_set_sha256, "expected media-set digest"
        ),
        "gate_implementation_bundle_sha256": _sha256_text(
            expected_gate_implementation_bundle_sha256,
            "expected gate implementation-bundle digest",
        ),
    }
    for key, expected in expected_bindings.items():
        if digests.get(key) != expected:
            raise NuisanceValidationError(f"shortcut output {key} binding differs")

    question_only = _mapping(
        shortcut.get("question_only_conflict"), "question-only conflict"
    )
    _exact_keys(question_only, {"transform", "records"}, "question-only conflict")
    if question_only.get("transform") != "remove_audio_and_video":
        raise NuisanceValidationError("question-only transform differs")
    question_only_records = _validate_record_population(
        question_only.get("records"),
        name="question only",
        required_fields={
            "record_id",
            "component_id",
            "gold_relation",
            "request",
            "request_sha256",
            "primary_output",
            "replay_output",
        },
        validator=_validate_question_only_record,
    )

    question_blind = _mapping(
        shortcut.get("question_blind_audiovisual_mismatch"),
        "question-blind audiovisual mismatch",
    )
    _exact_keys(
        question_blind,
        {"transform", "threshold", "threshold_record_sha256", "records"},
        "question-blind audiovisual mismatch",
    )
    if question_blind.get("transform") != "remove_question_and_options":
        raise NuisanceValidationError("question-blind transform differs")
    _exact_float(
        question_blind.get("threshold"),
        "question-blind threshold",
        minimum=0.0,
        maximum=1.0,
    )
    _sha256_text(
        question_blind.get("threshold_record_sha256"),
        "question-blind threshold-record digest",
    )
    question_blind_records = _validate_question_blind_records_v2(
        question_blind.get("records")
    )

    remux = _mapping(shortcut.get("remux_noninferiority"), "remux noninferiority")
    _exact_keys(
        remux,
        {"transform", "epsilon_p_gold", "epsilon_source", "records"},
        "remux noninferiority",
    )
    if remux.get("transform") != "container_remux_only":
        raise NuisanceValidationError("remux transform differs")
    _exact_float(
        remux.get("epsilon_p_gold"),
        "remux p_gold epsilon",
        minimum=0.0,
        maximum=1.0,
    )
    epsilon_source = _mapping(remux.get("epsilon_source"), "remux epsilon source")
    _exact_keys(
        epsilon_source,
        {"quantile", "semantic_fit_output_sha256"},
        "remux epsilon source",
    )
    if (
        _exact_float(
            epsilon_source.get("quantile"),
            "remux epsilon quantile",
            minimum=0.0,
            maximum=1.0,
        )
        != 0.95
    ):
        raise NuisanceValidationError("remux epsilon quantile must equal 0.95")
    if (
        _sha256_text(
            epsilon_source.get("semantic_fit_output_sha256"),
            "remux semantic-fit output digest",
        )
        != digests["semantic_fit_output_sha256"]
    ):
        raise NuisanceValidationError("remux epsilon source binding differs")
    remux_records = _validate_record_population(
        remux.get("records"),
        name="remux",
        required_fields={
            "record_id",
            "component_id",
            "gold_answer",
            "clean_request",
            "clean_request_sha256",
            "clean_output",
            "remux_request",
            "remux_request_sha256",
            "remux_output",
        },
        validator=_validate_remux_record,
    )

    orientation = _mapping(shortcut.get("orientation_symmetry"), "orientation symmetry")
    _exact_keys(orientation, {"transform", "records"}, "orientation symmetry")
    if orientation.get("transform") != "reverse_modality_precedence":
        raise NuisanceValidationError("orientation transform differs")
    orientation_records = _validate_record_population(
        orientation.get("records"),
        name="orientation",
        required_fields={
            "record_id",
            "component_id",
            "gold_relation",
            "audio_over_video_request",
            "audio_over_video_request_sha256",
            "audio_over_video_output",
            "video_over_audio_request",
            "video_over_audio_request_sha256",
            "video_over_audio_output",
        },
        validator=_validate_orientation_record,
    )

    permutation = _mapping(
        shortcut.get("option_permutation_equivalence"),
        "option-permutation equivalence",
    )
    _exact_keys(permutation, {"transform", "records"}, "option-permutation equivalence")
    if permutation.get("transform") != "cyclic_option_rotation":
        raise NuisanceValidationError("option-permutation transform differs")
    permutation_records = _validate_record_population(
        permutation.get("records"),
        name="option permutation",
        required_fields={
            "record_id",
            "component_id",
            "gold_relation",
            "requests",
            "request_sha256s",
            "outputs",
        },
        validator=_validate_permutation_record,
    )

    shuffled = _mapping(shortcut.get("shuffled_question"), "shuffled question")
    _exact_keys(shuffled, {"transform", "records"}, "shuffled question")
    if shuffled.get("transform") != "cross_component_question_derangement":
        raise NuisanceValidationError("shuffled-question transform differs")
    shuffled_records = _validate_record_population(
        shuffled.get("records"),
        name="shuffled question",
        required_fields={
            "record_id",
            "component_id",
            "gold_relation",
            "aligned_request",
            "aligned_request_sha256",
            "aligned_output",
            "shuffled_request",
            "shuffled_request_sha256",
            "shuffled_output",
        },
        validator=_validate_shuffled_record,
    )

    expected_inventory = {
        record["record_id"]: record["component_id"] for record in question_blind_records
    }
    for name, records in (
        ("question only", question_only_records),
        ("remux", remux_records),
        ("orientation", orientation_records),
        ("option permutation", permutation_records),
        ("shuffled question", shuffled_records),
    ):
        observed_inventory = {
            record["record_id"]: record["component_id"] for record in records
        }
        if observed_inventory != expected_inventory:
            raise NuisanceValidationError(
                f"{name} record inventory differs from the question-blind inventory"
            )
    return copy.deepcopy(dict(shortcut))


def validate_shortcut_output_against_preregistration_v2(
    output_value: Any,
    preregistration_value: Any,
    *,
    expected_shortcut_preregistration_sha256: str,
    expected_configuration_sha256: str,
    expected_gate_implementation_bundle_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind completed outputs to the prior output-free v2 registration."""

    preregistration = validate_shortcut_preregistration_v2(
        preregistration_value,
        expected_configuration_sha256=expected_configuration_sha256,
        expected_gate_implementation_bundle_sha256=(
            expected_gate_implementation_bundle_sha256
        ),
    )
    expected_preregistration_sha256 = _sha256_text(
        expected_shortcut_preregistration_sha256,
        "expected shortcut-preregistration digest",
    )
    if _canonical_digest(preregistration) != expected_preregistration_sha256:
        raise NuisanceValidationError("shortcut preregistration content digest differs")
    output = validate_shortcut_output_v2(
        output_value,
        expected_configuration_sha256=expected_configuration_sha256,
        expected_media_set_sha256=preregistration["inventory"][
            "media_inventory_sha256"
        ],
        expected_gate_implementation_bundle_sha256=(
            expected_gate_implementation_bundle_sha256
        ),
        expected_shortcut_preregistration_sha256=expected_preregistration_sha256,
    )
    observed_inventory = sorted(
        [
            {
                "record_id": record["record_id"],
                "component_id": record["component_id"],
            }
            for record in output["question_blind_audiovisual_mismatch"]["records"]
        ],
        key=lambda record: record["record_id"],
    )
    if observed_inventory != preregistration["inventory"]["records"]:
        raise NuisanceValidationError(
            "shortcut output inventory differs from the preregistration"
        )
    observed_control_plan = project_shortcut_control_plan_v2(output)
    if observed_control_plan != preregistration["control_plan"]:
        raise NuisanceValidationError(
            "shortcut output control plan differs from the preregistration"
        )
    output_digests = output["input_digests"]
    preregistration_digests = preregistration["input_digests"]
    required_matches = {
        "configuration_sha256": "configuration_sha256",
        "raw_conflict_index_sha256": "raw_conflict_index_sha256",
        "primary_system_sha256": "source_only_output_sha256",
        "semantic_gate_sha256": "semantic_gate_output_sha256",
        "gate_implementation_bundle_sha256": ("gate_implementation_bundle_sha256"),
    }
    for output_key, preregistration_key in required_matches.items():
        if output_digests[output_key] != preregistration_digests[preregistration_key]:
            raise NuisanceValidationError(
                f"shortcut output {output_key} differs from the preregistration"
            )
    expected_worker = preregistration["replay_worker"]
    observed_worker = {
        "implementation_sha256": output_digests["question_blind_implementation_sha256"],
        "configuration_sha256": output_digests["question_blind_configuration_sha256"],
        "model_sha256": output_digests["question_blind_model_sha256"],
    }
    if observed_worker != expected_worker:
        raise NuisanceValidationError(
            "shortcut output replay-worker binding differs from the preregistration"
        )
    return output, preregistration


def _component_mean_interval(
    values: Sequence[tuple[str, float]],
    *,
    repetitions: int,
    seed: int,
    confidence_level: float,
    bounded_unit_interval: bool = False,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for component_id, value in values:
        if not math.isfinite(value):
            raise NuisanceValidationError("component outcome is not finite")
        grouped.setdefault(component_id, []).append(float(value))
    components = sorted(grouped)
    if len(components) < 2:
        raise NuisanceValidationError(
            "component interval needs at least two components"
        )
    component_means = np.asarray(
        [float(np.mean(grouped[component])) for component in components],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(
        0,
        len(components),
        size=(repetitions, len(components)),
    )
    draws = np.mean(component_means[sampled_indices], axis=1)
    tail = 1.0 - confidence_level
    lower = float(np.quantile(draws, tail))
    upper = float(np.quantile(draws, confidence_level))
    if bounded_unit_interval and (
        np.any(component_means < 0.0) or np.any(component_means > 1.0)
    ):
        raise NuisanceValidationError("bounded component outcomes leave [0, 1]")
    result = {
        "component_count": len(components),
        "point_estimate": float(np.mean(component_means)),
        "one_sided_lower_confidence_bound": lower,
        "one_sided_upper_confidence_bound": upper,
        "confidence_level": confidence_level,
        "method": "analysis_component_cluster_percentile",
    }
    return result


def _component_balanced_accuracies(
    records: Sequence[Mapping[str, Any]],
    *,
    gold_key: str,
    prediction: Callable[[Mapping[str, Any]], str],
    labels: tuple[str, str],
    name: str,
) -> list[tuple[str, float]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["component_id"], []).append(record)
    values = []
    for component_id, component_records in sorted(grouped.items()):
        observed_labels = {record[gold_key] for record in component_records}
        if observed_labels != set(labels):
            raise NuisanceValidationError(
                f"each {name} component must contain both classes"
            )
        recalls = []
        for expected in labels:
            matching = [
                record for record in component_records if record[gold_key] == expected
            ]
            recalls.append(
                float(np.mean([prediction(record) == expected for record in matching]))
            )
        values.append((component_id, float(np.mean(recalls))))
    return values


def _classification_component_balanced_accuracies(
    outcomes: Sequence[Mapping[str, Any]],
) -> list[tuple[str, float]]:
    """Compute class-balanced accuracy within each analysis component."""

    return _component_balanced_accuracies(
        outcomes,
        gold_key="role",
        prediction=lambda outcome: str(outcome["predicted_role"]),
        labels=("same_answer_nuisance", "opposite_answer_candidate"),
        name="metadata classification",
    )


def _metadata_upper_bound_v2(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    endpoint_confidence_level: float | None = None,
) -> dict[str, Any]:
    validated = _validate_records(records)
    frozen_suite = freeze_detector_suite(validated, config)
    evaluation = [
        record
        for record in validated
        if record["partition"] == config["protocol"]["evaluation_partition"]
    ]
    protocol = config["shortcut_protocol"]
    confidence = (
        float(protocol["confidence_level"])
        if endpoint_confidence_level is None
        else endpoint_confidence_level
    )
    comparisons = len(frozen_suite["models"]) * 2
    simultaneous_confidence = 1.0 - (1.0 - confidence) / comparisons
    summaries = {}
    for family_index, (family, frozen) in enumerate(frozen_suite["models"].items()):
        scores = _score_frozen(frozen, evaluation)
        ranking_values = [
            (outcome["component_id"], float(outcome["credit"]))
            for outcome in _ranking_credits(evaluation, scores)
        ]
        predictions = (scores >= float(frozen["threshold"])).astype(float)
        classification_outcomes = [
            {
                "component_id": record["component_id"],
                "role": record["role"],
                "predicted_role": _ROLES[int(prediction)],
            }
            for record, prediction in zip(evaluation, predictions.tolist())
        ]
        classification_values = _classification_component_balanced_accuracies(
            classification_outcomes
        )
        summaries[family] = {
            "pair_ranking": _component_mean_interval(
                ranking_values,
                repetitions=int(protocol["bootstrap_repetitions"]),
                seed=int(protocol["bootstrap_seed"]) + family_index * 2,
                confidence_level=simultaneous_confidence,
                bounded_unit_interval=True,
            ),
            "thresholded_balanced_accuracy": _component_mean_interval(
                classification_values,
                repetitions=int(protocol["bootstrap_repetitions"]),
                seed=int(protocol["bootstrap_seed"]) + family_index * 2 + 1,
                confidence_level=simultaneous_confidence,
                bounded_unit_interval=True,
            ),
        }
    candidates = [
        (family, metric, summary["one_sided_upper_confidence_bound"])
        for family, family_summary in summaries.items()
        for metric, summary in family_summary.items()
    ]
    selected_family, selected_metric, upper = max(candidates, key=lambda item: item[2])
    return {
        "models": summaries,
        "comparison_count": comparisons,
        "simultaneous_confidence_level": simultaneous_confidence,
        "selected_family": selected_family,
        "selected_metric": selected_metric,
        "one_sided_upper_confidence_bound": float(upper),
    }


def _normalize_metadata_records_v2(
    records_value: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the exact strict-JSON metadata snapshot used by v2 inference."""

    validated = _validate_records(records_value)
    return [
        {
            **record,
            "features": np.asarray(record["features"], dtype=float).tolist(),
        }
        for record in validated
    ]


def _validate_metadata_record_inventory_v2(
    records_value: Sequence[Mapping[str, Any]],
    shortcut: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = _normalize_metadata_records_v2(records_value)
    evaluation_partition = config["protocol"]["evaluation_partition"]
    observed = {
        record["pair_id"]: record["component_id"]
        for record in records
        if record["partition"] == evaluation_partition
    }
    expected = {
        record["record_id"]: record["component_id"]
        for record in shortcut["question_blind_audiovisual_mismatch"]["records"]
    }
    if observed != expected:
        raise NuisanceValidationError(
            "metadata record inventory differs from the frozen shortcut inventory"
        )
    minimum_components = int(config["protocol"]["minimum_unseen_question_components"])
    if len(set(observed.values())) < minimum_components:
        raise NuisanceValidationError(
            f"v2 shortcut evaluation requires at least {minimum_components} components"
        )
    return records


def _choice_metrics(record: Mapping[str, Any], key: str) -> tuple[float, float]:
    output = record[key]
    probabilities = output["choice_probabilities"]
    gold = record["gold_answer"]
    predicted = max(("A", "B", "C"), key=lambda choice: probabilities[choice])
    return float(predicted == gold), float(probabilities[gold])


def _paired_balanced_difference(
    records: Sequence[Mapping[str, Any]],
    *,
    first_key: str,
    second_key: str,
    repetitions: int,
    seed: int,
    confidence_level: float,
    name: str,
) -> dict[str, Any]:
    first = dict(
        _component_balanced_accuracies(
            records,
            gold_key="gold_relation",
            prediction=lambda record: record[first_key]["prediction"],
            labels=("AGREE", "CONFLICT"),
            name=name,
        )
    )
    second = dict(
        _component_balanced_accuracies(
            records,
            gold_key="gold_relation",
            prediction=lambda record: record[second_key]["prediction"],
            labels=("AGREE", "CONFLICT"),
            name=name,
        )
    )
    return _component_mean_interval(
        [
            (component_id, first[component_id] - second[component_id])
            for component_id in sorted(first)
        ],
        repetitions=repetitions,
        seed=seed,
        confidence_level=confidence_level,
    )


def _shortcut_reason_codes_v2(
    *,
    metadata_summary: Mapping[str, Any],
    question_only_summary: Mapping[str, Any],
    question_blind_summary: Mapping[str, Any],
    remux_summary: Mapping[str, Any],
    orientation_summary: Mapping[str, Any],
    permutation_summary: Mapping[str, Any],
    shuffled_summary: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> list[str]:
    reasons = []
    if (
        metadata_summary["one_sided_upper_confidence_bound"]
        >= gate["maximum_metadata_balanced_accuracy_ucb_exclusive"]
    ):
        reasons.append("metadata_ucb_not_below_0_55")
    if (
        question_only_summary["one_sided_upper_confidence_bound"]
        >= gate["maximum_question_only_balanced_accuracy_ucb_exclusive"]
    ):
        reasons.append("question_only_ucb_not_below_0_55")
    if (
        question_blind_summary["one_sided_upper_confidence_bound"]
        >= gate["maximum_question_blind_balanced_accuracy_ucb_exclusive"]
    ):
        reasons.append("question_blind_ucb_not_below_0_55")
    if (
        remux_summary["accuracy_difference"]["one_sided_lower_confidence_bound"]
        <= gate["minimum_remux_accuracy_difference_exclusive"]
    ):
        reasons.append("remux_accuracy_noninferiority_failed")
    if remux_summary["p_gold_difference"]["one_sided_lower_confidence_bound"] <= -float(
        remux_summary["epsilon_p_gold"]
    ):
        reasons.append("remux_p_gold_noninferiority_failed")
    equivalence_low, equivalence_high = gate["equivalence_interval"]
    orientation_bounds = orientation_summary["audio_over_video_minus_video_over_audio"]
    if (
        orientation_bounds["one_sided_lower_confidence_bound"] < equivalence_low
        or orientation_bounds["one_sided_upper_confidence_bound"] > equivalence_high
    ):
        reasons.append("orientation_equivalence_failed")
    if any(
        summary["one_sided_lower_confidence_bound"] < equivalence_low
        or summary["one_sided_upper_confidence_bound"] > equivalence_high
        for summary in permutation_summary.values()
    ):
        reasons.append("option_permutation_equivalence_failed")
    if (
        shuffled_summary["shuffled"]["one_sided_upper_confidence_bound"]
        >= gate["maximum_shuffled_question_balanced_accuracy_ucb_exclusive"]
    ):
        reasons.append("shuffled_question_ucb_not_below_0_55")
    if (
        shuffled_summary["aligned_minus_shuffled"]["one_sided_lower_confidence_bound"]
        < gate["minimum_aligned_minus_shuffled_lcb_inclusive"]
    ):
        reasons.append("aligned_minus_shuffled_lcb_below_0_10")
    return reasons


def _build_authenticated_shortcut_records_v2(
    *,
    shortcut: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    expected_shortcut_preregistration_sha256: str,
    expected_shortcut_output_sha256: str,
    metadata_records: Sequence[Mapping[str, Any]],
    question_only_records: Sequence[Mapping[str, Any]],
    question_blind_records: Sequence[Mapping[str, Any]],
    remux_records: Sequence[Mapping[str, Any]],
    orientation_records: Sequence[Mapping[str, Any]],
    permutation_records: Sequence[Mapping[str, Any]],
    shuffled_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    snapshot = copy.deepcopy(dict(shortcut))
    expected_output = _sha256_text(
        expected_shortcut_output_sha256, "expected shortcut-output digest"
    )
    if hashlib.sha256(_json_bytes(snapshot)).hexdigest() != expected_output:
        raise NuisanceValidationError(
            "authenticated records do not bind the shortcut-output bytes"
        )
    preregistration_snapshot = copy.deepcopy(dict(preregistration))
    expected_preregistration = _sha256_text(
        expected_shortcut_preregistration_sha256,
        "expected shortcut-preregistration digest",
    )
    if _canonical_digest(preregistration_snapshot) != expected_preregistration:
        raise NuisanceValidationError(
            "authenticated records do not bind the shortcut preregistration"
        )
    normalized_metadata_records = _normalize_metadata_records_v2(metadata_records)
    if _canonical_digest(normalized_metadata_records) != preregistration_snapshot.get(
        "metadata_records_sha256"
    ):
        raise NuisanceValidationError(
            "authenticated metadata-record binding differs from the preregistration"
        )
    records_body = {
        "shortcut_output_snapshot": snapshot,
        "shortcut_preregistration_snapshot": preregistration_snapshot,
        "metadata_records": normalized_metadata_records,
        "question_only_records": copy.deepcopy(list(question_only_records)),
        "question_blind_records": copy.deepcopy(list(question_blind_records)),
        "remux_records": copy.deepcopy(list(remux_records)),
        "orientation_records": copy.deepcopy(list(orientation_records)),
        "permutation_records": copy.deepcopy(list(permutation_records)),
        "shuffled_records": copy.deepcopy(list(shuffled_records)),
    }
    return {
        "schema": "conflictbench.perception-shortcut-authenticated-records.v2",
        **records_body,
        "records_sha256": _canonical_digest(records_body),
    }


def _validate_authenticated_shortcut_records_v2(
    value: Any,
    *,
    report: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    authenticated = _mapping(value, "v2 authenticated shortcut records")
    record_keys = {
        "shortcut_output_snapshot",
        "shortcut_preregistration_snapshot",
        "metadata_records",
        "question_only_records",
        "question_blind_records",
        "remux_records",
        "orientation_records",
        "permutation_records",
        "shuffled_records",
    }
    _exact_keys(
        authenticated,
        {"schema", "records_sha256"} | record_keys,
        "v2 authenticated shortcut records",
    )
    if authenticated.get("schema") != (
        "conflictbench.perception-shortcut-authenticated-records.v2"
    ):
        raise NuisanceValidationError("v2 authenticated-record schema differs")
    records_body = {key: authenticated[key] for key in sorted(record_keys)}
    if authenticated.get("records_sha256") != _canonical_digest(records_body):
        raise NuisanceValidationError("v2 authenticated-record digest differs")

    raw_snapshot = _mapping(
        authenticated.get("shortcut_output_snapshot"),
        "v2 authenticated shortcut-output snapshot",
    )
    snapshot = validate_shortcut_output_v2(
        raw_snapshot,
        expected_configuration_sha256=report["configuration_sha256"],
        expected_media_set_sha256=raw_snapshot["input_digests"]["media_set_sha256"],
        expected_gate_implementation_bundle_sha256=raw_snapshot["input_digests"][
            "gate_implementation_bundle_sha256"
        ],
        expected_shortcut_preregistration_sha256=report[
            "shortcut_preregistration_sha256"
        ],
    )
    if (
        hashlib.sha256(_json_bytes(snapshot)).hexdigest()
        != report["shortcut_output_sha256"]
    ):
        raise NuisanceValidationError(
            "v2 authenticated records shortcut-output binding differs"
        )
    if snapshot["attestation_sha256"] != report["output_attestation_sha256"]:
        raise NuisanceValidationError(
            "v2 authenticated records output attestation differs"
        )

    raw_preregistration = _mapping(
        authenticated.get("shortcut_preregistration_snapshot"),
        "v2 authenticated shortcut-preregistration snapshot",
    )
    raw_preregistration_digests = _mapping(
        raw_preregistration.get("input_digests"),
        "v2 authenticated shortcut-preregistration input digests",
    )
    preregistration = validate_shortcut_preregistration_v2(
        raw_preregistration,
        expected_configuration_sha256=report["configuration_sha256"],
        expected_gate_implementation_bundle_sha256=raw_preregistration_digests.get(
            "gate_implementation_bundle_sha256"
        ),
    )
    if _canonical_digest(preregistration) != report["shortcut_preregistration_sha256"]:
        raise NuisanceValidationError(
            "v2 authenticated records shortcut-preregistration binding differs"
        )
    if (
        preregistration["inventory"]["record_inventory_sha256"]
        != report["record_inventory_sha256"]
    ):
        raise NuisanceValidationError(
            "v2 authenticated records record-inventory binding differs"
        )
    if (
        preregistration["input_digests"]["power_output_sha256"]
        != report["power_record_sha256"]
    ):
        raise NuisanceValidationError(
            "v2 authenticated records power-record binding differs"
        )
    expected_dependency_digests = {
        role: preregistration["input_digests"][verification_key]
        for role, (_, verification_key) in _V2_DEPENDENCY_ROLES.items()
    }
    if expected_dependency_digests != report["dependency_verification_sha256s"]:
        raise NuisanceValidationError(
            "v2 authenticated records dependency binding differs"
        )
    if preregistration["replay_worker"] != report["replay_worker"]:
        raise NuisanceValidationError(
            "v2 authenticated records replay-worker binding differs"
        )

    metadata_records = authenticated.get("metadata_records")
    if not isinstance(metadata_records, list):
        raise NuisanceValidationError(
            "v2 authenticated metadata records must be a list"
        )
    normalized_metadata_records = _validate_metadata_record_inventory_v2(
        metadata_records, snapshot, config
    )
    if (
        _canonical_digest(normalized_metadata_records)
        != preregistration["metadata_records_sha256"]
    ):
        raise NuisanceValidationError(
            "v2 authenticated metadata-record binding differs"
        )

    expected_question_only = [
        {
            **record,
            "verified_prediction": record["primary_output"]["prediction"],
        }
        for record in snapshot["question_only_conflict"]["records"]
    ]
    threshold = float(snapshot["question_blind_audiovisual_mismatch"]["threshold"])
    expected_question_blind = [
        {
            **record,
            "verified_prediction": (
                "mismatched"
                if float(record["primary_output"]["mismatch_probability"]) >= threshold
                else "matched"
            ),
        }
        for record in snapshot["question_blind_audiovisual_mismatch"]["records"]
    ]
    expected_sections = {
        "question_only_records": expected_question_only,
        "question_blind_records": expected_question_blind,
        "remux_records": snapshot["remux_noninferiority"]["records"],
        "orientation_records": snapshot["orientation_symmetry"]["records"],
        "permutation_records": snapshot["option_permutation_equivalence"]["records"],
        "shuffled_records": snapshot["shuffled_question"]["records"],
    }
    for section_name, expected_records in expected_sections.items():
        if authenticated.get(section_name) != expected_records:
            raise NuisanceValidationError(
                f"v2 authenticated records differ for {section_name}"
            )
    return copy.deepcopy(dict(authenticated))


def _summarize_authenticated_shortcut_records_v2(
    authenticated: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    worker: Mapping[str, Any],
    endpoint_confidence: float,
) -> dict[str, Any]:
    """Recompute every reported statistic from the authenticated record bundle."""

    shortcut = authenticated["shortcut_output_snapshot"]
    protocol = config["shortcut_protocol"]
    repetitions = int(protocol["bootstrap_repetitions"])
    seed = int(protocol["bootstrap_seed"])
    metadata_summary = _metadata_upper_bound_v2(
        authenticated["metadata_records"],
        config,
        endpoint_confidence_level=endpoint_confidence,
    )

    question_only_values = _component_balanced_accuracies(
        authenticated["question_only_records"],
        gold_key="gold_relation",
        prediction=lambda record: record["verified_prediction"],
        labels=("AGREE", "CONFLICT"),
        name="question only",
    )
    question_only_summary = _component_mean_interval(
        question_only_values,
        repetitions=repetitions,
        seed=seed - 1,
        confidence_level=endpoint_confidence,
        bounded_unit_interval=True,
    )
    question_only_summary["balanced_accuracy"] = question_only_summary.pop(
        "point_estimate"
    )
    question_only_summary.update(
        {
            "fresh_replays_per_record": 2,
            "request_schema": REPLAY_REQUEST_SCHEMA_V2,
            "output_schema": RELATION_OUTPUT_SCHEMA_V2,
            "verifier_bindings": copy.deepcopy(dict(worker)),
        }
    )

    question_blind_values = _component_balanced_accuracies(
        authenticated["question_blind_records"],
        gold_key="expected_relation",
        prediction=lambda record: record["verified_prediction"],
        labels=("matched", "mismatched"),
        name="question-blind",
    )
    question_blind_summary = _component_mean_interval(
        question_blind_values,
        repetitions=repetitions,
        seed=seed,
        confidence_level=endpoint_confidence,
        bounded_unit_interval=True,
    )
    question_blind_summary["balanced_accuracy"] = question_blind_summary.pop(
        "point_estimate"
    )
    question_blind_summary.update(
        {
            "fresh_replays_per_record": 2,
            "request_schema": QUESTION_BLIND_REQUEST_SCHEMA_V2,
            "output_schema": QUESTION_BLIND_OUTPUT_SCHEMA_V2,
            "threshold_record_sha256": shortcut["question_blind_audiovisual_mismatch"][
                "threshold_record_sha256"
            ],
            "verifier_bindings": copy.deepcopy(dict(worker)),
        }
    )

    accuracy_values = []
    p_gold_values = []
    for record in authenticated["remux_records"]:
        clean_accuracy, clean_p_gold = _choice_metrics(record, "clean_output")
        remux_accuracy, remux_p_gold = _choice_metrics(record, "remux_output")
        accuracy_values.append(
            (record["component_id"], remux_accuracy - clean_accuracy)
        )
        p_gold_values.append((record["component_id"], remux_p_gold - clean_p_gold))
    remux_summary = {
        "accuracy_difference": _component_mean_interval(
            accuracy_values,
            repetitions=repetitions,
            seed=seed + 10,
            confidence_level=endpoint_confidence,
        ),
        "p_gold_difference": _component_mean_interval(
            p_gold_values,
            repetitions=repetitions,
            seed=seed + 11,
            confidence_level=endpoint_confidence,
        ),
        "epsilon_p_gold": shortcut["remux_noninferiority"]["epsilon_p_gold"],
        "epsilon_source": copy.deepcopy(
            shortcut["remux_noninferiority"]["epsilon_source"]
        ),
    }

    orientation_summary = {
        "audio_over_video_minus_video_over_audio": _paired_balanced_difference(
            authenticated["orientation_records"],
            first_key="audio_over_video_output",
            second_key="video_over_audio_output",
            repetitions=repetitions,
            seed=seed + 20,
            confidence_level=endpoint_confidence,
            name="orientation",
        )
    }
    permutation_summary = {}
    for pair_index, (first, second) in enumerate((("0", "1"), ("0", "2"), ("1", "2"))):
        projected = [
            {
                **record,
                "first_output": record["outputs"][first],
                "second_output": record["outputs"][second],
            }
            for record in authenticated["permutation_records"]
        ]
        permutation_summary[f"{first}_minus_{second}"] = _paired_balanced_difference(
            projected,
            first_key="first_output",
            second_key="second_output",
            repetitions=repetitions,
            seed=seed + 30 + pair_index,
            confidence_level=endpoint_confidence,
            name="option permutation",
        )

    aligned_values = _component_balanced_accuracies(
        authenticated["shuffled_records"],
        gold_key="gold_relation",
        prediction=lambda record: record["aligned_output"]["prediction"],
        labels=("AGREE", "CONFLICT"),
        name="shuffled question",
    )
    shuffled_values = _component_balanced_accuracies(
        authenticated["shuffled_records"],
        gold_key="gold_relation",
        prediction=lambda record: record["shuffled_output"]["prediction"],
        labels=("AGREE", "CONFLICT"),
        name="shuffled question",
    )
    aligned_by_component = dict(aligned_values)
    shuffled_by_component = dict(shuffled_values)
    shuffled_summary = {
        "aligned": _component_mean_interval(
            aligned_values,
            repetitions=repetitions,
            seed=seed + 40,
            confidence_level=endpoint_confidence,
            bounded_unit_interval=True,
        ),
        "shuffled": _component_mean_interval(
            shuffled_values,
            repetitions=repetitions,
            seed=seed + 41,
            confidence_level=endpoint_confidence,
            bounded_unit_interval=True,
        ),
        "aligned_minus_shuffled": _component_mean_interval(
            [
                (
                    component_id,
                    aligned_by_component[component_id]
                    - shuffled_by_component[component_id],
                )
                for component_id in sorted(aligned_by_component)
            ],
            repetitions=repetitions,
            seed=seed + 42,
            confidence_level=endpoint_confidence,
        ),
    }
    return {
        "metadata_only": metadata_summary,
        "question_only_conflict": question_only_summary,
        "question_blind_audiovisual_mismatch": question_blind_summary,
        "remux_noninferiority": remux_summary,
        "orientation_symmetry": orientation_summary,
        "option_permutation_equivalence": permutation_summary,
        "shuffled_question": shuffled_summary,
    }


def _replay_bound_control_output_v2(
    verifier: SubprocessReplayWorkerSpec,
    request: Mapping[str, Any],
    bound_output: Mapping[str, Any],
    *,
    media_root: pathlib.Path,
    output_kind: str,
    label: str,
) -> dict[str, Any]:
    """Run two fresh subprocesses and return only a verified control output."""

    validator = (
        _validate_choice_output_v2
        if output_kind == "choice"
        else _validate_relation_output_v2
    )
    fresh_outputs = []
    for replay_index in range(2):
        try:
            fresh_value = _run_control_replay_v2(
                verifier,
                request,
                media_root=media_root,
            )
        except Exception as exc:
            raise NuisanceValidationError(
                f"{label} fresh replay {replay_index} failed"
            ) from exc
        fresh_outputs.append(
            validator(fresh_value, f"{label} fresh replay {replay_index}")
        )
    if fresh_outputs[0] != bound_output or fresh_outputs[1] != bound_output:
        raise NuisanceValidationError(
            f"{label} fresh replay differs from the bound raw output"
        )
    return copy.deepcopy(dict(fresh_outputs[0]))


def evaluate_shortcut_controls_v2(
    shortcut_output_value: Any,
    shortcut_preregistration_value: Any,
    metadata_records_value: Sequence[Mapping[str, Any]],
    config_value: Mapping[str, Any],
    *,
    expected_configuration_sha256: str,
    expected_gate_implementation_bundle_sha256: str,
    expected_shortcut_preregistration_sha256: str,
    expected_shortcut_output_sha256: str,
    replay_worker: SubprocessReplayWorkerSpec | None,
    derived_media_root: pathlib.Path | None,
) -> dict[str, Any]:
    """Replay and evaluate every preregistered v2 shortcut control."""

    config = _validate_config(config_value)
    if config["schema_version"] != 2:
        raise NuisanceValidationError("shortcut controls require configuration v2")
    shortcut, preregistration = validate_shortcut_output_against_preregistration_v2(
        shortcut_output_value,
        shortcut_preregistration_value,
        expected_configuration_sha256=expected_configuration_sha256,
        expected_gate_implementation_bundle_sha256=(
            expected_gate_implementation_bundle_sha256
        ),
        expected_shortcut_preregistration_sha256=(
            expected_shortcut_preregistration_sha256
        ),
    )
    if type(replay_worker) is not SubprocessReplayWorkerSpec:
        raise NuisanceValidationError(
            "hash-pinned subprocess replay worker is required"
        )
    if derived_media_root is None:
        raise NuisanceValidationError("authenticated derived-media root is required")
    verifier = replay_worker
    media_authentication_before = authenticate_v2_media_inventory(
        derived_media_root, preregistration["inventory"]["media_files"]
    )
    verifier_bindings = {
        "configuration_sha256": _sha256_text(
            verifier.configuration_sha256,
            "question-blind verifier configuration digest",
        ),
        "implementation_sha256": _sha256_text(
            verifier.executable_sha256,
            "question-blind verifier implementation digest",
        ),
        "model_sha256": _sha256_text(
            verifier.model_sha256, "question-blind verifier model digest"
        ),
    }
    expected_verifier_bindings = preregistration["replay_worker"]
    if verifier_bindings != expected_verifier_bindings:
        raise NuisanceValidationError("question-blind replay-verifier binding differs")

    metadata_records = _validate_metadata_record_inventory_v2(
        metadata_records_value, shortcut, config
    )
    if (
        _canonical_digest(metadata_records)
        != preregistration["metadata_records_sha256"]
    ):
        raise NuisanceValidationError(
            "v2 metadata-record binding differs from the preregistration"
        )

    protocol = config["shortcut_protocol"]
    repetitions = int(protocol["bootstrap_repetitions"])
    seed = int(protocol["bootstrap_seed"])
    nominal_confidence = float(protocol["confidence_level"])
    confidence = 1.0 - (1.0 - nominal_confidence) / len(_V2_POWER_ENDPOINTS)
    question_only_records = []
    for index, record in enumerate(shortcut["question_only_conflict"]["records"]):
        fresh_outputs = []
        for replay_index in range(2):
            try:
                fresh_value = run_replay_worker(verifier, record["request"])
            except Exception as exc:
                raise NuisanceValidationError(
                    f"question-only fresh replay {replay_index} failed for record {index}"
                ) from exc
            fresh_outputs.append(
                _validate_relation_output_v2(
                    fresh_value,
                    f"question-only fresh replay {replay_index} for record {index}",
                )
            )
        if (
            fresh_outputs[0] != record["primary_output"]
            or fresh_outputs[1] != record["replay_output"]
        ):
            raise NuisanceValidationError(
                "question-only fresh replay differs from the bound raw outputs"
            )
        question_only_records.append(
            {**record, "verified_prediction": fresh_outputs[0]["prediction"]}
        )
    question_only_values = _component_balanced_accuracies(
        question_only_records,
        gold_key="gold_relation",
        prediction=lambda record: record["verified_prediction"],
        labels=("AGREE", "CONFLICT"),
        name="question only",
    )
    question_only_summary = _component_mean_interval(
        question_only_values,
        repetitions=repetitions,
        seed=seed - 1,
        confidence_level=confidence,
        bounded_unit_interval=True,
    )
    question_only_summary["balanced_accuracy"] = question_only_summary.pop(
        "point_estimate"
    )
    question_only_summary.update(
        {
            "fresh_replays_per_record": 2,
            "request_schema": REPLAY_REQUEST_SCHEMA_V2,
            "output_schema": RELATION_OUTPUT_SCHEMA_V2,
            "verifier_bindings": verifier_bindings,
        }
    )
    question_blind = shortcut["question_blind_audiovisual_mismatch"]
    threshold = float(question_blind["threshold"])
    verified_records = []
    for index, record in enumerate(question_blind["records"]):
        fresh_outputs = []
        for replay_index in range(2):
            try:
                fresh_value = _run_question_blind_replay_v2(
                    verifier,
                    record["request"],
                    media_root=derived_media_root,
                )
            except Exception as exc:
                raise NuisanceValidationError(
                    f"question-blind fresh replay {replay_index} failed for record {index}"
                ) from exc
            fresh_outputs.append(
                _validate_mismatch_output_v2(
                    fresh_value,
                    f"question-blind fresh replay {replay_index} for record {index}",
                )
            )
        if (
            fresh_outputs[0] != record["primary_output"]
            or fresh_outputs[1] != record["replay_output"]
        ):
            raise NuisanceValidationError(
                "question-blind fresh replay differs from the bound raw outputs"
            )
        verified_records.append(
            {
                **record,
                "verified_prediction": (
                    "mismatched"
                    if float(fresh_outputs[0]["mismatch_probability"]) >= threshold
                    else "matched"
                ),
            }
        )
    question_blind_values = _component_balanced_accuracies(
        verified_records,
        gold_key="expected_relation",
        prediction=lambda record: record["verified_prediction"],
        labels=("matched", "mismatched"),
        name="question-blind",
    )
    question_blind_summary = _component_mean_interval(
        question_blind_values,
        repetitions=repetitions,
        seed=seed,
        confidence_level=confidence,
        bounded_unit_interval=True,
    )
    question_blind_summary["balanced_accuracy"] = question_blind_summary.pop(
        "point_estimate"
    )
    question_blind_summary.update(
        {
            "fresh_replays_per_record": 2,
            "request_schema": QUESTION_BLIND_REQUEST_SCHEMA_V2,
            "output_schema": QUESTION_BLIND_OUTPUT_SCHEMA_V2,
            "threshold_record_sha256": question_blind["threshold_record_sha256"],
            "verifier_bindings": verifier_bindings,
        }
    )

    metadata_summary = _metadata_upper_bound_v2(
        metadata_records,
        config,
        endpoint_confidence_level=confidence,
    )
    remux_records = []
    for index, record in enumerate(shortcut["remux_noninferiority"]["records"]):
        verified_record = dict(record)
        for prefix in ("clean", "remux"):
            verified_record[f"{prefix}_output"] = _replay_bound_control_output_v2(
                verifier,
                record[f"{prefix}_request"],
                record[f"{prefix}_output"],
                media_root=derived_media_root,
                output_kind="choice",
                label=f"remux record {index} {prefix}",
            )
        remux_records.append(verified_record)
    accuracy_values = []
    p_gold_values = []
    for record in remux_records:
        clean_accuracy, clean_p_gold = _choice_metrics(record, "clean_output")
        remux_accuracy, remux_p_gold = _choice_metrics(record, "remux_output")
        accuracy_values.append(
            (record["component_id"], remux_accuracy - clean_accuracy)
        )
        p_gold_values.append((record["component_id"], remux_p_gold - clean_p_gold))
    remux_summary = {
        "accuracy_difference": _component_mean_interval(
            accuracy_values,
            repetitions=repetitions,
            seed=seed + 10,
            confidence_level=confidence,
        ),
        "p_gold_difference": _component_mean_interval(
            p_gold_values,
            repetitions=repetitions,
            seed=seed + 11,
            confidence_level=confidence,
        ),
        "epsilon_p_gold": shortcut["remux_noninferiority"]["epsilon_p_gold"],
        "epsilon_source": shortcut["remux_noninferiority"]["epsilon_source"],
    }

    orientation_records = []
    for index, record in enumerate(shortcut["orientation_symmetry"]["records"]):
        verified_record = dict(record)
        for prefix in ("audio_over_video", "video_over_audio"):
            verified_record[f"{prefix}_output"] = _replay_bound_control_output_v2(
                verifier,
                record[f"{prefix}_request"],
                record[f"{prefix}_output"],
                media_root=derived_media_root,
                output_kind="relation",
                label=f"orientation record {index} {prefix}",
            )
        orientation_records.append(verified_record)
    orientation_summary = {
        "audio_over_video_minus_video_over_audio": _paired_balanced_difference(
            orientation_records,
            first_key="audio_over_video_output",
            second_key="video_over_audio_output",
            repetitions=repetitions,
            seed=seed + 20,
            confidence_level=confidence,
            name="orientation",
        )
    }

    permutation_records = []
    for index, record in enumerate(
        shortcut["option_permutation_equivalence"]["records"]
    ):
        verified_record = {**record, "outputs": {}}
        for permutation in ("0", "1", "2"):
            verified_record["outputs"][permutation] = _replay_bound_control_output_v2(
                verifier,
                record["requests"][permutation],
                record["outputs"][permutation],
                media_root=derived_media_root,
                output_kind="relation",
                label=f"option-permutation record {index} variant {permutation}",
            )
        permutation_records.append(verified_record)
    permutation_summary = {}
    for pair_index, (first, second) in enumerate((("0", "1"), ("0", "2"), ("1", "2"))):
        projected = [
            {
                **record,
                "first_output": record["outputs"][first],
                "second_output": record["outputs"][second],
            }
            for record in permutation_records
        ]
        permutation_summary[f"{first}_minus_{second}"] = _paired_balanced_difference(
            projected,
            first_key="first_output",
            second_key="second_output",
            repetitions=repetitions,
            seed=seed + 30 + pair_index,
            confidence_level=confidence,
            name="option permutation",
        )

    shuffled_records = []
    for index, record in enumerate(shortcut["shuffled_question"]["records"]):
        verified_record = dict(record)
        for prefix in ("aligned", "shuffled"):
            verified_record[f"{prefix}_output"] = _replay_bound_control_output_v2(
                verifier,
                record[f"{prefix}_request"],
                record[f"{prefix}_output"],
                media_root=derived_media_root,
                output_kind="relation",
                label=f"shuffled-question record {index} {prefix}",
            )
        shuffled_records.append(verified_record)
    aligned_values = _component_balanced_accuracies(
        shuffled_records,
        gold_key="gold_relation",
        prediction=lambda record: record["aligned_output"]["prediction"],
        labels=("AGREE", "CONFLICT"),
        name="shuffled question",
    )
    shuffled_values = _component_balanced_accuracies(
        shuffled_records,
        gold_key="gold_relation",
        prediction=lambda record: record["shuffled_output"]["prediction"],
        labels=("AGREE", "CONFLICT"),
        name="shuffled question",
    )
    aligned_by_component = dict(aligned_values)
    shuffled_by_component = dict(shuffled_values)
    shuffled_summary = {
        "aligned": _component_mean_interval(
            aligned_values,
            repetitions=repetitions,
            seed=seed + 40,
            confidence_level=confidence,
            bounded_unit_interval=True,
        ),
        "shuffled": _component_mean_interval(
            shuffled_values,
            repetitions=repetitions,
            seed=seed + 41,
            confidence_level=confidence,
            bounded_unit_interval=True,
        ),
        "aligned_minus_shuffled": _component_mean_interval(
            [
                (
                    component_id,
                    aligned_by_component[component_id]
                    - shuffled_by_component[component_id],
                )
                for component_id in sorted(aligned_by_component)
            ],
            repetitions=repetitions,
            seed=seed + 42,
            confidence_level=confidence,
        ),
    }

    reasons = _shortcut_reason_codes_v2(
        metadata_summary=metadata_summary,
        question_only_summary=question_only_summary,
        question_blind_summary=question_blind_summary,
        remux_summary=remux_summary,
        orientation_summary=orientation_summary,
        permutation_summary=permutation_summary,
        shuffled_summary=shuffled_summary,
        gate=config["shortcut_gate"],
    )

    media_authentication_after = authenticate_v2_media_inventory(
        derived_media_root, preregistration["inventory"]["media_files"]
    )
    if media_authentication_after != media_authentication_before:
        raise NuisanceValidationError("derived media changed during v2 replay")

    authenticated_records = _build_authenticated_shortcut_records_v2(
        shortcut=shortcut,
        preregistration=preregistration,
        expected_shortcut_preregistration_sha256=(
            expected_shortcut_preregistration_sha256
        ),
        expected_shortcut_output_sha256=expected_shortcut_output_sha256,
        metadata_records=metadata_records,
        question_only_records=question_only_records,
        question_blind_records=verified_records,
        remux_records=remux_records,
        orientation_records=orientation_records,
        permutation_records=permutation_records,
        shuffled_records=shuffled_records,
    )

    report = {
        "schema": "conflictbench.perception-shortcut-report.v2",
        "status": "complete",
        "configuration": copy.deepcopy(dict(config)),
        "configuration_sha256": expected_configuration_sha256,
        "shortcut_output_sha256": _sha256_text(
            expected_shortcut_output_sha256, "expected shortcut-output digest"
        ),
        "output_attestation_sha256": shortcut["attestation_sha256"],
        "shortcut_preregistration_sha256": (expected_shortcut_preregistration_sha256),
        "record_inventory_sha256": preregistration["inventory"][
            "record_inventory_sha256"
        ],
        "power_record_sha256": preregistration["input_digests"]["power_output_sha256"],
        "dependency_verification_sha256s": {
            role: preregistration["input_digests"][verification_key]
            for role, (_, verification_key) in _V2_DEPENDENCY_ROLES.items()
        },
        "derived_media_authentication": {
            "before": media_authentication_before,
            "after": media_authentication_after,
        },
        "replay_worker": verifier_bindings,
        "inference": {
            "method": protocol["uncertainty_method"],
            "endpoint_count": len(_V2_POWER_ENDPOINTS),
            "nominal_familywise_confidence_level": nominal_confidence,
            "per_endpoint_confidence_level": confidence,
        },
        "authenticated_records": authenticated_records,
        "metadata_only": metadata_summary,
        "question_only_conflict": question_only_summary,
        "question_blind_audiovisual_mismatch": question_blind_summary,
        "remux_noninferiority": remux_summary,
        "orientation_symmetry": orientation_summary,
        "option_permutation_equivalence": permutation_summary,
        "shuffled_question": shuffled_summary,
        "decision": {
            "status": "pass" if not reasons else "fail",
            "reason_codes": reasons,
        },
    }
    report["attestation_sha256"] = _canonical_digest(report)
    return report


def _validate_component_interval_v2(
    value: Any,
    *,
    label: str,
    point_key: str,
    expected_confidence_level: float,
    minimum_component_count: int,
    bounded_unit_interval: bool = False,
) -> dict[str, Any]:
    interval = _mapping(value, label)
    _exact_keys(
        interval,
        {
            "component_count",
            point_key,
            "one_sided_lower_confidence_bound",
            "one_sided_upper_confidence_bound",
            "confidence_level",
            "method",
        },
        label,
    )
    component_count = _positive_integer(
        interval.get("component_count"), f"{label} component count"
    )
    if component_count < minimum_component_count:
        raise NuisanceValidationError(f"{label} component count is too small")
    point = _exact_float(interval.get(point_key), f"{label} {point_key}")
    lower = _exact_float(
        interval.get("one_sided_lower_confidence_bound"), f"{label} lower bound"
    )
    upper = _exact_float(
        interval.get("one_sided_upper_confidence_bound"), f"{label} upper bound"
    )
    if lower > upper:
        raise NuisanceValidationError(f"{label} interval is reversed")
    if bounded_unit_interval and any(
        number < 0.0 or number > 1.0 for number in (point, lower, upper)
    ):
        raise NuisanceValidationError(f"{label} leaves the unit interval")
    if (
        _exact_float(interval.get("confidence_level"), f"{label} confidence level")
        != expected_confidence_level
    ):
        raise NuisanceValidationError(f"{label} confidence level differs")
    if interval.get("method") != "analysis_component_cluster_percentile":
        raise NuisanceValidationError(f"{label} method differs")
    return copy.deepcopy(dict(interval))


def _validate_shortcut_report_summaries_v2(
    report: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    worker: Mapping[str, Any],
    endpoint_confidence: float,
) -> None:
    minimum_components = int(config["protocol"]["minimum_unseen_question_components"])
    metadata = _mapping(report.get("metadata_only"), "metadata summary")
    _exact_keys(
        metadata,
        {
            "models",
            "comparison_count",
            "simultaneous_confidence_level",
            "selected_family",
            "selected_metric",
            "one_sided_upper_confidence_bound",
        },
        "metadata summary",
    )
    families = list(config["model"]["families"])
    comparison_count = len(families) * 2
    if metadata.get("comparison_count") != comparison_count:
        raise NuisanceValidationError("metadata comparison count differs")
    simultaneous_confidence = 1.0 - (1.0 - endpoint_confidence) / comparison_count
    if (
        _exact_float(
            metadata.get("simultaneous_confidence_level"),
            "metadata simultaneous confidence",
        )
        != simultaneous_confidence
    ):
        raise NuisanceValidationError("metadata simultaneous confidence differs")
    models = _mapping(metadata.get("models"), "metadata models")
    _exact_keys(models, set(families), "metadata models")
    candidates = []
    for family in families:
        family_summary = _mapping(models[family], f"metadata {family}")
        _exact_keys(
            family_summary,
            {"pair_ranking", "thresholded_balanced_accuracy"},
            f"metadata {family}",
        )
        for metric in ("pair_ranking", "thresholded_balanced_accuracy"):
            interval = _validate_component_interval_v2(
                family_summary[metric],
                label=f"metadata {family} {metric}",
                point_key="point_estimate",
                expected_confidence_level=simultaneous_confidence,
                minimum_component_count=minimum_components,
                bounded_unit_interval=True,
            )
            candidates.append(
                (
                    family,
                    metric,
                    interval["one_sided_upper_confidence_bound"],
                )
            )
    selected_family, selected_metric, selected_upper = max(
        candidates, key=lambda item: item[2]
    )
    if (
        metadata.get("selected_family") != selected_family
        or metadata.get("selected_metric") != selected_metric
        or _exact_float(
            metadata.get("one_sided_upper_confidence_bound"),
            "metadata selected upper bound",
        )
        != selected_upper
    ):
        raise NuisanceValidationError("metadata selected maximum differs")

    for section_name, output_schema, request_schema in (
        (
            "question_only_conflict",
            RELATION_OUTPUT_SCHEMA_V2,
            REPLAY_REQUEST_SCHEMA_V2,
        ),
        (
            "question_blind_audiovisual_mismatch",
            QUESTION_BLIND_OUTPUT_SCHEMA_V2,
            QUESTION_BLIND_REQUEST_SCHEMA_V2,
        ),
    ):
        summary = _mapping(report.get(section_name), f"{section_name} summary")
        extra = {"threshold_record_sha256"} if "blind" in section_name else set()
        _exact_keys(
            summary,
            {
                "component_count",
                "balanced_accuracy",
                "one_sided_lower_confidence_bound",
                "one_sided_upper_confidence_bound",
                "confidence_level",
                "method",
                "fresh_replays_per_record",
                "request_schema",
                "output_schema",
                "verifier_bindings",
            }
            | extra,
            f"{section_name} summary",
        )
        base = {
            key: item
            for key, item in summary.items()
            if key
            in {
                "component_count",
                "balanced_accuracy",
                "one_sided_lower_confidence_bound",
                "one_sided_upper_confidence_bound",
                "confidence_level",
                "method",
            }
        }
        _validate_component_interval_v2(
            base,
            label=f"{section_name} interval",
            point_key="balanced_accuracy",
            expected_confidence_level=endpoint_confidence,
            minimum_component_count=minimum_components,
            bounded_unit_interval=True,
        )
        if summary.get("fresh_replays_per_record") != 2:
            raise NuisanceValidationError(f"{section_name} replay count differs")
        if summary.get("request_schema") != request_schema:
            raise NuisanceValidationError(f"{section_name} request schema differs")
        if summary.get("output_schema") != output_schema:
            raise NuisanceValidationError(f"{section_name} output schema differs")
        if summary.get("verifier_bindings") != worker:
            raise NuisanceValidationError(f"{section_name} worker binding differs")
        if extra:
            _sha256_text(
                summary.get("threshold_record_sha256"),
                "question-blind threshold record",
            )

    remux = _mapping(report.get("remux_noninferiority"), "remux summary")
    _exact_keys(
        remux,
        {
            "accuracy_difference",
            "p_gold_difference",
            "epsilon_p_gold",
            "epsilon_source",
        },
        "remux summary",
    )
    for metric in ("accuracy_difference", "p_gold_difference"):
        _validate_component_interval_v2(
            remux[metric],
            label=f"remux {metric}",
            point_key="point_estimate",
            expected_confidence_level=endpoint_confidence,
            minimum_component_count=minimum_components,
        )
    _exact_float(
        remux.get("epsilon_p_gold"),
        "remux epsilon",
        minimum=0.0,
        maximum=1.0,
    )
    epsilon_source = _mapping(remux.get("epsilon_source"), "remux epsilon source")
    _exact_keys(
        epsilon_source,
        {"quantile", "semantic_fit_output_sha256"},
        "remux epsilon source",
    )
    if _exact_float(epsilon_source.get("quantile"), "remux epsilon quantile") != 0.95:
        raise NuisanceValidationError("remux epsilon quantile differs")
    _sha256_text(
        epsilon_source.get("semantic_fit_output_sha256"),
        "remux semantic-fit digest",
    )

    orientation = _mapping(report.get("orientation_symmetry"), "orientation summary")
    _exact_keys(
        orientation,
        {"audio_over_video_minus_video_over_audio"},
        "orientation summary",
    )
    _validate_component_interval_v2(
        orientation["audio_over_video_minus_video_over_audio"],
        label="orientation difference",
        point_key="point_estimate",
        expected_confidence_level=endpoint_confidence,
        minimum_component_count=minimum_components,
    )

    permutation = _mapping(
        report.get("option_permutation_equivalence"), "option-permutation summary"
    )
    _exact_keys(
        permutation,
        {"0_minus_1", "0_minus_2", "1_minus_2"},
        "option-permutation summary",
    )
    for name, interval in permutation.items():
        _validate_component_interval_v2(
            interval,
            label=f"option-permutation {name}",
            point_key="point_estimate",
            expected_confidence_level=endpoint_confidence,
            minimum_component_count=minimum_components,
        )

    shuffled = _mapping(report.get("shuffled_question"), "shuffled summary")
    _exact_keys(
        shuffled,
        {"aligned", "shuffled", "aligned_minus_shuffled"},
        "shuffled summary",
    )
    for name in ("aligned", "shuffled", "aligned_minus_shuffled"):
        _validate_component_interval_v2(
            shuffled[name],
            label=f"shuffled {name}",
            point_key="point_estimate",
            expected_confidence_level=endpoint_confidence,
            minimum_component_count=minimum_components,
            bounded_unit_interval=name != "aligned_minus_shuffled",
        )


def validate_shortcut_report_v2(value: Any) -> dict[str, Any]:
    """Validate a complete v2 report and recompute its gate decision."""

    report = _validate_attested_v2(value, "v2 shortcut report")
    required = {
        "schema",
        "status",
        "configuration",
        "configuration_sha256",
        "shortcut_output_sha256",
        "output_attestation_sha256",
        "shortcut_preregistration_sha256",
        "record_inventory_sha256",
        "power_record_sha256",
        "dependency_verification_sha256s",
        "derived_media_authentication",
        "replay_worker",
        "inference",
        "authenticated_records",
        "metadata_only",
        "question_only_conflict",
        "question_blind_audiovisual_mismatch",
        "remux_noninferiority",
        "orientation_symmetry",
        "option_permutation_equivalence",
        "shuffled_question",
        "decision",
        "attestation_sha256",
    }
    _exact_keys(report, required, "v2 shortcut report")
    if report.get("schema") != "conflictbench.perception-shortcut-report.v2":
        raise NuisanceValidationError("v2 shortcut report schema differs")
    if report.get("status") != "complete":
        raise NuisanceValidationError("v2 shortcut report is incomplete")
    config = _validate_config(report.get("configuration"))
    if config["schema_version"] != 2:
        raise NuisanceValidationError("v2 shortcut report configuration differs")
    for name in (
        "configuration_sha256",
        "shortcut_output_sha256",
        "output_attestation_sha256",
        "shortcut_preregistration_sha256",
        "record_inventory_sha256",
        "power_record_sha256",
    ):
        _sha256_text(report.get(name), f"v2 shortcut report {name}")
    dependency_sha256s = _mapping(
        report.get("dependency_verification_sha256s"),
        "v2 report dependency verification digests",
    )
    _exact_keys(
        dependency_sha256s,
        set(_V2_DEPENDENCY_ROLES),
        "v2 report dependency verification digests",
    )
    for role, digest in dependency_sha256s.items():
        _sha256_text(digest, f"v2 report {role} verification digest")
    worker = _mapping(report.get("replay_worker"), "v2 report replay worker")
    _exact_keys(worker, _V2_WORKER_DIGEST_KEYS, "v2 report replay worker")
    for name in _V2_WORKER_DIGEST_KEYS:
        _sha256_text(worker.get(name), f"v2 report replay worker {name}")

    inference = _mapping(report.get("inference"), "v2 report inference")
    _exact_keys(
        inference,
        {
            "method",
            "endpoint_count",
            "nominal_familywise_confidence_level",
            "per_endpoint_confidence_level",
        },
        "v2 report inference",
    )
    if inference.get("method") != (
        "one_sided_component_cluster_percentile_bonferroni_familywise"
    ):
        raise NuisanceValidationError("v2 report inference method differs")
    if inference.get("endpoint_count") != len(_V2_POWER_ENDPOINTS):
        raise NuisanceValidationError("v2 report endpoint count differs")
    nominal_confidence = _exact_float(
        inference.get("nominal_familywise_confidence_level"),
        "v2 report familywise confidence",
        minimum=0.0,
        maximum=1.0,
    )
    if nominal_confidence != 0.95:
        raise NuisanceValidationError("v2 report familywise confidence differs")
    endpoint_confidence = _exact_float(
        inference.get("per_endpoint_confidence_level"),
        "v2 report per-endpoint confidence",
        minimum=0.0,
        maximum=1.0,
    )
    if endpoint_confidence != 1.0 - (1.0 - nominal_confidence) / len(
        _V2_POWER_ENDPOINTS
    ):
        raise NuisanceValidationError("v2 report per-endpoint confidence differs")

    authenticated_records = _validate_authenticated_shortcut_records_v2(
        report.get("authenticated_records"),
        report=report,
        config=config,
    )

    _validate_shortcut_report_summaries_v2(
        report,
        config=config,
        worker=worker,
        endpoint_confidence=endpoint_confidence,
    )
    recomputed_summaries = _summarize_authenticated_shortcut_records_v2(
        authenticated_records,
        config=config,
        worker=worker,
        endpoint_confidence=endpoint_confidence,
    )
    for section_name, recomputed in recomputed_summaries.items():
        if report.get(section_name) != recomputed:
            raise NuisanceValidationError(
                f"v2 {section_name} summary differs from authenticated records"
            )

    media = _mapping(
        report.get("derived_media_authentication"),
        "v2 report derived-media authentication",
    )
    _exact_keys(media, {"before", "after"}, "v2 derived-media authentication")
    if media["before"] != media["after"]:
        raise NuisanceValidationError("v2 report derived media changed during replay")
    authenticated = _mapping(media["before"], "v2 report media authentication")
    _exact_keys(
        authenticated,
        {"media_file_count", "media_files", "media_inventory_sha256"},
        "v2 report media authentication",
    )
    _positive_integer(
        authenticated.get("media_file_count"), "v2 report media-file count"
    )
    if authenticated.get("media_file_count") != len(
        authenticated.get("media_files", [])
    ):
        raise NuisanceValidationError("v2 report media-file count differs")
    if _canonical_digest(authenticated.get("media_files")) != _sha256_text(
        authenticated.get("media_inventory_sha256"),
        "v2 report media-inventory digest",
    ):
        raise NuisanceValidationError("v2 report media-inventory digest differs")

    summaries = {
        "metadata_summary": recomputed_summaries["metadata_only"],
        "question_only_summary": recomputed_summaries["question_only_conflict"],
        "question_blind_summary": recomputed_summaries[
            "question_blind_audiovisual_mismatch"
        ],
        "remux_summary": recomputed_summaries["remux_noninferiority"],
        "orientation_summary": recomputed_summaries["orientation_symmetry"],
        "permutation_summary": recomputed_summaries["option_permutation_equivalence"],
        "shuffled_summary": recomputed_summaries["shuffled_question"],
    }
    reasons = _shortcut_reason_codes_v2(
        **summaries,
        gate=config["shortcut_gate"],
    )
    expected_decision = {
        "status": "pass" if not reasons else "fail",
        "reason_codes": reasons,
    }
    if report.get("decision") != expected_decision:
        raise NuisanceValidationError("v2 shortcut report decision differs")
    return copy.deepcopy(dict(report))


def validate_shortcut_report_bindings_v2(
    value: Any,
    *,
    expected_configuration_sha256: str,
    expected_shortcut_preregistration_sha256: str,
    expected_shortcut_output_sha256: str,
    expected_output_attestation_sha256: str,
    expected_record_inventory_sha256: str,
    expected_power_record_sha256: str,
    expected_dependency_verification_sha256s: Mapping[str, str],
    expected_replay_worker: Mapping[str, str],
    expected_media_inventory_sha256: str,
) -> dict[str, Any]:
    """Bind a deeply validated report to the authenticated runner inputs."""

    report = validate_shortcut_report_v2(value)
    scalar_bindings = {
        "configuration_sha256": expected_configuration_sha256,
        "shortcut_preregistration_sha256": (expected_shortcut_preregistration_sha256),
        "shortcut_output_sha256": expected_shortcut_output_sha256,
        "output_attestation_sha256": expected_output_attestation_sha256,
        "record_inventory_sha256": expected_record_inventory_sha256,
        "power_record_sha256": expected_power_record_sha256,
    }
    for key, expected in scalar_bindings.items():
        if report[key] != _sha256_text(expected, f"expected v2 report {key}"):
            raise NuisanceValidationError(f"v2 report {key} binding differs")
    expected_dependency = {
        role: _sha256_text(digest, f"expected v2 {role} verification digest")
        for role, digest in expected_dependency_verification_sha256s.items()
    }
    if set(expected_dependency) != set(_V2_DEPENDENCY_ROLES):
        raise NuisanceValidationError("expected v2 dependency role set differs")
    if report["dependency_verification_sha256s"] != expected_dependency:
        raise NuisanceValidationError("v2 report dependency bindings differ")
    if report["replay_worker"] != dict(expected_replay_worker):
        raise NuisanceValidationError("v2 report replay-worker binding differs")
    expected_media = _sha256_text(
        expected_media_inventory_sha256, "expected v2 media inventory digest"
    )
    if (
        report["derived_media_authentication"]["before"]["media_inventory_sha256"]
        != expected_media
    ):
        raise NuisanceValidationError("v2 report media binding differs")
    return report


def build_shortcut_result_v2(
    *, report: Mapping[str, Any], report_sha256: str
) -> dict[str, Any]:
    """Build the minimal v2 handoff from a validated complete report."""

    validated = validate_shortcut_report_v2(report)
    observed_report_sha256 = hashlib.sha256(_json_bytes(validated)).hexdigest()
    if observed_report_sha256 != _sha256_text(
        report_sha256, "v2 shortcut report digest"
    ):
        raise NuisanceValidationError("v2 shortcut report byte digest differs")
    decision = _mapping(validated["decision"], "v2 shortcut report decision")
    result = {
        "schema": "conflictbench.perception-shortcut-result.v2",
        "status": "complete",
        "input_digests": {
            "configuration_sha256": validated["configuration_sha256"],
            "shortcut_preregistration_sha256": validated[
                "shortcut_preregistration_sha256"
            ],
            "shortcut_output_attestation_sha256": validated[
                "output_attestation_sha256"
            ],
            "shortcut_output_sha256": validated["shortcut_output_sha256"],
            "report_sha256": observed_report_sha256,
        },
        "decision": {
            "status": decision["status"],
            "reason_codes": list(decision["reason_codes"]),
            "scope": "automated_perception_shortcut_controls",
        },
    }
    return _attest_v2(result)


def validate_shortcut_result_v2(
    value: Any, *, report: Mapping[str, Any], expected_report_sha256: str
) -> dict[str, Any]:
    """Reject an incomplete, changed, or report-unbound v2 result."""

    result = _validate_attested_v2(value, "v2 shortcut result")
    _exact_keys(
        result,
        {"schema", "status", "input_digests", "decision", "attestation_sha256"},
        "v2 shortcut result",
    )
    if result.get("schema") != "conflictbench.perception-shortcut-result.v2":
        raise NuisanceValidationError("v2 shortcut result schema differs")
    if result.get("status") != "complete":
        raise NuisanceValidationError("v2 shortcut result is incomplete")
    rebuilt = build_shortcut_result_v2(
        report=report,
        report_sha256=expected_report_sha256,
    )
    if dict(result) != rebuilt:
        raise NuisanceValidationError("v2 shortcut result differs from its report")
    return copy.deepcopy(dict(result))


def write_shortcut_gate_outputs_v2(
    report_path: pathlib.Path,
    result_path: pathlib.Path,
    *,
    report: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    """Validate and write the v2 report/result pair exactly once."""

    report_target = pathlib.Path(report_path)
    result_target = pathlib.Path(result_path)
    if report_target == result_target:
        raise NuisanceValidationError("report and result paths must differ")
    if any(
        path.exists() or path.is_symlink() for path in (report_target, result_target)
    ):
        raise NuisanceValidationError("output already exists")
    validated_report = validate_shortcut_report_v2(report)
    report_payload = _json_bytes(validated_report)
    report_sha256 = hashlib.sha256(report_payload).hexdigest()
    validated_result = validate_shortcut_result_v2(
        result,
        report=validated_report,
        expected_report_sha256=report_sha256,
    )
    _write_once(report_target, report_payload)
    try:
        _write_once(result_target, _json_bytes(validated_result))
    except Exception:
        report_target.unlink(missing_ok=True)
        raise


def _sha256_text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NuisanceValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _attest_v2(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result["attestation_sha256"] = _canonical_digest(result)
    return result


def _validate_attested_v2(value: Any, label: str) -> Mapping[str, Any]:
    record = _mapping(value, label)
    attestation = _sha256_text(record.get("attestation_sha256"), f"{label} attestation")
    unsigned = dict(record)
    unsigned.pop("attestation_sha256", None)
    if _canonical_digest(unsigned) != attestation:
        raise NuisanceValidationError(f"{label} attestation differs")
    return record


def _normalize_v2_inventory(
    records_value: Sequence[Mapping[str, Any]],
    media_files_value: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(records_value, Sequence) or isinstance(
        records_value, (str, bytes)
    ):
        raise NuisanceValidationError("v2 record inventory must be a sequence")
    records: list[dict[str, str]] = []
    record_ids: set[str] = set()
    for index, raw_record in enumerate(records_value):
        record = _mapping(raw_record, f"v2 inventory record {index}")
        _exact_keys(
            record,
            {"record_id", "component_id"},
            f"v2 inventory record {index}",
        )
        record_id = _strict_identifier(
            record.get("record_id"), f"v2 inventory record {index} ID"
        )
        component_id = _strict_identifier(
            record.get("component_id"), f"v2 inventory record {index} component"
        )
        if record_id in record_ids:
            raise NuisanceValidationError("v2 inventory record IDs must be unique")
        record_ids.add(record_id)
        records.append({"record_id": record_id, "component_id": component_id})
    if not records:
        raise NuisanceValidationError("v2 record inventory must be nonempty")
    if records != sorted(records, key=lambda item: item["record_id"]):
        raise NuisanceValidationError("v2 record inventory must be in record-ID order")
    components = {record["component_id"] for record in records}
    if len(components) < 2:
        raise NuisanceValidationError("v2 inventory needs at least two components")

    if not isinstance(media_files_value, Sequence) or isinstance(
        media_files_value, (str, bytes)
    ):
        raise NuisanceValidationError("v2 media inventory must be a sequence")
    media_files: list[dict[str, Any]] = []
    relative_paths: set[str] = set()
    for index, raw_media in enumerate(media_files_value):
        media = _mapping(raw_media, f"v2 media file {index}")
        _exact_keys(
            media,
            {"relative_path", "size_bytes", "sha256"},
            f"v2 media file {index}",
        )
        relative_path = _public_relative_path(
            media.get("relative_path"), f"v2 media file {index} path"
        )
        if relative_path in relative_paths:
            raise NuisanceValidationError("v2 media paths must be unique")
        relative_paths.add(relative_path)
        media_files.append(
            {
                "relative_path": relative_path,
                "size_bytes": _positive_integer(
                    media.get("size_bytes"), f"v2 media file {index} size"
                ),
                "sha256": _sha256_text(
                    media.get("sha256"), f"v2 media file {index} digest"
                ),
            }
        )
    if not media_files:
        raise NuisanceValidationError("v2 media inventory must be nonempty")
    if media_files != sorted(media_files, key=lambda item: item["relative_path"]):
        raise NuisanceValidationError("v2 media inventory must be in path order")
    inventory_body = {"records": records, "media_files": media_files}
    return {
        **inventory_body,
        "record_count": len(records),
        "component_count": len(components),
        "media_file_count": len(media_files),
        "record_inventory_sha256": _canonical_digest(records),
        "media_inventory_sha256": _canonical_digest(media_files),
        "inventory_sha256": _canonical_digest(inventory_body),
    }


_V2_CONTROL_SECTIONS = (
    "question_only_conflict",
    "question_blind_audiovisual_mismatch",
    "remux_noninferiority",
    "orientation_symmetry",
    "option_permutation_equivalence",
    "shuffled_question",
)


def project_shortcut_control_plan_v2(value: Any) -> dict[str, Any]:
    """Project a completed shortcut record onto its output-free frozen inputs."""

    shortcut = _mapping(value, "shortcut output for control-plan projection")
    projected: dict[str, Any] = {}
    output_fields = {
        "primary_output",
        "replay_output",
        "clean_output",
        "remux_output",
        "audio_over_video_output",
        "video_over_audio_output",
        "outputs",
        "aligned_output",
        "shuffled_output",
    }
    for section_name in _V2_CONTROL_SECTIONS:
        section = _mapping(shortcut.get(section_name), f"{section_name} section")
        records = section.get("records")
        if not isinstance(records, list):
            raise NuisanceValidationError(f"{section_name} records must be a list")
        projected[section_name] = {
            key: copy.deepcopy(item)
            for key, item in section.items()
            if key != "records"
        }
        projected[section_name]["records"] = [
            {
                key: copy.deepcopy(item)
                for key, item in _mapping(
                    record, f"{section_name} projection record"
                ).items()
                if key not in output_fields
            }
            for record in records
        ]
    return projected


def _validate_control_plan_v2(
    value: Any,
    *,
    expected_records: Sequence[Mapping[str, Any]],
    media_files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the complete output-free plan and its shared semantic inventory."""

    plan = _mapping(value, "v2 control plan")
    _exact_keys(plan, set(_V2_CONTROL_SECTIONS), "v2 control plan")
    forbidden_fields = {
        "primary_output",
        "replay_output",
        "clean_output",
        "remux_output",
        "audio_over_video_output",
        "video_over_audio_output",
        "outputs",
        "aligned_output",
        "shuffled_output",
    }

    def reject_outputs(item: Any) -> None:
        if isinstance(item, Mapping):
            if set(item) & forbidden_fields:
                raise NuisanceValidationError(
                    "v2 control plan contains evaluated model outputs"
                )
            for child in item.values():
                reject_outputs(child)
        elif isinstance(item, list):
            for child in item:
                reject_outputs(child)

    reject_outputs(plan)
    expected_inventory = {
        record["record_id"]: record["component_id"] for record in expected_records
    }
    relation_maps: dict[str, dict[str, str]] = {}
    media_references: dict[str, str] = {}

    def records_for(section_name: str) -> list[Mapping[str, Any]]:
        section = _mapping(plan.get(section_name), f"v2 {section_name} plan")
        records = section.get("records")
        if not isinstance(records, list) or not records:
            raise NuisanceValidationError(
                f"v2 {section_name} plan records must be nonempty"
            )
        inventory: dict[str, str] = {}
        normalized: list[Mapping[str, Any]] = []
        for index, raw in enumerate(records):
            record = _mapping(raw, f"v2 {section_name} plan record {index}")
            record_id = _strict_identifier(
                record.get("record_id"), f"v2 {section_name} record ID"
            )
            component_id = _strict_identifier(
                record.get("component_id"), f"v2 {section_name} component ID"
            )
            if record_id in inventory:
                raise NuisanceValidationError(
                    f"v2 {section_name} plan record IDs must be unique"
                )
            inventory[record_id] = component_id
            normalized.append(record)
        if inventory != expected_inventory:
            raise NuisanceValidationError(f"v2 {section_name} plan inventory differs")
        return normalized

    question_only_section = _mapping(
        plan["question_only_conflict"], "v2 question-only plan"
    )
    _exact_keys(
        question_only_section,
        {"transform", "records"},
        "v2 question-only plan",
    )
    if question_only_section["transform"] != "remove_audio_and_video":
        raise NuisanceValidationError("v2 question-only transform differs")
    question_only_records = records_for("question_only_conflict")
    question_only_relations: dict[str, str] = {}
    for index, record in enumerate(question_only_records):
        _exact_keys(
            record,
            {
                "record_id",
                "component_id",
                "gold_relation",
                "request",
                "request_sha256",
            },
            f"v2 question-only plan record {index}",
        )
        if record["gold_relation"] not in {"AGREE", "CONFLICT"}:
            raise NuisanceValidationError("v2 question-only gold relation differs")
        request = _validate_replay_request_v2(record["request"])
        if request["task"] != "question_only_conflict":
            raise NuisanceValidationError("v2 question-only task differs")
        if record["request_sha256"] != _canonical_digest(request):
            raise NuisanceValidationError("v2 question-only request digest differs")
        question_only_relations[record["record_id"]] = record["gold_relation"]
    relation_maps["question_only"] = question_only_relations

    question_blind_section = _mapping(
        plan["question_blind_audiovisual_mismatch"], "v2 question-blind plan"
    )
    _exact_keys(
        question_blind_section,
        {"transform", "threshold", "threshold_record_sha256", "records"},
        "v2 question-blind plan",
    )
    if question_blind_section["transform"] != "remove_question_and_options":
        raise NuisanceValidationError("v2 question-blind transform differs")
    _exact_float(
        question_blind_section["threshold"],
        "v2 question-blind threshold",
        minimum=0.0,
        maximum=1.0,
    )
    _sha256_text(
        question_blind_section["threshold_record_sha256"],
        "v2 question-blind threshold record",
    )
    question_blind_relations: dict[str, str] = {}
    for index, record in enumerate(records_for("question_blind_audiovisual_mismatch")):
        _exact_keys(
            record,
            {
                "record_id",
                "component_id",
                "expected_relation",
                "request",
                "request_sha256",
            },
            f"v2 question-blind plan record {index}",
        )
        if record["expected_relation"] not in {"matched", "mismatched"}:
            raise NuisanceValidationError("v2 question-blind relation differs")
        request = _validate_question_blind_request_v2(
            record["request"], f"v2 question-blind plan record {index} request"
        )
        if record["request_sha256"] != _canonical_digest(request):
            raise NuisanceValidationError("v2 question-blind request digest differs")
        question_blind_relations[record["record_id"]] = (
            "AGREE" if record["expected_relation"] == "matched" else "CONFLICT"
        )
        for modality in ("audio", "video"):
            media = request[modality]
            prior = media_references.setdefault(media["relative_path"], media["sha256"])
            if prior != media["sha256"]:
                raise NuisanceValidationError("v2 control media digest differs")
    relation_maps["question_blind"] = question_blind_relations

    section_shapes = {
        "remux_noninferiority": (
            {"transform", "epsilon_p_gold", "epsilon_source", "records"},
            "container_remux_only",
        ),
        "orientation_symmetry": (
            {"transform", "records"},
            "reverse_modality_precedence",
        ),
        "option_permutation_equivalence": (
            {"transform", "records"},
            "cyclic_option_rotation",
        ),
        "shuffled_question": (
            {"transform", "records"},
            "cross_component_question_derangement",
        ),
    }
    for section_name, (keys, transform) in section_shapes.items():
        section = _mapping(plan[section_name], f"v2 {section_name} plan")
        _exact_keys(section, keys, f"v2 {section_name} plan")
        if section["transform"] != transform:
            raise NuisanceValidationError(f"v2 {section_name} transform differs")
        records_for(section_name)

    remux_section = _mapping(plan["remux_noninferiority"], "v2 remux plan")
    _exact_float(
        remux_section["epsilon_p_gold"],
        "v2 remux epsilon",
        minimum=0.0,
        maximum=1.0,
    )
    epsilon_source = _mapping(remux_section["epsilon_source"], "v2 remux source")
    _exact_keys(
        epsilon_source,
        {"quantile", "semantic_fit_output_sha256"},
        "v2 remux source",
    )
    if _exact_float(epsilon_source["quantile"], "v2 remux quantile") != 0.95:
        raise NuisanceValidationError("v2 remux quantile differs")
    _sha256_text(
        epsilon_source["semantic_fit_output_sha256"],
        "v2 remux semantic-fit digest",
    )

    relation_record_fields = {
        "orientation_symmetry": (
            "audio_over_video_request",
            "video_over_audio_request",
        ),
        "shuffled_question": ("aligned_request", "shuffled_request"),
    }
    for section_name, prefixes in relation_record_fields.items():
        relation_map: dict[str, str] = {}
        for index, record in enumerate(records_for(section_name)):
            required = {"record_id", "component_id", "gold_relation"}
            for prefix in prefixes:
                required |= {prefix, f"{prefix}_sha256"}
            _exact_keys(record, required, f"v2 {section_name} plan record {index}")
            gold = record["gold_relation"]
            if gold not in {"AGREE", "CONFLICT"}:
                raise NuisanceValidationError(f"v2 {section_name} gold differs")
            relation_map[record["record_id"]] = gold
            for prefix in prefixes:
                request = _validate_control_request_v2(
                    record[prefix], f"v2 {section_name} {prefix}"
                )
                if request["task"] != "conflict_relation":
                    raise NuisanceValidationError(
                        f"v2 {section_name} request task differs"
                    )
                if record[f"{prefix}_sha256"] != _canonical_digest(request):
                    raise NuisanceValidationError(
                        f"v2 {section_name} request digest differs"
                    )
                for modality in ("audio", "video"):
                    media = request[modality]
                    prior = media_references.setdefault(
                        media["relative_path"], media["sha256"]
                    )
                    if prior != media["sha256"]:
                        raise NuisanceValidationError("v2 control media digest differs")
        relation_maps[section_name] = relation_map

    for index, record in enumerate(records_for("remux_noninferiority")):
        _exact_keys(
            record,
            {
                "record_id",
                "component_id",
                "gold_answer",
                "clean_request",
                "clean_request_sha256",
                "remux_request",
                "remux_request_sha256",
            },
            f"v2 remux plan record {index}",
        )
        if record["gold_answer"] not in {"A", "B", "C"}:
            raise NuisanceValidationError("v2 remux gold answer differs")
        for prefix in ("clean", "remux"):
            request = _validate_control_request_v2(
                record[f"{prefix}_request"], f"v2 remux {prefix} request"
            )
            if request["task"] != "source_choice":
                raise NuisanceValidationError("v2 remux request task differs")
            if record[f"{prefix}_request_sha256"] != _canonical_digest(request):
                raise NuisanceValidationError("v2 remux request digest differs")
            for modality in ("audio", "video"):
                media = request[modality]
                prior = media_references.setdefault(
                    media["relative_path"], media["sha256"]
                )
                if prior != media["sha256"]:
                    raise NuisanceValidationError("v2 control media digest differs")

    permutation_relations: dict[str, str] = {}
    for index, record in enumerate(records_for("option_permutation_equivalence")):
        _exact_keys(
            record,
            {
                "record_id",
                "component_id",
                "gold_relation",
                "requests",
                "request_sha256s",
            },
            f"v2 option-permutation plan record {index}",
        )
        gold = record["gold_relation"]
        if gold not in {"AGREE", "CONFLICT"}:
            raise NuisanceValidationError("v2 option-permutation gold differs")
        permutation_relations[record["record_id"]] = gold
        requests = _mapping(record["requests"], "v2 option-permutation requests")
        request_sha256s = _mapping(
            record["request_sha256s"], "v2 option-permutation request digests"
        )
        _exact_keys(requests, {"0", "1", "2"}, "v2 option-permutation requests")
        _exact_keys(
            request_sha256s,
            {"0", "1", "2"},
            "v2 option-permutation request digests",
        )
        option_orders = []
        for permutation in ("0", "1", "2"):
            request = _validate_control_request_v2(
                requests[permutation], "v2 option-permutation request"
            )
            if request["task"] != "conflict_relation":
                raise NuisanceValidationError(
                    "v2 option-permutation request task differs"
                )
            if request_sha256s[permutation] != _canonical_digest(request):
                raise NuisanceValidationError(
                    "v2 option-permutation request digest differs"
                )
            option_orders.append(tuple(request["options"]))
            for modality in ("audio", "video"):
                media = request[modality]
                prior = media_references.setdefault(
                    media["relative_path"], media["sha256"]
                )
                if prior != media["sha256"]:
                    raise NuisanceValidationError("v2 control media digest differs")
        if len(set(option_orders)) != 3 or any(
            sorted(order) != sorted(option_orders[0]) for order in option_orders[1:]
        ):
            raise NuisanceValidationError(
                "v2 option-permutation orders are not exact permutations"
            )
    relation_maps["option_permutation_equivalence"] = permutation_relations

    if any(mapping != question_only_relations for mapping in relation_maps.values()):
        raise NuisanceValidationError("v2 shared gold-relation inventory differs")
    expected_media = {media["relative_path"]: media["sha256"] for media in media_files}
    if media_references != expected_media:
        raise NuisanceValidationError("v2 control media assignment inventory differs")
    return copy.deepcopy(dict(plan))


def build_shortcut_preregistration_v2(
    *,
    configuration_sha256: str,
    gate_implementation_bundle_sha256: str,
    dependency_digests: Mapping[str, str],
    replay_worker_digests: Mapping[str, str],
    records: Sequence[Mapping[str, Any]],
    media_files: Sequence[Mapping[str, Any]],
    metadata_records: Sequence[Mapping[str, Any]],
    control_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a pre-run v2 record that contains no evaluated model outputs."""

    dependencies = _mapping(dependency_digests, "v2 dependency digests")
    _exact_keys(dependencies, _V2_DEPENDENCY_DIGEST_KEYS, "v2 dependency digests")
    normalized_dependencies = {
        key: _sha256_text(dependencies[key], f"v2 dependency {key}")
        for key in sorted(_V2_DEPENDENCY_DIGEST_KEYS)
    }
    worker = _mapping(replay_worker_digests, "v2 replay-worker digests")
    _exact_keys(worker, _V2_WORKER_DIGEST_KEYS, "v2 replay-worker digests")
    normalized_worker = {
        key: _sha256_text(worker[key], f"v2 replay-worker {key}")
        for key in sorted(_V2_WORKER_DIGEST_KEYS)
    }
    inventory = _normalize_v2_inventory(records, media_files)
    normalized_metadata_records = _normalize_metadata_records_v2(metadata_records)
    normalized_control_plan = _validate_control_plan_v2(
        control_plan,
        expected_records=inventory["records"],
        media_files=inventory["media_files"],
    )
    value = {
        "schema": SHORTCUT_PREREGISTRATION_SCHEMA_V2,
        "status": "preregistered_not_run",
        "input_digests": {
            "configuration_sha256": _sha256_text(
                configuration_sha256, "v2 configuration digest"
            ),
            "gate_implementation_bundle_sha256": _sha256_text(
                gate_implementation_bundle_sha256,
                "v2 gate implementation-bundle digest",
            ),
            **normalized_dependencies,
        },
        "replay_worker": normalized_worker,
        "inventory": inventory,
        "metadata_records_sha256": _canonical_digest(normalized_metadata_records),
        "control_plan": normalized_control_plan,
        "control_plan_sha256": _canonical_digest(normalized_control_plan),
        "chronology": {
            "evaluation_outputs_access": "forbidden",
            "model_replay_status": "not_run",
        },
    }
    return _attest_v2(value)


def validate_shortcut_preregistration_v2(
    value: Any,
    *,
    expected_configuration_sha256: str,
    expected_gate_implementation_bundle_sha256: str,
) -> dict[str, Any]:
    """Validate an output-free v2 preregistration and its exact inventory."""

    preregistration = _validate_attested_v2(value, "v2 shortcut preregistration")
    _exact_keys(
        preregistration,
        {
            "schema",
            "status",
            "input_digests",
            "replay_worker",
            "inventory",
            "metadata_records_sha256",
            "control_plan",
            "control_plan_sha256",
            "chronology",
            "attestation_sha256",
        },
        "v2 shortcut preregistration",
    )
    if preregistration.get("schema") != SHORTCUT_PREREGISTRATION_SCHEMA_V2:
        raise NuisanceValidationError("v2 shortcut preregistration schema differs")
    if preregistration.get("status") != "preregistered_not_run":
        raise NuisanceValidationError("v2 shortcut preregistration status differs")
    chronology = _mapping(preregistration.get("chronology"), "v2 shortcut chronology")
    if chronology != {
        "evaluation_outputs_access": "forbidden",
        "model_replay_status": "not_run",
    }:
        raise NuisanceValidationError("v2 shortcut chronology differs")
    digests = _mapping(
        preregistration.get("input_digests"), "v2 shortcut input digests"
    )
    expected_digest_keys = _V2_DEPENDENCY_DIGEST_KEYS | {
        "configuration_sha256",
        "gate_implementation_bundle_sha256",
    }
    _exact_keys(digests, expected_digest_keys, "v2 shortcut input digests")
    for key in expected_digest_keys:
        _sha256_text(digests.get(key), f"v2 shortcut {key}")
    if digests["configuration_sha256"] != _sha256_text(
        expected_configuration_sha256, "expected v2 configuration digest"
    ):
        raise NuisanceValidationError("v2 configuration binding differs")
    if digests["gate_implementation_bundle_sha256"] != _sha256_text(
        expected_gate_implementation_bundle_sha256,
        "expected v2 gate implementation-bundle digest",
    ):
        raise NuisanceValidationError("v2 gate implementation binding differs")
    worker = _mapping(preregistration.get("replay_worker"), "v2 replay-worker digests")
    _exact_keys(worker, _V2_WORKER_DIGEST_KEYS, "v2 replay-worker digests")
    for key in _V2_WORKER_DIGEST_KEYS:
        _sha256_text(worker.get(key), f"v2 replay-worker {key}")
    inventory = _mapping(preregistration.get("inventory"), "v2 inventory")
    _exact_keys(
        inventory,
        {
            "records",
            "media_files",
            "record_count",
            "component_count",
            "media_file_count",
            "record_inventory_sha256",
            "media_inventory_sha256",
            "inventory_sha256",
        },
        "v2 inventory",
    )
    expected_inventory = _normalize_v2_inventory(
        inventory.get("records"), inventory.get("media_files")
    )
    if dict(inventory) != expected_inventory:
        raise NuisanceValidationError("v2 inventory summary differs")
    _sha256_text(
        preregistration.get("metadata_records_sha256"),
        "v2 preregistered metadata-record digest",
    )
    control_plan = _validate_control_plan_v2(
        preregistration.get("control_plan"),
        expected_records=expected_inventory["records"],
        media_files=expected_inventory["media_files"],
    )
    if preregistration.get("control_plan_sha256") != _canonical_digest(control_plan):
        raise NuisanceValidationError("v2 control-plan digest differs")
    serialized = json.dumps(preregistration, ensure_ascii=False, sort_keys=True)
    if any(name in serialized for name in ("primary_output", "replay_output")):
        raise NuisanceValidationError(
            "v2 preregistration contains evaluated model outputs"
        )
    return copy.deepcopy(dict(preregistration))


def authenticate_v2_media_inventory(
    media_root: pathlib.Path, media_files_value: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Authenticate the exact read-only derived-media set at one point in time."""

    root_input = pathlib.Path(media_root)
    if root_input.is_symlink() or not root_input.is_dir():
        raise NuisanceValidationError(
            "v2 derived-media root must be a non-symlink directory"
        )
    root = root_input.resolve(strict=True)
    if not isinstance(media_files_value, Sequence) or isinstance(
        media_files_value, (str, bytes)
    ):
        raise NuisanceValidationError("v2 media inventory must be a sequence")
    normalized: list[dict[str, Any]] = []
    expected_paths: set[str] = set()
    for index, raw_media in enumerate(media_files_value):
        media = _mapping(raw_media, f"v2 media file {index}")
        _exact_keys(
            media,
            {"relative_path", "size_bytes", "sha256"},
            f"v2 media file {index}",
        )
        relative_path = _public_relative_path(
            media.get("relative_path"), f"v2 media file {index} path"
        )
        if relative_path in expected_paths:
            raise NuisanceValidationError("v2 media paths must be unique")
        expected_paths.add(relative_path)
        expected_size = _positive_integer(
            media.get("size_bytes"), f"v2 media file {index} size"
        )
        expected_digest = _sha256_text(
            media.get("sha256"), f"v2 media file {index} digest"
        )
        path = root.joinpath(*pathlib.PurePosixPath(relative_path).parts)
        try:
            resolved = path.resolve(strict=True)
            file_stat = path.stat()
        except OSError as exc:
            raise NuisanceValidationError(
                f"v2 derived media is missing: {relative_path}"
            ) from exc
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise NuisanceValidationError("v2 media path escapes its root") from exc
        if path.is_symlink() or not stat.S_ISREG(file_stat.st_mode):
            raise NuisanceValidationError(
                f"v2 derived media is not a regular file: {relative_path}"
            )
        if stat.S_IMODE(file_stat.st_mode) & 0o222:
            raise NuisanceValidationError(
                f"v2 derived media is writable: {relative_path}"
            )
        if file_stat.st_size != expected_size:
            raise NuisanceValidationError(
                f"v2 derived media size differs: {relative_path}"
            )
        if _sha256_path(path) != expected_digest:
            raise NuisanceValidationError(
                f"v2 derived media digest differs: {relative_path}"
            )
        normalized.append(
            {
                "relative_path": relative_path,
                "size_bytes": expected_size,
                "sha256": expected_digest,
            }
        )
    normalized.sort(key=lambda item: item["relative_path"])
    observed_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if observed_paths != expected_paths:
        raise NuisanceValidationError("v2 derived-media membership differs")
    return {
        "media_file_count": len(normalized),
        "media_files": normalized,
        "media_inventory_sha256": _canonical_digest(normalized),
    }


def _validate_shortcut_power_dgp_v2(
    value: Any, *, expected_remux_p_gold_epsilon: float | None = None
) -> dict[str, Any]:
    dgp = _mapping(value, "v2 shortcut power DGP")
    _exact_keys(
        dgp,
        {
            "schema",
            "analysis_unit",
            "endpoint_designs",
            "decision_statistic",
            "source",
        },
        "v2 shortcut power DGP",
    )
    if dgp.get("schema") != SHORTCUT_POWER_DGP_SCHEMA_V2:
        raise NuisanceValidationError("v2 shortcut power DGP schema differs")
    if dgp.get("analysis_unit") != "endpoint_level_simulation_draw":
        raise NuisanceValidationError("v2 shortcut power DGP analysis unit differs")
    _nonempty_text(dgp.get("source"), "v2 shortcut power DGP source")
    designs = _mapping(
        dgp.get("endpoint_designs"),
        "v2 shortcut power DGP endpoint designs",
    )
    _exact_keys(
        designs,
        set(_V2_POWER_ENDPOINTS),
        "v2 shortcut power DGP endpoint designs",
    )
    normalized_designs: dict[str, dict[str, Any]] = {}
    for endpoint in _V2_POWER_ENDPOINTS:
        design = _mapping(
            designs[endpoint], f"v2 shortcut power DGP design for {endpoint}"
        )
        _exact_keys(
            design,
            {
                "sample_size_components",
                "alternative_value",
                "marginal_standard_deviation",
                "within_component_correlation",
                "within_endpoint_statistic_count",
                "within_endpoint_statistic_correlation",
                "critical_value",
                "decision_rule",
            },
            f"v2 shortcut power DGP design for {endpoint}",
        )
        sample_size = _positive_integer(
            design.get("sample_size_components"),
            f"v2 shortcut power DGP sample size for {endpoint}",
        )
        alternative = _exact_float(
            design.get("alternative_value"),
            f"v2 shortcut power DGP alternative for {endpoint}",
        )
        marginal_standard_deviation = _exact_float(
            design.get("marginal_standard_deviation"),
            f"v2 shortcut power DGP standard deviation for {endpoint}",
            minimum=0.0,
        )
        if marginal_standard_deviation <= 0.0:
            raise NuisanceValidationError(
                f"v2 shortcut power DGP standard deviation for {endpoint} must be positive"
            )
        correlation = _exact_float(
            design.get("within_component_correlation"),
            f"v2 shortcut power DGP correlation for {endpoint}",
            minimum=-1.0,
            maximum=1.0,
        )
        if correlation in {-1.0, 1.0}:
            raise NuisanceValidationError(
                f"v2 shortcut power DGP correlation for {endpoint} must be open"
            )
        within_endpoint_count = _positive_integer(
            design.get("within_endpoint_statistic_count"),
            f"v2 shortcut power DGP within-endpoint count for {endpoint}",
        )
        expected_within_endpoint_count = (
            6 if endpoint == "metadata_balanced_accuracy" else 1
        )
        if within_endpoint_count != expected_within_endpoint_count:
            raise NuisanceValidationError(
                f"v2 shortcut power DGP within-endpoint count differs for {endpoint}"
            )
        within_endpoint_correlation = _exact_float(
            design.get("within_endpoint_statistic_correlation"),
            f"v2 shortcut power DGP within-endpoint correlation for {endpoint}",
            minimum=0.0,
            maximum=1.0,
        )
        if within_endpoint_correlation == 1.0:
            raise NuisanceValidationError(
                f"v2 shortcut power DGP within-endpoint correlation for {endpoint} must be open"
            )
        if within_endpoint_count == 1 and within_endpoint_correlation != 0.0:
            raise NuisanceValidationError(
                f"v2 shortcut power DGP singleton correlation differs for {endpoint}"
            )
        critical_value = _exact_float(
            design.get("critical_value"),
            f"v2 shortcut power DGP critical value for {endpoint}",
        )
        expected_critical_value = statistics.NormalDist().inv_cdf(
            1.0 - 0.05 / (len(_V2_POWER_ENDPOINTS) * within_endpoint_count)
        )
        if critical_value != expected_critical_value:
            raise NuisanceValidationError(
                f"v2 shortcut power DGP critical value differs for {endpoint}"
            )
        rule = _mapping(
            design.get("decision_rule"),
            f"v2 shortcut power DGP decision rule for {endpoint}",
        )
        expected_rule = copy.deepcopy(_V2_POWER_ENDPOINT_RULES[endpoint])
        if endpoint == "remux_p_gold_difference":
            if expected_remux_p_gold_epsilon is None:
                supplied_boundary = rule.get("lower_boundary")
                if (
                    type(supplied_boundary) is not float
                    or not math.isfinite(supplied_boundary)
                    or supplied_boundary >= 0.0
                ):
                    raise NuisanceValidationError(
                        "v2 shortcut power DGP remux p-gold boundary must be negative"
                    )
                expected_rule["lower_boundary"] = supplied_boundary
            else:
                epsilon = _exact_float(
                    expected_remux_p_gold_epsilon,
                    "expected v2 shortcut power remux p-gold epsilon",
                    minimum=0.0,
                )
                if epsilon <= 0.0:
                    raise NuisanceValidationError(
                        "expected v2 shortcut power remux p-gold epsilon must be positive"
                    )
                expected_rule["lower_boundary"] = -epsilon
        _exact_keys(
            rule,
            set(expected_rule),
            f"v2 shortcut power DGP decision rule for {endpoint}",
        )
        normalized_rule = {"type": rule.get("type")}
        for boundary_name in set(expected_rule) - {"type"}:
            normalized_rule[boundary_name] = _exact_float(
                rule.get(boundary_name),
                f"v2 shortcut power DGP {boundary_name} for {endpoint}",
            )
        if normalized_rule != expected_rule:
            if endpoint == "remux_p_gold_difference":
                raise NuisanceValidationError(
                    "v2 shortcut power DGP remux p-gold boundary differs"
                )
            raise NuisanceValidationError(
                f"v2 shortcut power DGP decision rule differs for {endpoint}"
            )
        rule_type = normalized_rule["type"]
        if (
            rule_type == "upper_confidence_bound_below"
            and alternative >= normalized_rule["upper_boundary"]
        ):
            raise NuisanceValidationError(
                f"v2 shortcut power DGP alternative is not favorable for {endpoint}"
            )
        if (
            rule_type == "lower_confidence_bound_above"
            and alternative <= normalized_rule["lower_boundary"]
        ):
            raise NuisanceValidationError(
                f"v2 shortcut power DGP alternative is not favorable for {endpoint}"
            )
        if rule_type == "confidence_interval_within" and not (
            normalized_rule["lower_boundary"]
            < alternative
            < normalized_rule["upper_boundary"]
        ):
            raise NuisanceValidationError(
                f"v2 shortcut power DGP alternative is not favorable for {endpoint}"
            )
        normalized_designs[endpoint] = {
            "sample_size_components": sample_size,
            "alternative_value": alternative,
            "marginal_standard_deviation": marginal_standard_deviation,
            "within_component_correlation": correlation,
            "within_endpoint_statistic_count": within_endpoint_count,
            "within_endpoint_statistic_correlation": within_endpoint_correlation,
            "critical_value": critical_value,
            "decision_rule": normalized_rule,
        }

    statistic = _mapping(
        dgp.get("decision_statistic"), "v2 shortcut power decision statistic"
    )
    _exact_keys(
        statistic,
        {
            "name",
            "sampling_model",
            "familywise_alpha",
            "multiplicity_correction",
        },
        "v2 shortcut power decision statistic",
    )
    if statistic.get("name") != "endpoint_specific_gaussian_component_mean_bounds":
        raise NuisanceValidationError("v2 shortcut power decision statistic differs")
    if statistic.get("sampling_model") != "paired_gaussian_known_variance":
        raise NuisanceValidationError("v2 shortcut power sampling model differs")
    familywise_alpha = _probability(
        statistic.get("familywise_alpha"),
        "v2 shortcut power familywise alpha",
        open_interval=True,
    )
    if familywise_alpha != 0.05:
        raise NuisanceValidationError("v2 shortcut power familywise alpha differs")
    if statistic.get("multiplicity_correction") != "bonferroni":
        raise NuisanceValidationError(
            "v2 shortcut power multiplicity correction differs"
        )
    return {
        "schema": SHORTCUT_POWER_DGP_SCHEMA_V2,
        "analysis_unit": "endpoint_level_simulation_draw",
        "endpoint_designs": normalized_designs,
        "decision_statistic": {
            "name": "endpoint_specific_gaussian_component_mean_bounds",
            "sampling_model": "paired_gaussian_known_variance",
            "familywise_alpha": familywise_alpha,
            "multiplicity_correction": "bonferroni",
        },
        "source": dgp["source"],
    }


def _expected_shortcut_power_simulation_v2(
    dgp: Mapping[str, Any], *, seed: int, repetitions: int
) -> dict[str, dict[str, list[Any]]]:
    """Generate exact endpoint statistics and decisions from the frozen DGP."""

    seed_value = _nonnegative_integer(seed, "v2 shortcut power seed")
    repetitions_value = _positive_integer(
        repetitions, "v2 power simulation repetitions"
    )
    dgp_value = _validate_shortcut_power_dgp_v2(dgp)
    dgp_sha256 = _canonical_digest(dgp_value)
    draws: dict[str, list[int]] = {}
    statistics_by_endpoint: dict[str, list[float]] = {}
    denominator = float(1 << 64)
    for endpoint in _V2_POWER_ENDPOINTS:
        design = dgp_value["endpoint_designs"][endpoint]
        sample_size = design["sample_size_components"]
        alternative = design["alternative_value"]
        marginal_standard_deviation = design["marginal_standard_deviation"]
        correlation = design["within_component_correlation"]
        standard_error = marginal_standard_deviation * math.sqrt(
            2.0 * (1.0 - correlation) / sample_size
        )
        within_endpoint_count = design["within_endpoint_statistic_count"]
        within_endpoint_correlation = design["within_endpoint_statistic_correlation"]
        critical_value = design["critical_value"]
        decision_rule = design["decision_rule"]
        endpoint_draws = []
        endpoint_statistics = []
        for draw_index in range(repetitions_value):
            counter_prefix = (
                f"{POWER_SIMULATION_ALGORITHM_V2}\0{dgp_sha256}\0{seed_value}\0"
                f"{endpoint}\0{draw_index}"
            )
            common_first = hashlib.sha256(
                f"{counter_prefix}\0common_normal_1".encode()
            ).digest()
            common_second = hashlib.sha256(
                f"{counter_prefix}\0common_normal_2".encode()
            ).digest()
            common_uniform_1 = (
                int.from_bytes(common_first[:8], "big") + 0.5
            ) / denominator
            common_uniform_2 = (
                int.from_bytes(common_second[:8], "big") + 0.5
            ) / denominator
            common_normal = math.sqrt(-2.0 * math.log(common_uniform_1)) * math.cos(
                2.0 * math.pi * common_uniform_2
            )
            estimates = []
            for statistic_index in range(within_endpoint_count):
                first = hashlib.sha256(
                    f"{counter_prefix}\0statistic_{statistic_index}_normal_1".encode()
                ).digest()
                second = hashlib.sha256(
                    f"{counter_prefix}\0statistic_{statistic_index}_normal_2".encode()
                ).digest()
                uniform_1 = (int.from_bytes(first[:8], "big") + 0.5) / denominator
                uniform_2 = (int.from_bytes(second[:8], "big") + 0.5) / denominator
                independent_normal = math.sqrt(-2.0 * math.log(uniform_1)) * math.cos(
                    2.0 * math.pi * uniform_2
                )
                standard_normal = (
                    math.sqrt(within_endpoint_correlation) * common_normal
                    + math.sqrt(1.0 - within_endpoint_correlation) * independent_normal
                )
                estimates.append(alternative + standard_error * standard_normal)
            if decision_rule["type"] == "upper_confidence_bound_below":
                estimate = max(estimates)
                upper_bound = estimate + critical_value * standard_error
                detected = upper_bound < decision_rule["upper_boundary"]
            elif decision_rule["type"] == "lower_confidence_bound_above":
                estimate = min(estimates)
                lower_bound = estimate - critical_value * standard_error
                detected = lower_bound > decision_rule["lower_boundary"]
            else:
                midpoint = (
                    decision_rule["lower_boundary"] + decision_rule["upper_boundary"]
                ) / 2.0
                estimate = max(estimates, key=lambda item: abs(item - midpoint))
                lower_bound = estimate - critical_value * standard_error
                upper_bound = estimate + critical_value * standard_error
                detected = (
                    lower_bound >= decision_rule["lower_boundary"]
                    and upper_bound <= decision_rule["upper_boundary"]
                )
            endpoint_statistics.append(float(estimate))
            endpoint_draws.append(int(detected))
        draws[endpoint] = endpoint_draws
        statistics_by_endpoint[endpoint] = endpoint_statistics
    return {
        "endpoint_statistics": statistics_by_endpoint,
        "endpoint_draws": draws,
    }


def _expected_shortcut_power_draws_v2(
    dgp: Mapping[str, Any], *, seed: int, repetitions: int
) -> dict[str, list[int]]:
    """Return the independently generated endpoint decisions."""

    return _expected_shortcut_power_simulation_v2(
        dgp, seed=seed, repetitions=repetitions
    )["endpoint_draws"]


def _validate_shortcut_power_statistics_v2(
    value: Any, *, repetitions: int
) -> dict[str, list[float]]:
    statistics_value = _mapping(value, "v2 shortcut power endpoint statistics")
    _exact_keys(
        statistics_value,
        set(_V2_POWER_ENDPOINTS),
        "v2 shortcut power endpoint statistics",
    )
    normalized: dict[str, list[float]] = {}
    for endpoint in _V2_POWER_ENDPOINTS:
        endpoint_statistics = statistics_value[endpoint]
        if (
            not isinstance(endpoint_statistics, list)
            or len(endpoint_statistics) != repetitions
            or any(
                type(statistic_value) is not float or not math.isfinite(statistic_value)
                for statistic_value in endpoint_statistics
            )
        ):
            raise NuisanceValidationError(
                f"v2 shortcut power statistics differ for {endpoint}"
            )
        normalized[endpoint] = list(endpoint_statistics)
    return normalized


def _validate_shortcut_power_draws_v2(
    value: Any, *, repetitions: int
) -> dict[str, list[int]]:
    draws = _mapping(value, "v2 shortcut power endpoint draws")
    _exact_keys(draws, set(_V2_POWER_ENDPOINTS), "v2 shortcut power endpoint draws")
    normalized: dict[str, list[int]] = {}
    for endpoint in _V2_POWER_ENDPOINTS:
        endpoint_draws = draws[endpoint]
        if (
            not isinstance(endpoint_draws, list)
            or len(endpoint_draws) != repetitions
            or any(
                type(draw) is not int or draw not in {0, 1} for draw in endpoint_draws
            )
        ):
            raise NuisanceValidationError(
                f"v2 shortcut power draws differ for {endpoint}"
            )
        normalized[endpoint] = list(endpoint_draws)
    return normalized


def _run_shortcut_power_simulator_v2(
    spec: SubprocessPowerSimulatorSpec,
    *,
    inventory_sha256: str,
    dgp: Mapping[str, Any],
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    if type(spec) is not SubprocessPowerSimulatorSpec:
        raise NuisanceValidationError("hash-pinned power simulator is required")
    normalized_dgp = _validate_shortcut_power_dgp_v2(dgp)
    inventory = _sha256_text(inventory_sha256, "v2 power inventory digest")
    seed_value = _nonnegative_integer(seed, "v2 shortcut power seed")
    repetitions_value = _positive_integer(
        repetitions, "v2 power simulation repetitions"
    )
    bindings = {
        "algorithm": POWER_SIMULATION_ALGORITHM_V2,
        "configuration_sha256": _sha256_text(
            spec.configuration_sha256, "v2 power simulator configuration"
        ),
        "dgp_sha256": _canonical_digest(normalized_dgp),
        "implementation_sha256": _sha256_text(
            spec.executable_sha256, "v2 power simulator implementation"
        ),
        "inventory_sha256": inventory,
        "repetitions": repetitions_value,
        "seed": seed_value,
    }
    envelope = {
        "schema": POWER_SIMULATOR_REQUEST_SCHEMA_V2,
        "bindings": bindings,
        "dgp": normalized_dgp,
    }
    response = _run_pinned_json_tool_v2(
        executable_path=spec.executable_path,
        executable_sha256=spec.executable_sha256,
        configuration_path=spec.configuration_path,
        configuration_sha256=spec.configuration_sha256,
        envelope=envelope,
        label="v2 power simulator",
    )
    _exact_keys(
        response,
        {"schema", "bindings", "endpoint_statistics", "endpoint_draws"},
        "v2 power simulator response",
    )
    if response.get("schema") != POWER_SIMULATOR_RESPONSE_SCHEMA_V2:
        raise NuisanceValidationError("v2 power simulator response schema differs")
    if response.get("bindings") != bindings:
        raise NuisanceValidationError("v2 power simulator response binding differs")
    observed_statistics = _validate_shortcut_power_statistics_v2(
        response.get("endpoint_statistics"), repetitions=repetitions_value
    )
    observed_draws = _validate_shortcut_power_draws_v2(
        response.get("endpoint_draws"), repetitions=repetitions_value
    )
    expected = _expected_shortcut_power_simulation_v2(
        normalized_dgp, seed=seed_value, repetitions=repetitions_value
    )
    observed = {
        "endpoint_statistics": observed_statistics,
        "endpoint_draws": observed_draws,
    }
    if observed != expected:
        raise NuisanceValidationError(
            "v2 power simulator did not execute the frozen DGP and seed"
        )
    return observed


def _shortcut_power_summary_v2(
    draws: Mapping[str, Sequence[int]], *, repetitions: int
) -> tuple[dict[str, dict[str, Any]], float]:
    endpoints: dict[str, dict[str, Any]] = {}
    for endpoint in _V2_POWER_ENDPOINTS:
        detected = int(sum(draws[endpoint]))
        lower, _ = _wilson_interval(detected, repetitions, confidence_level=0.95)
        endpoints[endpoint] = {
            "detected_count": detected,
            "power_point_estimate": float(detected / repetitions),
            "power_lower_bound": float(lower),
        }
    minimum_lower = min(
        endpoint["power_lower_bound"] for endpoint in endpoints.values()
    )
    return endpoints, float(minimum_lower)


def build_shortcut_power_record_v2(
    *,
    inventory_sha256: str,
    component_count: int,
    remux_p_gold_epsilon: float,
    simulation_repetitions: int,
    endpoint_detection_counts: Mapping[str, int] | None = None,
    dgp: Mapping[str, Any] | None = None,
    simulator: SubprocessPowerSimulatorSpec | None = None,
    seed: int = 20_270_918,
) -> dict[str, Any]:
    """Execute a pinned simulator and retain every pre-run endpoint draw."""

    if endpoint_detection_counts is not None:
        raise NuisanceValidationError(
            "caller-supplied power detection counts are forbidden"
        )
    if dgp is None or type(simulator) is not SubprocessPowerSimulatorSpec:
        raise NuisanceValidationError(
            "a frozen DGP and hash-pinned power simulator are required"
        )
    repetitions = _positive_integer(
        simulation_repetitions, "v2 power simulation repetitions"
    )
    if repetitions != 10_000:
        raise NuisanceValidationError(
            "v2 shortcut power requires exactly 10000 simulations"
        )
    seed_value = _nonnegative_integer(seed, "v2 shortcut power seed")
    inventory = _sha256_text(inventory_sha256, "v2 power inventory digest")
    expected_components = _positive_integer(
        component_count, "v2 power inventory component count"
    )
    normalized_dgp = _validate_shortcut_power_dgp_v2(
        dgp, expected_remux_p_gold_epsilon=remux_p_gold_epsilon
    )
    if any(
        design["sample_size_components"] != expected_components
        for design in normalized_dgp["endpoint_designs"].values()
    ):
        raise NuisanceValidationError(
            "v2 shortcut power sample size differs from the inventory"
        )
    simulation_run = _run_shortcut_power_simulator_v2(
        simulator,
        inventory_sha256=inventory,
        dgp=normalized_dgp,
        seed=seed_value,
        repetitions=repetitions,
    )
    endpoint_statistics = simulation_run["endpoint_statistics"]
    draws = simulation_run["endpoint_draws"]
    endpoints, minimum_lower = _shortcut_power_summary_v2(
        draws, repetitions=repetitions
    )
    value = {
        "schema": SHORTCUT_POWER_SCHEMA_V2,
        "role": "pre_run",
        "status": ("adequately_powered" if minimum_lower >= 0.80 else "underpowered"),
        "evaluation_outcomes_access": "forbidden",
        "inventory_sha256": inventory,
        "inventory_component_count": expected_components,
        "dgp": normalized_dgp,
        "dgp_sha256": _canonical_digest(normalized_dgp),
        "simulator": {
            "implementation_sha256": _sha256_text(
                simulator.executable_sha256,
                "v2 power simulator implementation",
            ),
            "configuration_sha256": _sha256_text(
                simulator.configuration_sha256,
                "v2 power simulator configuration",
            ),
        },
        "simulation": {
            "repetitions": repetitions,
            "seed": seed_value,
            "algorithm": POWER_SIMULATION_ALGORITHM_V2,
            "target_power": 0.80,
            "interval_method": "two_sided_wilson_lower_bound_95",
            "minimum_power_lower_bound": minimum_lower,
            "endpoint_statistics": endpoint_statistics,
            "endpoint_statistics_sha256": _canonical_digest(endpoint_statistics),
            "endpoint_draws": draws,
            "endpoint_draws_sha256": _canonical_digest(draws),
            "endpoints": endpoints,
        },
    }
    return _attest_v2(value)


def validate_shortcut_power_record_v2(
    value: Any,
    *,
    expected_inventory_sha256: str,
    expected_component_count: int,
    expected_remux_p_gold_epsilon: float,
    simulator: SubprocessPowerSimulatorSpec | None = None,
) -> dict[str, Any]:
    """Replay all 10,000 frozen-DGP endpoint draws before accepting power."""

    if type(simulator) is not SubprocessPowerSimulatorSpec:
        raise NuisanceValidationError(
            "an authenticated hash-pinned power simulator is required"
        )
    record = _validate_attested_v2(value, "v2 shortcut power record")
    _exact_keys(
        record,
        {
            "schema",
            "role",
            "status",
            "evaluation_outcomes_access",
            "inventory_sha256",
            "inventory_component_count",
            "dgp",
            "dgp_sha256",
            "simulator",
            "simulation",
            "attestation_sha256",
        },
        "v2 shortcut power record",
    )
    if record.get("schema") != SHORTCUT_POWER_SCHEMA_V2:
        raise NuisanceValidationError("v2 shortcut power schema differs")
    if record.get("role") != "pre_run":
        raise NuisanceValidationError("v2 shortcut power role differs")
    if record.get("evaluation_outcomes_access") != "forbidden":
        raise NuisanceValidationError("v2 shortcut power used evaluation outcomes")
    inventory = _sha256_text(
        expected_inventory_sha256, "expected v2 power inventory digest"
    )
    if record.get("inventory_sha256") != inventory:
        raise NuisanceValidationError("v2 shortcut power inventory differs")
    component_count = _positive_integer(
        expected_component_count, "expected v2 power component count"
    )
    if record.get("inventory_component_count") != component_count:
        raise NuisanceValidationError("v2 shortcut power component count differs")
    dgp = _validate_shortcut_power_dgp_v2(
        record.get("dgp"),
        expected_remux_p_gold_epsilon=expected_remux_p_gold_epsilon,
    )
    if any(
        design["sample_size_components"] != component_count
        for design in dgp["endpoint_designs"].values()
    ):
        raise NuisanceValidationError(
            "v2 shortcut power sample size differs from the inventory"
        )
    if record.get("dgp_sha256") != _canonical_digest(dgp):
        raise NuisanceValidationError("v2 shortcut power DGP digest differs")
    simulator_record = _mapping(record.get("simulator"), "v2 shortcut power simulator")
    expected_simulator = {
        "implementation_sha256": _sha256_text(
            simulator.executable_sha256, "v2 power simulator implementation"
        ),
        "configuration_sha256": _sha256_text(
            simulator.configuration_sha256, "v2 power simulator configuration"
        ),
    }
    if dict(simulator_record) != expected_simulator:
        raise NuisanceValidationError("v2 shortcut power simulator binding differs")
    simulation = _mapping(record.get("simulation"), "v2 shortcut power simulation")
    _exact_keys(
        simulation,
        {
            "repetitions",
            "seed",
            "algorithm",
            "target_power",
            "interval_method",
            "minimum_power_lower_bound",
            "endpoint_statistics",
            "endpoint_statistics_sha256",
            "endpoint_draws",
            "endpoint_draws_sha256",
            "endpoints",
        },
        "v2 shortcut power simulation",
    )
    if simulation.get("repetitions") != 10_000:
        raise NuisanceValidationError(
            "v2 shortcut power requires exactly 10000 simulations"
        )
    seed = _nonnegative_integer(simulation.get("seed"), "v2 shortcut power seed")
    if simulation.get("algorithm") != POWER_SIMULATION_ALGORITHM_V2:
        raise NuisanceValidationError("v2 shortcut power algorithm differs")
    if _exact_float(simulation.get("target_power"), "v2 shortcut target power") != 0.80:
        raise NuisanceValidationError("v2 shortcut target power differs")
    if simulation.get("interval_method") != "two_sided_wilson_lower_bound_95":
        raise NuisanceValidationError("v2 shortcut power interval method differs")
    endpoint_statistics = _validate_shortcut_power_statistics_v2(
        simulation.get("endpoint_statistics"), repetitions=10_000
    )
    if simulation.get("endpoint_statistics_sha256") != _canonical_digest(
        endpoint_statistics
    ):
        raise NuisanceValidationError("v2 shortcut power statistic digest differs")
    draws = _validate_shortcut_power_draws_v2(
        simulation.get("endpoint_draws"), repetitions=10_000
    )
    if simulation.get("endpoint_draws_sha256") != _canonical_digest(draws):
        raise NuisanceValidationError("v2 shortcut power draw digest differs")
    expected_run = _expected_shortcut_power_simulation_v2(
        dgp, seed=seed, repetitions=10_000
    )
    observed_run = {
        "endpoint_statistics": endpoint_statistics,
        "endpoint_draws": draws,
    }
    if observed_run != expected_run:
        raise NuisanceValidationError(
            "v2 shortcut power draws differ from the DGP statistics"
        )
    replayed_run = _run_shortcut_power_simulator_v2(
        simulator,
        inventory_sha256=inventory,
        dgp=dgp,
        seed=seed,
        repetitions=10_000,
    )
    if replayed_run != observed_run:
        raise NuisanceValidationError("v2 shortcut power replay differs")
    endpoints, minimum_lower = _shortcut_power_summary_v2(draws, repetitions=10_000)
    expected_simulation = {
        "repetitions": 10_000,
        "seed": seed,
        "algorithm": POWER_SIMULATION_ALGORITHM_V2,
        "target_power": 0.80,
        "interval_method": "two_sided_wilson_lower_bound_95",
        "minimum_power_lower_bound": minimum_lower,
        "endpoint_statistics": endpoint_statistics,
        "endpoint_statistics_sha256": _canonical_digest(endpoint_statistics),
        "endpoint_draws": draws,
        "endpoint_draws_sha256": _canonical_digest(draws),
        "endpoints": endpoints,
    }
    if dict(simulation) != expected_simulation:
        raise NuisanceValidationError("v2 shortcut power summary differs")
    if minimum_lower < 0.80 or record.get("status") != "adequately_powered":
        raise NuisanceValidationError("v2 shortcut power is inadequate")
    return copy.deepcopy(dict(record))


def _validate_dependency_subject_projection_v2(
    value: Any, *, role: str, subject_sha256: str
) -> dict[str, Any]:
    projection = _mapping(value, "v2 dependency subject projection")
    _exact_keys(
        projection,
        {"schema", "role", "subject_sha256", "projection"},
        "v2 dependency subject projection",
    )
    if projection.get("schema") != DEPENDENCY_SUBJECT_PROJECTION_SCHEMA_V2:
        raise NuisanceValidationError("v2 dependency projection schema differs")
    if projection.get("role") != role:
        raise NuisanceValidationError("v2 dependency projection role differs")
    if projection.get("subject_sha256") != subject_sha256:
        raise NuisanceValidationError("v2 dependency projection subject differs")
    _mapping(projection.get("projection"), "v2 dependency projected fields")
    return copy.deepcopy(dict(projection))


def _validate_dependency_transcript_v2(value: Any, *, role: str) -> dict[str, Any]:
    transcript = _mapping(value, "v2 dependency transcript")
    _exact_keys(
        transcript,
        {"schema", "role", "events"},
        "v2 dependency transcript",
    )
    if transcript.get("schema") != DEPENDENCY_TRANSCRIPT_SCHEMA_V2:
        raise NuisanceValidationError("v2 dependency transcript schema differs")
    if transcript.get("role") != role:
        raise NuisanceValidationError("v2 dependency transcript role differs")
    events = transcript.get("events")
    if not isinstance(events, list) or not events:
        raise NuisanceValidationError("v2 dependency transcript events are empty")
    for index, event in enumerate(events):
        _mapping(event, f"v2 dependency transcript event {index}")
    return copy.deepcopy(dict(transcript))


def _run_dependency_verifier_v2(
    spec: SubprocessDependencyVerifierSpec,
    *,
    role: str,
    subject_sha256: str,
    subject_projection: Mapping[str, Any],
    transcript: Mapping[str, Any],
) -> dict[str, Any]:
    if type(spec) is not SubprocessDependencyVerifierSpec:
        raise NuisanceValidationError("hash-pinned dependency verifier is required")
    normalized_role = _strict_identifier(role, "v2 dependency role")
    normalized_subject = _sha256_text(subject_sha256, "v2 dependency subject")
    projection = _validate_dependency_subject_projection_v2(
        subject_projection, role=normalized_role, subject_sha256=normalized_subject
    )
    transcript_value = _validate_dependency_transcript_v2(
        transcript, role=normalized_role
    )
    bindings = {
        "configuration_sha256": _sha256_text(
            spec.configuration_sha256, "v2 dependency verifier configuration"
        ),
        "implementation_sha256": _sha256_text(
            spec.executable_sha256, "v2 dependency verifier implementation"
        ),
        "subject_projection_sha256": _canonical_digest(projection),
        "transcript_sha256": _canonical_digest(transcript_value),
    }
    response = _run_pinned_json_tool_v2(
        executable_path=spec.executable_path,
        executable_sha256=spec.executable_sha256,
        configuration_path=spec.configuration_path,
        configuration_sha256=spec.configuration_sha256,
        envelope={
            "schema": DEPENDENCY_VERIFIER_REQUEST_SCHEMA_V2,
            "bindings": bindings,
            "role": normalized_role,
            "subject_projection": projection,
            "transcript": transcript_value,
        },
        label=f"v2 {normalized_role} dependency verifier",
    )
    _exact_keys(
        response,
        {"schema", "bindings", "result"},
        "v2 dependency verifier response",
    )
    if response.get("schema") != DEPENDENCY_VERIFIER_RESPONSE_SCHEMA_V2:
        raise NuisanceValidationError("v2 dependency verifier response schema differs")
    if response.get("bindings") != bindings:
        raise NuisanceValidationError("v2 dependency verifier response binding differs")
    result = _mapping(response.get("result"), "v2 dependency verifier result")
    _exact_keys(
        result,
        {"decision_status", "reconstructed_subject_sha256"},
        "v2 dependency verifier result",
    )
    decision = _nonempty_text(
        result.get("decision_status"), "v2 dependency verifier decision"
    )
    reconstructed = _sha256_text(
        result.get("reconstructed_subject_sha256"),
        "v2 reconstructed dependency subject",
    )
    if reconstructed != normalized_subject:
        raise NuisanceValidationError(
            "v2 dependency verifier reconstructed subject differs"
        )
    return {
        "bindings": bindings,
        "decision_status": decision,
        "reconstructed_subject_sha256": reconstructed,
        "response_sha256": _canonical_digest(response),
    }


def build_verified_dependency_record_v2(
    *,
    role: str,
    subject_sha256: str,
    decision_status: str | None = None,
    verifier_sha256: str | None = None,
    transcript_sha256: str | None = None,
    subject_projection: Mapping[str, Any] | None = None,
    transcript: Mapping[str, Any] | None = None,
    verifier: SubprocessDependencyVerifierSpec | None = None,
) -> dict[str, Any]:
    """Execute a pinned verifier over authenticated dependency evidence."""

    if any(
        legacy is not None
        for legacy in (decision_status, verifier_sha256, transcript_sha256)
    ):
        raise NuisanceValidationError(
            "hash-only dependency self-attestation is forbidden; authenticated "
            "subject projections and transcripts must be verified"
        )
    if (
        subject_projection is None
        or transcript is None
        or type(verifier) is not SubprocessDependencyVerifierSpec
    ):
        raise NuisanceValidationError(
            "authenticated subject projection, transcript, and verifier are required"
        )
    normalized_role = _strict_identifier(role, "v2 dependency role")
    normalized_subject = _sha256_text(subject_sha256, "v2 dependency subject")
    verification = _run_dependency_verifier_v2(
        verifier,
        role=normalized_role,
        subject_sha256=normalized_subject,
        subject_projection=subject_projection,
        transcript=transcript,
    )
    value = {
        "schema": VERIFIED_DEPENDENCY_SCHEMA_V2,
        "role": normalized_role,
        "status": "replay_verified",
        "decision_status": verification["decision_status"],
        "subject_sha256": normalized_subject,
        "subject_projection_sha256": verification["bindings"][
            "subject_projection_sha256"
        ],
        "transcript_sha256": verification["bindings"]["transcript_sha256"],
        "verifier": {
            "implementation_sha256": verification["bindings"]["implementation_sha256"],
            "configuration_sha256": verification["bindings"]["configuration_sha256"],
        },
        "replay": {
            "status": "verified",
            "reconstructed_subject_sha256": verification[
                "reconstructed_subject_sha256"
            ],
            "response_sha256": verification["response_sha256"],
        },
    }
    return _attest_v2(value)


def validate_verified_dependency_record_v2(
    value: Any,
    *,
    expected_role: str,
    expected_subject_sha256: str,
    subject_projection: Mapping[str, Any] | None = None,
    transcript: Mapping[str, Any] | None = None,
    verifier: SubprocessDependencyVerifierSpec | None = None,
) -> dict[str, Any]:
    """Re-execute the pinned verifier over authenticated upstream evidence."""

    if (
        subject_projection is None
        or transcript is None
        or type(verifier) is not SubprocessDependencyVerifierSpec
    ):
        raise NuisanceValidationError(
            "authenticated subject projection, transcript, and verifier are required"
        )
    record = _validate_attested_v2(value, "v2 verified dependency")
    _exact_keys(
        record,
        {
            "schema",
            "role",
            "status",
            "decision_status",
            "subject_sha256",
            "subject_projection_sha256",
            "transcript_sha256",
            "verifier",
            "replay",
            "attestation_sha256",
        },
        "v2 verified dependency",
    )
    expected_role_value = _strict_identifier(expected_role, "expected role")
    expected_subject = _sha256_text(
        expected_subject_sha256, "expected v2 dependency subject"
    )
    if record.get("schema") != VERIFIED_DEPENDENCY_SCHEMA_V2:
        raise NuisanceValidationError("v2 dependency schema differs")
    if record.get("role") != expected_role_value:
        raise NuisanceValidationError("v2 dependency role differs")
    if record.get("subject_sha256") != expected_subject:
        raise NuisanceValidationError("v2 dependency subject differs")
    if record.get("status") != "replay_verified":
        raise NuisanceValidationError("v2 dependency was not replay verified")
    verification = _run_dependency_verifier_v2(
        verifier,
        role=expected_role_value,
        subject_sha256=expected_subject,
        subject_projection=subject_projection,
        transcript=transcript,
    )
    if (
        record.get("subject_projection_sha256")
        != verification["bindings"]["subject_projection_sha256"]
    ):
        raise NuisanceValidationError("v2 dependency subject projection differs")
    if record.get("transcript_sha256") != verification["bindings"]["transcript_sha256"]:
        raise NuisanceValidationError("v2 dependency transcript differs")
    if record.get("verifier") != {
        "implementation_sha256": verification["bindings"]["implementation_sha256"],
        "configuration_sha256": verification["bindings"]["configuration_sha256"],
    }:
        raise NuisanceValidationError("v2 dependency verifier binding differs")
    expected_replay = {
        "status": "verified",
        "reconstructed_subject_sha256": verification["reconstructed_subject_sha256"],
        "response_sha256": verification["response_sha256"],
    }
    if record.get("replay") != expected_replay:
        raise NuisanceValidationError("v2 dependency replay differs")
    if record.get("decision_status") != verification["decision_status"]:
        raise NuisanceValidationError("v2 dependency decision differs")
    if record.get("decision_status") not in {"pass", "complete"}:
        raise NuisanceValidationError("v2 dependency did not pass")
    return copy.deepcopy(dict(record))


def validate_v2_dependency_chain(
    preregistration_value: Any,
    *,
    power_record: Any,
    power_record_sha256: str,
    power_simulator: SubprocessPowerSimulatorSpec | None = None,
    dependency_records: Mapping[str, Any],
    dependency_record_sha256s: Mapping[str, str],
    dependency_subject_projections: Mapping[str, Mapping[str, Any]] | None = None,
    dependency_transcripts: Mapping[str, Mapping[str, Any]] | None = None,
    dependency_verifiers: Mapping[str, SubprocessDependencyVerifierSpec] | None = None,
) -> dict[str, Any]:
    """Validate every replay receipt and the pre-run power record bound above."""

    preregistration = _validate_attested_v2(
        preregistration_value, "v2 shortcut preregistration"
    )
    if preregistration.get("schema") != SHORTCUT_PREREGISTRATION_SCHEMA_V2:
        raise NuisanceValidationError("v2 shortcut preregistration schema differs")
    digests = _mapping(
        preregistration.get("input_digests"), "v2 shortcut input digests"
    )
    records = _mapping(dependency_records, "v2 dependency records")
    record_sha256s = _mapping(dependency_record_sha256s, "v2 dependency record digests")
    if (
        dependency_subject_projections is None
        or dependency_transcripts is None
        or dependency_verifiers is None
    ):
        raise NuisanceValidationError(
            "authenticated dependency projections, transcripts, and verifiers are required"
        )
    projections = _mapping(
        dependency_subject_projections, "v2 dependency subject projections"
    )
    transcripts = _mapping(dependency_transcripts, "v2 dependency transcripts")
    verifiers = _mapping(dependency_verifiers, "v2 dependency verifiers")
    expected_roles = set(_V2_DEPENDENCY_ROLES)
    if any(
        set(items) != expected_roles
        for items in (records, record_sha256s, projections, transcripts, verifiers)
    ):
        raise NuisanceValidationError("v2 dependency role set differs")
    validated_dependencies: dict[str, Any] = {}
    for role, (subject_key, verification_key) in _V2_DEPENDENCY_ROLES.items():
        observed_record_sha256 = _sha256_text(
            record_sha256s[role], f"v2 {role} verification-record digest"
        )
        if observed_record_sha256 != digests.get(verification_key):
            raise NuisanceValidationError(
                f"v2 {role} verification-record binding differs"
            )
        if _canonical_digest(records[role]) != observed_record_sha256:
            raise NuisanceValidationError(
                f"v2 {role} verification-record content digest differs"
            )
        validated_dependencies[role] = validate_verified_dependency_record_v2(
            records[role],
            expected_role=role,
            expected_subject_sha256=digests.get(subject_key),
            subject_projection=projections[role],
            transcript=transcripts[role],
            verifier=verifiers[role],
        )
    observed_power_sha256 = _sha256_text(power_record_sha256, "v2 power-record digest")
    if observed_power_sha256 != digests.get("power_output_sha256"):
        raise NuisanceValidationError("v2 power-record binding differs")
    if _canonical_digest(power_record) != observed_power_sha256:
        raise NuisanceValidationError("v2 power-record content digest differs")
    validated_power = validate_shortcut_power_record_v2(
        power_record,
        expected_inventory_sha256=preregistration["inventory"]["inventory_sha256"],
        expected_component_count=preregistration["inventory"]["component_count"],
        expected_remux_p_gold_epsilon=preregistration["control_plan"][
            "remux_noninferiority"
        ]["epsilon_p_gold"],
        simulator=power_simulator,
    )
    return {
        "dependencies": validated_dependencies,
        "dependency_record_sha256s": dict(record_sha256s),
        "power_record": validated_power,
        "power_record_sha256": observed_power_sha256,
    }


def build_nuisance_contract(
    *,
    configuration_sha256: str,
    pilot_index_sha256: str,
    media_receipt_sha256: str,
    media_set_sha256: str,
    implementation_bundle_sha256: str,
    shortcut_preregistration_sha256: str | None = None,
) -> dict[str, Any]:
    """Create the exact immutable contract accepted by the nuisance gate."""

    input_digests = {
        "configuration_sha256": _sha256_text(
            configuration_sha256, "configuration digest"
        ),
        "pilot_index_sha256": _sha256_text(pilot_index_sha256, "pilot-index digest"),
        "media_receipt_sha256": _sha256_text(
            media_receipt_sha256, "media-receipt digest"
        ),
        "media_set_sha256": _sha256_text(media_set_sha256, "media-set digest"),
        "implementation_bundle_sha256": _sha256_text(
            implementation_bundle_sha256, "implementation-bundle digest"
        ),
    }
    schema = "conflictbench.perception-nuisance-detection-contract.v1"
    if shortcut_preregistration_sha256 is not None:
        schema = "conflictbench.perception-nuisance-detection-contract.v2"
        input_digests["shortcut_preregistration_sha256"] = _sha256_text(
            shortcut_preregistration_sha256, "shortcut-preregistration digest"
        )
    return {
        "schema": schema,
        "status": "preregistered_not_run",
        "input_digests": input_digests,
        "partitions": {
            "fit": "scorer_fit",
            "threshold": "threshold_calibration",
            "evaluation": "pilot_gate",
        },
        "human_evaluation": "forbidden",
        "outcome_based_sample_selection": "forbidden",
    }


def validate_nuisance_contract(
    value: Any,
    *,
    expected_pilot_index_sha256: str,
    expected_media_receipt_sha256: str,
    expected_media_set_sha256: str,
    expected_configuration_sha256: str,
    expected_implementation_bundle_sha256: str,
    expected_shortcut_preregistration_sha256: str | None = None,
) -> None:
    """Validate the minimal preregistration record consumed by the full gate."""

    contract = _mapping(value, "nuisance contract")
    _exact_keys(
        contract,
        {
            "schema",
            "status",
            "input_digests",
            "partitions",
            "human_evaluation",
            "outcome_based_sample_selection",
        },
        "nuisance contract",
    )
    expected_schema = (
        "conflictbench.perception-nuisance-detection-contract.v1"
        if expected_shortcut_preregistration_sha256 is None
        else "conflictbench.perception-nuisance-detection-contract.v2"
    )
    if contract.get("schema") != expected_schema:
        raise NuisanceValidationError("nuisance contract schema differs")
    if contract.get("status") != "preregistered_not_run":
        raise NuisanceValidationError(
            "nuisance contract is not in its preregistered state"
        )
    expected_input_digests = {
        "configuration_sha256": _sha256_text(
            expected_configuration_sha256, "expected configuration digest"
        ),
        "pilot_index_sha256": _sha256_text(
            expected_pilot_index_sha256, "expected pilot-index digest"
        ),
        "media_receipt_sha256": _sha256_text(
            expected_media_receipt_sha256, "expected media-receipt digest"
        ),
        "media_set_sha256": _sha256_text(
            expected_media_set_sha256, "expected media-set digest"
        ),
        "implementation_bundle_sha256": _sha256_text(
            expected_implementation_bundle_sha256,
            "expected implementation-bundle digest",
        ),
    }
    if expected_shortcut_preregistration_sha256 is not None:
        expected_input_digests["shortcut_preregistration_sha256"] = _sha256_text(
            expected_shortcut_preregistration_sha256,
            "expected shortcut-preregistration digest",
        )
    if contract.get("input_digests") != expected_input_digests:
        raise NuisanceValidationError("nuisance contract input binding differs")
    if contract.get("partitions") != {
        "fit": "scorer_fit",
        "threshold": "threshold_calibration",
        "evaluation": "pilot_gate",
    }:
        raise NuisanceValidationError("nuisance contract partition isolation differs")
    if contract.get("human_evaluation") != "forbidden":
        raise NuisanceValidationError("nuisance contract permits human evaluation")
    if contract.get("outcome_based_sample_selection") != "forbidden":
        raise NuisanceValidationError(
            "nuisance contract permits outcome-based sample selection"
        )


def _validate_complete_report(
    value: Any,
    *,
    expected_input_digests: Mapping[str, str],
    expected_configuration: Mapping[str, Any],
) -> Mapping[str, Any]:
    report = _mapping(value, "nuisance report")
    required = {
        "schema",
        "human_evaluation_used",
        "outcome_based_sample_selection",
        "feature_contract",
        "partition_integrity",
        "multiplicity_control",
        "frozen_detector",
        "frozen_detector_suite",
        "evaluation",
        "question_key_sensitivity",
        "decision",
        "configuration",
        "input_digests",
        "runtime",
        "counts",
        "records",
        "attestation_sha256",
    }
    _exact_keys(report, required, "complete nuisance report")
    _validate_report_attestation(report)
    if report.get("human_evaluation_used") is not False:
        raise NuisanceValidationError("nuisance report used human evaluation")
    if report.get("outcome_based_sample_selection") != "forbidden":
        raise NuisanceValidationError("nuisance report used outcome-based selection")
    config = _validate_config(report.get("configuration"))
    locked_config = _validate_config(expected_configuration)
    if config != locked_config:
        raise NuisanceValidationError(
            "nuisance report configuration differs from the locked configuration"
        )
    if report.get("input_digests") != dict(expected_input_digests):
        raise NuisanceValidationError("nuisance report input binding differs")

    runtime = _mapping(report.get("runtime"), "nuisance report runtime")
    _exact_keys(
        runtime,
        {"numpy_version", "ffprobe", "implementation_source_sha256"},
        "nuisance report runtime",
    )
    _nonempty_text(runtime.get("numpy_version"), "NumPy version")
    source_digests = _mapping(
        runtime.get("implementation_source_sha256"), "implementation source digests"
    )
    if set(source_digests) != set(IMPLEMENTATION_SOURCE_ROLES):
        raise NuisanceValidationError("implementation source set differs")
    for name, digest in source_digests.items():
        _nonempty_text(name, "implementation source name")
        _sha256_text(digest, f"implementation source digest for {name}")
    if (
        _canonical_digest(dict(sorted(source_digests.items())))
        != expected_input_digests["implementation_bundle_sha256"]
    ):
        raise NuisanceValidationError("implementation-bundle digest differs")

    records = _validate_records(report.get("records"))
    counts = _mapping(report.get("counts"), "nuisance report counts")
    _exact_keys(
        counts,
        {"media_file_count", "feature_record_count", "component_count"},
        "nuisance report counts",
    )
    if counts.get("feature_record_count") != len(records):
        raise NuisanceValidationError("nuisance report feature-record count differs")
    if counts.get("component_count") != len(
        {record["component_id"] for record in records}
    ):
        raise NuisanceValidationError("nuisance report component count differs")
    _positive_integer(counts.get("media_file_count"), "nuisance report media count")

    suite = _mapping(report.get("frozen_detector_suite"), "frozen detector suite")
    _exact_keys(
        suite,
        {"families", "selection", "models", "detector_suite_sha256"},
        "frozen detector suite",
    )
    suite_unsigned = dict(suite)
    suite_digest = _sha256_text(
        suite_unsigned.pop("detector_suite_sha256"), "detector-suite digest"
    )
    if _canonical_digest(suite_unsigned) != suite_digest:
        raise NuisanceValidationError("detector-suite digest differs")
    if suite.get("families") != config["model"]["families"]:
        raise NuisanceValidationError("detector-suite families differ")
    models = _mapping(suite.get("models"), "frozen detector models")
    if list(models) != config["model"]["families"]:
        raise NuisanceValidationError("frozen detector model order differs")
    for family, model_value in models.items():
        frozen = _mapping(model_value, f"frozen detector {family}")
        unsigned = dict(frozen)
        threshold_digest = _sha256_text(
            unsigned.pop("threshold_record_sha256"),
            f"threshold-record digest for {family}",
        )
        if _canonical_digest(unsigned) != threshold_digest:
            raise NuisanceValidationError(
                f"threshold-record digest differs for {family}"
            )
    if report.get("frozen_detector") != models["ridge_linear"]:
        raise NuisanceValidationError("legacy frozen-detector view differs")

    expected_multiplicity = {
        "familywise_confidence_level": float(config["uncertainty"]["confidence_level"]),
        "comparison_count": len(config["model"]["families"]) * 2,
        "per_interval_confidence_level": float(
            config["uncertainty"]["confidence_level"]
        ),
        "interval_method": "per_detector_then_maximum_upper_bound",
        "permutation_method": "single_max_statistic",
    }
    if report.get("multiplicity_control") != expected_multiplicity:
        raise NuisanceValidationError("multiplicity-control record differs")

    evaluation = _mapping(report.get("evaluation"), "nuisance report evaluation")
    _exact_keys(
        evaluation,
        {"detector", "detector_suite", "structural_chance", "permutation"},
        "nuisance report evaluation",
    )
    detector = _mapping(evaluation.get("detector"), "detector aggregate")
    evaluated_suite = _mapping(
        evaluation.get("detector_suite"), "evaluated detector suite"
    )
    evaluated_models = _mapping(
        evaluated_suite.get("models"), "evaluated detector models"
    )
    if list(evaluated_models) != config["model"]["families"]:
        raise NuisanceValidationError("evaluated detector families differ")
    recomputed_aggregate = _aggregate_detector_summaries(evaluated_models)
    for key in (
        "selected_model_for_point_estimate",
        "selected_model_for_upper_bound",
        "role_recovery_point_estimate",
        "conservative_primary_upper_confidence_bound",
    ):
        if detector.get(key) != recomputed_aggregate.get(key):
            raise NuisanceValidationError("detector aggregate differs")
    if evaluated_suite != {"models": dict(evaluated_models), **dict(detector)}:
        raise NuisanceValidationError("evaluated detector-suite summary differs")
    chance = _mapping(
        evaluation.get("structural_chance"), "structural chance reference"
    )
    if chance.get("method") != "paired_role_structure_fixed_probability":
        raise NuisanceValidationError("structural chance method differs")
    if float(chance.get("fixed_probability", -1.0)) != 0.5:
        raise NuisanceValidationError("structural chance probability differs")
    permutation = _mapping(evaluation.get("permutation"), "permutation summary")
    if (
        permutation.get("multiplicity_control")
        != "single_max_statistic_permutation_family"
    ):
        raise NuisanceValidationError("permutation multiplicity control differs")
    question_sensitivity = _mapping(
        report.get("question_key_sensitivity"), "question-key sensitivity"
    )
    expected_reasons = _decision_reason_codes(
        detector=detector,
        structural_chance=chance,
        question_sensitivity=question_sensitivity,
        permutation=permutation,
        gate=config["gate"],
    )
    expected_decision = {
        "nuisance_detection_status": "pass" if not expected_reasons else "fail",
        "reason_codes": expected_reasons,
    }
    if report.get("decision") != expected_decision:
        raise NuisanceValidationError(
            "nuisance report decision differs on recomputation"
        )
    return report


def build_handoff_result(
    *,
    report: Mapping[str, Any],
    configuration: Mapping[str, Any],
    report_sha256: str,
    pilot_index_sha256: str,
    media_receipt_sha256: str,
    media_set_sha256: str,
    nuisance_contract_sha256: str,
    configuration_sha256: str,
    implementation_bundle_sha256: str,
) -> dict[str, Any]:
    """Build a result only from a fully authenticated detailed report."""

    expected = {
        "configuration_sha256": _sha256_text(
            configuration_sha256, "configuration digest"
        ),
        "pilot_index_sha256": _sha256_text(pilot_index_sha256, "pilot-index digest"),
        "media_receipt_sha256": _sha256_text(
            media_receipt_sha256, "media-receipt digest"
        ),
        "media_set_sha256": _sha256_text(media_set_sha256, "media-set digest"),
        "nuisance_contract_sha256": _sha256_text(
            nuisance_contract_sha256, "nuisance-contract digest"
        ),
        "implementation_bundle_sha256": _sha256_text(
            implementation_bundle_sha256, "implementation-bundle digest"
        ),
    }
    report_value = _validate_complete_report(
        report,
        expected_input_digests=expected,
        expected_configuration=configuration,
    )
    observed_report_sha256 = hashlib.sha256(_json_bytes(report_value)).hexdigest()
    if observed_report_sha256 != _sha256_text(report_sha256, "report digest"):
        raise NuisanceValidationError("nuisance report byte digest differs")
    decision = _mapping(report_value["decision"], "nuisance report decision")
    suite = _mapping(report_value["frozen_detector_suite"], "frozen detector suite")
    result = {
        "schema": "conflictbench.perception-nuisance-detection-result.v1",
        "status": "complete",
        "input_digests": expected,
        "human_evaluation_used": False,
        "outcome_based_sample_selection": "forbidden",
        "decision": {
            "nuisance_detection_status": decision["nuisance_detection_status"],
            "reason_codes": list(decision["reason_codes"]),
            "report_sha256": observed_report_sha256,
            "detector_suite_sha256": _sha256_text(
                suite.get("detector_suite_sha256"), "detector-suite digest"
            ),
            "scope": "metadata_level_edit_role_detection",
            "limitation": (
                "A pass means only that the frozen container and stream diagnostics "
                "did not recover edit role under this pilot; it does not establish "
                "perceptual indistinguishability."
            ),
        },
    }
    result["attestation_sha256"] = _canonical_digest(result)
    return result


def validate_handoff_result(
    value: Any,
    *,
    report: Mapping[str, Any],
    expected_configuration: Mapping[str, Any],
    expected_pilot_index_sha256: str,
    expected_media_receipt_sha256: str,
    expected_media_set_sha256: str,
    expected_nuisance_contract_sha256: str,
    expected_configuration_sha256: str,
    expected_implementation_bundle_sha256: str,
    expected_report_sha256: str,
) -> None:
    """Reject a changed or incompletely bound handoff result."""

    result = _mapping(value, "nuisance result")
    _exact_keys(
        result,
        {
            "schema",
            "status",
            "input_digests",
            "human_evaluation_used",
            "outcome_based_sample_selection",
            "decision",
            "attestation_sha256",
        },
        "nuisance result",
    )
    attestation = _sha256_text(
        result.get("attestation_sha256"), "nuisance-result attestation"
    )
    unsigned = dict(result)
    unsigned.pop("attestation_sha256")
    if _canonical_digest(unsigned) != attestation:
        raise NuisanceValidationError("nuisance-result attestation differs")
    if result.get("schema") != "conflictbench.perception-nuisance-detection-result.v1":
        raise NuisanceValidationError("nuisance-result schema differs")
    if result.get("status") != "complete":
        raise NuisanceValidationError("nuisance result is incomplete")
    rebuilt = build_handoff_result(
        report=report,
        configuration=expected_configuration,
        report_sha256=expected_report_sha256,
        pilot_index_sha256=expected_pilot_index_sha256,
        media_receipt_sha256=expected_media_receipt_sha256,
        media_set_sha256=expected_media_set_sha256,
        nuisance_contract_sha256=expected_nuisance_contract_sha256,
        configuration_sha256=expected_configuration_sha256,
        implementation_bundle_sha256=expected_implementation_bundle_sha256,
    )
    if dict(result) != rebuilt:
        raise NuisanceValidationError(
            "nuisance result differs from the recomputed detailed-report decision"
        )


def _sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise NuisanceValidationError(
            f"required file cannot be read: {path.name}"
        ) from exc
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise NuisanceValidationError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise NuisanceValidationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_locked_json(
    path: pathlib.Path, expected_sha256: str, label: str
) -> tuple[bytes, Mapping[str, Any]]:
    source = pathlib.Path(path)
    if source.is_symlink() or not source.is_file():
        raise NuisanceValidationError(f"{label} must be a regular non-symlink file")
    expected = _sha256_text(expected_sha256, f"expected {label} digest")
    try:
        payload = source.read_bytes()
    except OSError as exc:
        raise NuisanceValidationError(f"{label} cannot be read") from exc
    if hashlib.sha256(payload).hexdigest() != expected:
        raise NuisanceValidationError(f"{label} digest differs")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NuisanceValidationError(f"{label} is not strict UTF-8 JSON") from exc
    return payload, _mapping(value, label)


def _authenticate_locked_file(
    path: pathlib.Path, expected_sha256: str, label: str
) -> str:
    source = pathlib.Path(path)
    if source.is_symlink() or not source.is_file():
        raise NuisanceValidationError(f"{label} must be a regular non-symlink file")
    expected = _sha256_text(expected_sha256, f"expected {label} digest")
    if _sha256_path(source) != expected:
        raise NuisanceValidationError(f"{label} digest differs")
    return expected


def validate_media_receipt(
    value: Any,
    *,
    media_root: pathlib.Path,
    expected_pilot_index_sha256: str,
    expected_media_set_sha256: str,
    expected_video_ids: set[str],
) -> list[dict[str, Any]]:
    """Reauthenticate the extracted media bytes and their exact directory set."""

    receipt = _mapping(value, "media receipt")
    _exact_keys(
        receipt,
        {
            "schema",
            "status",
            "input_digests",
            "media_root_name",
            "media_file_count",
            "media_files",
            "stream_probe",
        },
        "media receipt",
    )
    if receipt.get("schema") != "conflictbench.perception-pilot-media-receipt.v1":
        raise NuisanceValidationError("media receipt schema differs")
    if receipt.get("status") != "selected_media_extracted":
        raise NuisanceValidationError("media receipt is incomplete")
    input_digests = _mapping(
        receipt.get("input_digests"), "media receipt input digests"
    )
    _exact_keys(
        input_digests,
        {"train_archive_sha256", "pilot_index_sha256"},
        "media receipt input digests",
    )
    _sha256_text(input_digests.get("train_archive_sha256"), "train archive digest")
    if input_digests.get("pilot_index_sha256") != _sha256_text(
        expected_pilot_index_sha256, "expected pilot-index digest"
    ):
        raise NuisanceValidationError("media receipt pilot-index digest differs")

    root_input = pathlib.Path(media_root)
    if root_input.is_symlink() or not root_input.is_dir():
        raise NuisanceValidationError("media root must be a non-symlink directory")
    root = root_input.resolve(strict=True)
    if receipt.get("media_root_name") != root.name:
        raise NuisanceValidationError("media receipt root name differs")
    files = receipt.get("media_files")
    if not isinstance(files, list) or not files:
        raise NuisanceValidationError("media receipt file list is empty")
    if receipt.get("media_file_count") != len(files):
        raise NuisanceValidationError("media receipt file count differs")

    normalized = []
    names: set[str] = set()
    observed_video_ids: set[str] = set()
    for index, raw in enumerate(files):
        item = _mapping(raw, f"media receipt file {index}")
        _exact_keys(
            item,
            {
                "filename",
                "size_bytes",
                "sha256",
                "zip_crc32",
                "resumed_existing",
                "stream_types",
            },
            f"media receipt file {index}",
        )
        filename = _nonempty_text(item.get("filename"), "media filename")
        if pathlib.PurePath(filename).name != filename or not filename.endswith(".mp4"):
            raise NuisanceValidationError("media filename is unsafe")
        video_id = pathlib.Path(filename).stem
        if filename in names or video_id in observed_video_ids:
            raise NuisanceValidationError("media receipt contains duplicate files")
        names.add(filename)
        observed_video_ids.add(video_id)
        media_path = root / filename
        if media_path.is_symlink():
            raise NuisanceValidationError(
                f"media file must not be a symlink: {filename}"
            )
        try:
            file_stat = media_path.stat()
        except OSError as exc:
            raise NuisanceValidationError(f"media file is missing: {filename}") from exc
        if not stat.S_ISREG(file_stat.st_mode):
            raise NuisanceValidationError(f"media file is not regular: {filename}")
        if stat.S_IMODE(file_stat.st_mode) & 0o222:
            raise NuisanceValidationError(f"media file is writable: {filename}")
        if file_stat.st_size != _positive_integer(
            item.get("size_bytes"), f"media size for {filename}"
        ):
            raise NuisanceValidationError(f"media file size differs: {filename}")
        expected_digest = _sha256_text(
            item.get("sha256"), f"media digest for {filename}"
        )
        if _sha256_path(media_path) != expected_digest:
            raise NuisanceValidationError(f"media file digest differs: {filename}")
        if item.get("stream_types") != ["audio", "video"]:
            raise NuisanceValidationError(f"media stream record differs: {filename}")
        normalized.append(
            {
                "video_id": video_id,
                "filename": filename,
                "size_bytes": file_stat.st_size,
                "sha256": expected_digest,
                "stream_types": ["audio", "video"],
            }
        )
    if observed_video_ids != expected_video_ids:
        raise NuisanceValidationError("media receipt video membership differs")
    if {entry.name for entry in root.iterdir()} != names:
        raise NuisanceValidationError("media root membership differs")
    normalized.sort(key=lambda item: item["video_id"])
    if _canonical_digest(normalized) != _sha256_text(
        expected_media_set_sha256, "expected media-set digest"
    ):
        raise NuisanceValidationError("media-set digest differs")
    probe = _mapping(receipt.get("stream_probe"), "media receipt stream probe")
    _exact_keys(probe, {"filename", "sha256", "version"}, "media receipt stream probe")
    _nonempty_text(probe.get("filename"), "stream-probe filename")
    _sha256_text(probe.get("sha256"), "stream-probe digest")
    _nonempty_text(probe.get("version"), "stream-probe version")
    return normalized


def probe_media_file(
    ffprobe_path: pathlib.Path,
    media_path: pathlib.Path,
    *,
    expected_size_bytes: int,
) -> dict[str, Any]:
    """Run one bounded metadata probe and normalize its strict JSON response."""

    probe = pathlib.Path(ffprobe_path)
    media = pathlib.Path(media_path)
    if probe.is_symlink() or not probe.is_file() or not os.access(probe, os.X_OK):
        raise NuisanceValidationError("ffprobe must be an executable non-symlink file")
    if media.is_symlink() or not media.is_file():
        raise NuisanceValidationError("media input must be a regular non-symlink file")
    expected_size = _positive_integer(expected_size_bytes, "expected media size")
    if media.stat().st_size != expected_size:
        raise NuisanceValidationError("media size changed before probing")
    try:
        completed = subprocess.run(
            [
                str(probe),
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(media),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NuisanceValidationError("ffprobe execution failed") from exc
    if completed.returncode != 0 or len(completed.stdout.encode("utf-8")) > 5_000_000:
        raise NuisanceValidationError("ffprobe response is missing or too large")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise NuisanceValidationError("ffprobe response is not valid JSON") from exc
    if media.stat().st_size != expected_size:
        raise NuisanceValidationError("media size changed during probing")
    return parse_ffprobe_record(payload, size_bytes=expected_size)


def _probe_identity(ffprobe_path: pathlib.Path) -> dict[str, str]:
    probe = pathlib.Path(ffprobe_path)
    if probe.is_symlink() or not probe.is_file() or not os.access(probe, os.X_OK):
        raise NuisanceValidationError("ffprobe must be an executable non-symlink file")
    try:
        completed = subprocess.run(
            [str(probe), "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NuisanceValidationError("ffprobe identity check failed") from exc
    first_line = completed.stdout.splitlines()[0] if completed.stdout else ""
    if completed.returncode != 0 or not first_line or len(first_line) > 500:
        raise NuisanceValidationError("ffprobe identity check failed")
    return {
        "filename": probe.name,
        "sha256": _sha256_path(probe),
        "version": first_line,
    }


def _run_nuisance_gate_v2(
    *,
    config_bytes: bytes,
    config: Mapping[str, Any],
    expected_config_sha256: str,
    pilot_index_path: pathlib.Path,
    expected_pilot_index_sha256: str,
    media_receipt_path: pathlib.Path,
    expected_media_receipt_sha256: str,
    media_root: pathlib.Path,
    derived_media_root: pathlib.Path,
    expected_media_set_sha256: str,
    expected_implementation_bundle_sha256: str,
    nuisance_contract_path: pathlib.Path,
    expected_nuisance_contract_sha256: str,
    shortcut_preregistration_path: pathlib.Path,
    expected_shortcut_preregistration_sha256: str,
    shortcut_output_path: pathlib.Path,
    expected_shortcut_output_sha256: str,
    power_record_path: pathlib.Path,
    expected_power_record_sha256: str,
    dependency_subject_paths: Mapping[str, pathlib.Path],
    dependency_subject_sha256s: Mapping[str, str],
    dependency_record_paths: Mapping[str, pathlib.Path],
    dependency_record_sha256s: Mapping[str, str],
    replay_worker_path: pathlib.Path,
    expected_replay_worker_sha256: str,
    replay_worker_configuration_path: pathlib.Path,
    expected_replay_worker_configuration_sha256: str,
    replay_worker_model_path: pathlib.Path,
    expected_replay_worker_model_sha256: str,
    ffprobe_path: pathlib.Path,
    report_path: pathlib.Path,
    result_path: pathlib.Path,
    source_paths: Mapping[str, pathlib.Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate the complete v2 chain, replay it, and write once."""

    _, preregistration = _load_locked_json(
        shortcut_preregistration_path,
        expected_shortcut_preregistration_sha256,
        "shortcut preregistration",
    )
    validate_shortcut_preregistration_v2(
        preregistration,
        expected_configuration_sha256=expected_config_sha256,
        expected_gate_implementation_bundle_sha256=(
            expected_implementation_bundle_sha256
        ),
    )
    _, contract = _load_locked_json(
        nuisance_contract_path,
        expected_nuisance_contract_sha256,
        "nuisance contract",
    )
    validate_nuisance_contract(
        contract,
        expected_pilot_index_sha256=expected_pilot_index_sha256,
        expected_media_receipt_sha256=expected_media_receipt_sha256,
        expected_media_set_sha256=expected_media_set_sha256,
        expected_configuration_sha256=expected_config_sha256,
        expected_implementation_bundle_sha256=expected_implementation_bundle_sha256,
        expected_shortcut_preregistration_sha256=(
            expected_shortcut_preregistration_sha256
        ),
    )

    expected_roles = set(_V2_DEPENDENCY_ROLES)
    if (
        set(dependency_subject_paths) != expected_roles
        or set(dependency_subject_sha256s) != expected_roles
        or set(dependency_record_paths) != expected_roles
        or set(dependency_record_sha256s) != expected_roles
    ):
        raise NuisanceValidationError("v2 dependency role set differs")
    preregistration_digests = preregistration["input_digests"]
    dependency_records: dict[str, Mapping[str, Any]] = {}
    for role, (subject_key, _) in _V2_DEPENDENCY_ROLES.items():
        subject_sha256 = _authenticate_locked_file(
            dependency_subject_paths[role],
            dependency_subject_sha256s[role],
            f"v2 {role} subject",
        )
        if subject_sha256 != preregistration_digests[subject_key]:
            raise NuisanceValidationError(f"v2 {role} subject binding differs")
        _, dependency_record = _load_locked_json(
            dependency_record_paths[role],
            dependency_record_sha256s[role],
            f"v2 {role} verification record",
        )
        dependency_records[role] = dependency_record
    _, power_record = _load_locked_json(
        power_record_path,
        expected_power_record_sha256,
        "v2 power record",
    )
    validate_v2_dependency_chain(
        preregistration,
        power_record=power_record,
        power_record_sha256=expected_power_record_sha256,
        dependency_records=dependency_records,
        dependency_record_sha256s=dependency_record_sha256s,
    )

    _, shortcut_output = _load_locked_json(
        shortcut_output_path,
        expected_shortcut_output_sha256,
        "shortcut output",
    )
    validate_shortcut_output_against_preregistration_v2(
        shortcut_output,
        preregistration,
        expected_shortcut_preregistration_sha256=(
            expected_shortcut_preregistration_sha256
        ),
        expected_configuration_sha256=expected_config_sha256,
        expected_gate_implementation_bundle_sha256=(
            expected_implementation_bundle_sha256
        ),
    )

    if set(source_paths) != set(IMPLEMENTATION_SOURCE_ROLES):
        raise NuisanceValidationError("implementation source role set differs")
    source_digests: dict[str, str] = {}
    for role in IMPLEMENTATION_SOURCE_ROLES:
        source = pathlib.Path(source_paths[role])
        if source.is_symlink() or not source.is_file():
            raise NuisanceValidationError("implementation source is missing or unsafe")
        source_digests[role] = _sha256_path(source)
    if _canonical_digest(dict(sorted(source_digests.items()))) != _sha256_text(
        expected_implementation_bundle_sha256,
        "expected implementation-bundle digest",
    ):
        raise NuisanceValidationError("implementation-bundle digest differs")

    worker_sha256 = _authenticate_locked_file(
        replay_worker_path,
        expected_replay_worker_sha256,
        "v2 replay worker",
    )
    worker_configuration_sha256 = _authenticate_locked_file(
        replay_worker_configuration_path,
        expected_replay_worker_configuration_sha256,
        "v2 replay-worker configuration",
    )
    worker_model_sha256 = _authenticate_locked_file(
        replay_worker_model_path,
        expected_replay_worker_model_sha256,
        "v2 replay-worker model",
    )
    if (
        worker_configuration_sha256
        != preregistration["replay_worker"]["configuration_sha256"]
    ):
        raise NuisanceValidationError("v2 replay-worker configuration binding differs")
    if worker_model_sha256 != preregistration["replay_worker"]["model_sha256"]:
        raise NuisanceValidationError("v2 replay-worker model binding differs")
    worker = SubprocessReplayWorkerSpec(
        executable_path=pathlib.Path(replay_worker_path),
        executable_sha256=worker_sha256,
        configuration_path=pathlib.Path(replay_worker_configuration_path),
        configuration_sha256=worker_configuration_sha256,
        model_path=pathlib.Path(replay_worker_model_path),
        model_sha256=worker_model_sha256,
    )

    _, pilot = _load_locked_json(
        pilot_index_path, expected_pilot_index_sha256, "pilot index"
    )
    try:
        from .perception_media_pilot import validate_media_pilot_attestation
        from .perception_omni_gate import _validate_pilot

        validate_media_pilot_attestation(pilot)
        pairs = _validate_pilot(pilot)
    except (TypeError, ValueError) as exc:
        raise NuisanceValidationError("pilot index validation failed") from exc
    _, receipt = _load_locked_json(
        media_receipt_path, expected_media_receipt_sha256, "media receipt"
    )
    expected_video_ids = {
        pair[source_role]["video_id"]
        for pair in pairs
        for source_role in ("target", "donor")
    }
    media_rows = validate_media_receipt(
        receipt,
        media_root=media_root,
        expected_pilot_index_sha256=expected_pilot_index_sha256,
        expected_media_set_sha256=expected_media_set_sha256,
        expected_video_ids=expected_video_ids,
    )
    probe_identity = _probe_identity(ffprobe_path)
    if dict(_mapping(receipt.get("stream_probe"), "media receipt stream probe")) != (
        probe_identity
    ):
        raise NuisanceValidationError("ffprobe identity differs from the media receipt")
    root = pathlib.Path(media_root).resolve(strict=True)
    diagnostics: dict[str, dict[str, Any]] = {}
    for media_row in media_rows:
        media_path = root / media_row["filename"]
        before_digest = _sha256_path(media_path)
        diagnostics[media_row["video_id"]] = probe_media_file(
            ffprobe_path,
            media_path,
            expected_size_bytes=int(media_row["size_bytes"]),
        )
        if (
            _sha256_path(media_path) != before_digest
            or before_digest != media_row["sha256"]
        ):
            raise NuisanceValidationError(
                f"media changed during diagnostics: {media_row['filename']}"
            )
    records = build_feature_records(pairs, diagnostics)
    report = evaluate_shortcut_controls_v2(
        shortcut_output,
        preregistration,
        records,
        config,
        expected_configuration_sha256=hashlib.sha256(config_bytes).hexdigest(),
        expected_gate_implementation_bundle_sha256=(
            expected_implementation_bundle_sha256
        ),
        expected_shortcut_preregistration_sha256=(
            expected_shortcut_preregistration_sha256
        ),
        expected_shortcut_output_sha256=expected_shortcut_output_sha256,
        replay_worker=worker,
        derived_media_root=derived_media_root,
    )
    validate_shortcut_report_bindings_v2(
        report,
        expected_configuration_sha256=expected_config_sha256,
        expected_shortcut_preregistration_sha256=(
            expected_shortcut_preregistration_sha256
        ),
        expected_shortcut_output_sha256=expected_shortcut_output_sha256,
        expected_output_attestation_sha256=shortcut_output["attestation_sha256"],
        expected_record_inventory_sha256=preregistration["inventory"][
            "record_inventory_sha256"
        ],
        expected_power_record_sha256=expected_power_record_sha256,
        expected_dependency_verification_sha256s=dependency_record_sha256s,
        expected_replay_worker=preregistration["replay_worker"],
        expected_media_inventory_sha256=preregistration["inventory"][
            "media_inventory_sha256"
        ],
    )
    report_payload = _json_bytes(report)
    result = build_shortcut_result_v2(
        report=report,
        report_sha256=hashlib.sha256(report_payload).hexdigest(),
    )
    write_shortcut_gate_outputs_v2(
        report_path,
        result_path,
        report=report,
        result=result,
    )
    return report, result


def run_nuisance_gate(
    *,
    config_path: pathlib.Path,
    expected_config_sha256: str,
    pilot_index_path: pathlib.Path,
    expected_pilot_index_sha256: str,
    media_receipt_path: pathlib.Path,
    expected_media_receipt_sha256: str,
    media_root: pathlib.Path,
    expected_media_set_sha256: str,
    expected_implementation_bundle_sha256: str,
    nuisance_contract_path: pathlib.Path,
    expected_nuisance_contract_sha256: str,
    ffprobe_path: pathlib.Path,
    report_path: pathlib.Path,
    result_path: pathlib.Path,
    source_paths: Mapping[str, pathlib.Path],
    derived_media_root: pathlib.Path | None = None,
    shortcut_preregistration_path: pathlib.Path | None = None,
    expected_shortcut_preregistration_sha256: str | None = None,
    shortcut_output_path: pathlib.Path | None = None,
    expected_shortcut_output_sha256: str | None = None,
    power_record_path: pathlib.Path | None = None,
    expected_power_record_sha256: str | None = None,
    dependency_subject_paths: Mapping[str, pathlib.Path] | None = None,
    dependency_subject_sha256s: Mapping[str, str] | None = None,
    dependency_record_paths: Mapping[str, pathlib.Path] | None = None,
    dependency_record_sha256s: Mapping[str, str] | None = None,
    replay_worker_path: pathlib.Path | None = None,
    expected_replay_worker_sha256: str | None = None,
    replay_worker_configuration_path: pathlib.Path | None = None,
    expected_replay_worker_configuration_sha256: str | None = None,
    replay_worker_model_path: pathlib.Path | None = None,
    expected_replay_worker_model_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate, evaluate, and write a complete nuisance-gate result."""

    config_bytes, config = _load_locked_json(
        config_path, expected_config_sha256, "configuration"
    )
    _validate_config(config)
    required_v2 = {
        "derived media root": derived_media_root,
        "shortcut preregistration": shortcut_preregistration_path,
        "shortcut preregistration digest": expected_shortcut_preregistration_sha256,
        "shortcut output": shortcut_output_path,
        "shortcut output digest": expected_shortcut_output_sha256,
        "power record": power_record_path,
        "power record digest": expected_power_record_sha256,
        "dependency subject paths": dependency_subject_paths,
        "dependency subject digests": dependency_subject_sha256s,
        "dependency verification paths": dependency_record_paths,
        "dependency verification digests": dependency_record_sha256s,
        "replay worker": replay_worker_path,
        "replay worker digest": expected_replay_worker_sha256,
        "replay worker configuration": replay_worker_configuration_path,
        "replay worker configuration digest": (
            expected_replay_worker_configuration_sha256
        ),
        "replay worker model": replay_worker_model_path,
        "replay worker model digest": expected_replay_worker_model_sha256,
    }
    if config["schema_version"] == 2:
        missing = sorted(name for name, value in required_v2.items() if value is None)
        if missing:
            raise NuisanceValidationError(
                "configuration v2 is missing required inputs: " + ", ".join(missing)
            )
        raise NuisanceValidationError(
            "v2 production execution is disabled until frozen dependency "
            "replayers and power simulator interfaces, their authenticated "
            "transcripts, and an externally timestamped preregistration are supplied"
        )
        return _run_nuisance_gate_v2(
            config_bytes=config_bytes,
            config=config,
            expected_config_sha256=expected_config_sha256,
            pilot_index_path=pilot_index_path,
            expected_pilot_index_sha256=expected_pilot_index_sha256,
            media_receipt_path=media_receipt_path,
            expected_media_receipt_sha256=expected_media_receipt_sha256,
            media_root=media_root,
            derived_media_root=derived_media_root,
            expected_media_set_sha256=expected_media_set_sha256,
            expected_implementation_bundle_sha256=(
                expected_implementation_bundle_sha256
            ),
            nuisance_contract_path=nuisance_contract_path,
            expected_nuisance_contract_sha256=expected_nuisance_contract_sha256,
            shortcut_preregistration_path=shortcut_preregistration_path,
            expected_shortcut_preregistration_sha256=(
                expected_shortcut_preregistration_sha256
            ),
            shortcut_output_path=shortcut_output_path,
            expected_shortcut_output_sha256=expected_shortcut_output_sha256,
            power_record_path=power_record_path,
            expected_power_record_sha256=expected_power_record_sha256,
            dependency_subject_paths=dependency_subject_paths,
            dependency_subject_sha256s=dependency_subject_sha256s,
            dependency_record_paths=dependency_record_paths,
            dependency_record_sha256s=dependency_record_sha256s,
            replay_worker_path=replay_worker_path,
            expected_replay_worker_sha256=expected_replay_worker_sha256,
            replay_worker_configuration_path=replay_worker_configuration_path,
            expected_replay_worker_configuration_sha256=(
                expected_replay_worker_configuration_sha256
            ),
            replay_worker_model_path=replay_worker_model_path,
            expected_replay_worker_model_sha256=(expected_replay_worker_model_sha256),
            ffprobe_path=ffprobe_path,
            report_path=report_path,
            result_path=result_path,
            source_paths=source_paths,
        )
    if any(value is not None for value in required_v2.values()):
        raise NuisanceValidationError(
            "configuration v1 must not receive v2-only inputs"
        )
    _, pilot = _load_locked_json(
        pilot_index_path, expected_pilot_index_sha256, "pilot index"
    )
    try:
        from .perception_media_pilot import validate_media_pilot_attestation
        from .perception_omni_gate import _validate_pilot

        validate_media_pilot_attestation(pilot)
        pairs = _validate_pilot(pilot)
    except (TypeError, ValueError) as exc:
        raise NuisanceValidationError("pilot index validation failed") from exc

    _, receipt = _load_locked_json(
        media_receipt_path, expected_media_receipt_sha256, "media receipt"
    )
    expected_video_ids = {
        pair[source_role]["video_id"]
        for pair in pairs
        for source_role in ("target", "donor")
    }
    media_rows = validate_media_receipt(
        receipt,
        media_root=media_root,
        expected_pilot_index_sha256=expected_pilot_index_sha256,
        expected_media_set_sha256=expected_media_set_sha256,
        expected_video_ids=expected_video_ids,
    )
    if set(source_paths) != set(IMPLEMENTATION_SOURCE_ROLES):
        raise NuisanceValidationError("implementation source role set differs")
    source_digests = {}
    for role in IMPLEMENTATION_SOURCE_ROLES:
        source_path = source_paths[role]
        source = pathlib.Path(source_path)
        if source.is_symlink() or not source.is_file():
            raise NuisanceValidationError("implementation source is missing or unsafe")
        source_digests[role] = _sha256_path(source)
    observed_implementation_bundle = _canonical_digest(
        dict(sorted(source_digests.items()))
    )
    if observed_implementation_bundle != _sha256_text(
        expected_implementation_bundle_sha256,
        "expected implementation-bundle digest",
    ):
        raise NuisanceValidationError("implementation-bundle digest differs")

    _, contract = _load_locked_json(
        nuisance_contract_path,
        expected_nuisance_contract_sha256,
        "nuisance contract",
    )
    validate_nuisance_contract(
        contract,
        expected_pilot_index_sha256=expected_pilot_index_sha256,
        expected_media_receipt_sha256=expected_media_receipt_sha256,
        expected_media_set_sha256=expected_media_set_sha256,
        expected_configuration_sha256=expected_config_sha256,
        expected_implementation_bundle_sha256=expected_implementation_bundle_sha256,
    )
    probe_identity = _probe_identity(ffprobe_path)
    if (
        dict(_mapping(receipt.get("stream_probe"), "media receipt stream probe"))
        != probe_identity
    ):
        raise NuisanceValidationError("ffprobe identity differs from the media receipt")

    root = pathlib.Path(media_root).resolve(strict=True)
    diagnostics: dict[str, dict[str, Any]] = {}
    for media_row in media_rows:
        media_path = root / media_row["filename"]
        before_digest = _sha256_path(media_path)
        diagnostics[media_row["video_id"]] = probe_media_file(
            ffprobe_path,
            media_path,
            expected_size_bytes=int(media_row["size_bytes"]),
        )
        if (
            _sha256_path(media_path) != before_digest
            or before_digest != media_row["sha256"]
        ):
            raise NuisanceValidationError(
                f"media changed during diagnostics: {media_row['filename']}"
            )

    records = build_feature_records(pairs, diagnostics)
    report = evaluate_nuisance_records(records, config)
    report.pop("attestation_sha256")
    report.update(
        {
            "configuration": dict(config),
            "input_digests": {
                "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "pilot_index_sha256": expected_pilot_index_sha256,
                "media_receipt_sha256": expected_media_receipt_sha256,
                "media_set_sha256": expected_media_set_sha256,
                "nuisance_contract_sha256": expected_nuisance_contract_sha256,
                "implementation_bundle_sha256": expected_implementation_bundle_sha256,
            },
            "runtime": {
                "numpy_version": np.__version__,
                "ffprobe": probe_identity,
                "implementation_source_sha256": dict(sorted(source_digests.items())),
            },
            "counts": {
                "media_file_count": len(media_rows),
                "feature_record_count": len(records),
                "component_count": len({record["component_id"] for record in records}),
            },
            "records": records,
        }
    )
    report["attestation_sha256"] = _canonical_digest(report)
    report_payload = _json_bytes(report)
    result = build_handoff_result(
        report=report,
        configuration=config,
        report_sha256=hashlib.sha256(report_payload).hexdigest(),
        pilot_index_sha256=expected_pilot_index_sha256,
        media_receipt_sha256=expected_media_receipt_sha256,
        media_set_sha256=expected_media_set_sha256,
        nuisance_contract_sha256=expected_nuisance_contract_sha256,
        configuration_sha256=expected_config_sha256,
        implementation_bundle_sha256=expected_implementation_bundle_sha256,
    )
    write_gate_outputs(
        report_path,
        result_path,
        report=report,
        result=result,
        configuration=config,
    )
    return report, result


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NuisanceValidationError(f"output is not strict JSON: {exc}") from exc


def _validate_report_attestation(report: Mapping[str, Any]) -> None:
    attestation = _sha256_text(
        report.get("attestation_sha256"), "nuisance-report attestation"
    )
    unsigned = dict(report)
    unsigned.pop("attestation_sha256")
    if _canonical_digest(unsigned) != attestation:
        raise NuisanceValidationError("nuisance-report attestation differs")
    if report.get("schema") != "conflictbench.perception-nuisance-report.v1":
        raise NuisanceValidationError("nuisance-report schema differs")


def _write_once(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise NuisanceValidationError(f"output already exists: {path.name}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        path.chmod(0o400)
    except FileExistsError as exc:
        raise NuisanceValidationError(f"output already exists: {path.name}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def write_gate_outputs(
    report_path: pathlib.Path,
    result_path: pathlib.Path,
    *,
    report: Mapping[str, Any],
    result: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> None:
    """Write the detailed report and minimal handoff result exactly once."""

    report_target = pathlib.Path(report_path)
    result_target = pathlib.Path(result_path)
    if report_target == result_target:
        raise NuisanceValidationError("report and result paths must differ")
    if any(
        path.exists() or path.is_symlink() for path in (report_target, result_target)
    ):
        raise NuisanceValidationError("output already exists")
    _validate_report_attestation(report)
    input_digests = _mapping(
        result.get("input_digests"), "nuisance-result input digests"
    )
    decision = _mapping(result.get("decision"), "nuisance-result decision")
    validate_handoff_result(
        result,
        report=report,
        expected_configuration=configuration,
        expected_pilot_index_sha256=input_digests.get("pilot_index_sha256"),
        expected_media_receipt_sha256=input_digests.get("media_receipt_sha256"),
        expected_media_set_sha256=input_digests.get("media_set_sha256"),
        expected_nuisance_contract_sha256=input_digests.get("nuisance_contract_sha256"),
        expected_configuration_sha256=input_digests.get("configuration_sha256"),
        expected_implementation_bundle_sha256=input_digests.get(
            "implementation_bundle_sha256"
        ),
        expected_report_sha256=decision.get("report_sha256"),
    )
    report_payload = _json_bytes(report)
    if hashlib.sha256(report_payload).hexdigest() != decision.get("report_sha256"):
        raise NuisanceValidationError("handoff result does not bind the report bytes")
    result_payload = _json_bytes(result)
    _write_once(report_target, report_payload)
    try:
        _write_once(result_target, result_payload)
    except Exception:
        report_target.unlink(missing_ok=True)
        raise
