"""Frozen multimodal inference and source-sufficiency evaluation.

The runner authenticates a deterministic training-only candidate index, hashes
every required MP4 before inference, evaluates the complete index, and emits a
self-authenticating JSON result. Heavy model and media dependencies load only
when the first inference batch is executed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import pathlib
import re
import stat
import statistics
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Protocol

from .perception_candidate_audit import normalize_text
from .perception_media_pilot import (
    PilotConstructionError,
    validate_media_pilot_attestation,
)

MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
MODEL_REVISION = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
TRANSFORMERS_VERSION = "4.57.6"
QWEN_OMNI_UTILS_VERSION = "0.0.8"
TORCH_VERSION = "2.11.0+cu126"
CHOICE_SCORE_METHOD = "diagnostic_contextual_first_generated_token_logit"
OUTPUT_SCHEMA = "conflictbench.perception-omni-source-gate.v4"
INFERENCE_TRANSCRIPT_SCHEMA = "conflictbench.perception-omni-inference-transcript.v1"
CONDITIONS = ("audio_only", "video_only", "audiovisual")
SOURCE_ROLES = ("target", "donor")
PAIR_ROLES = ("same_answer_nuisance", "opposite_answer_candidate")
EVALUATIONS = (
    "source_sufficiency",
    "question_only",
    "same_question_shuffled_media",
)
CONTROL_TYPES = EVALUATIONS[1:]
PARTITIONS = ("scorer_fit", "threshold_calibration", "pilot_gate")
CHOICES = ("A", "B", "C")
EXPECTED_TARGET_COUNT = 100
EXPECTED_PAIR_COUNT = 200
EXPECTED_ELIGIBLE_KEY_COUNT = 85
EXPECTED_ELIGIBLE_KEY_CARDINALITY = {"2": 55, "3": 30}
EXPECTED_PARTITION_TARGET_COUNTS = {
    "scorer_fit": 60,
    "threshold_calibration": 20,
    "pilot_gate": 20,
}
VIDEO_PREPROCESSING = {
    "fps": 2.0,
    "min_frames": 4,
    "max_frames": 32,
    "min_pixels": 100352,
    "max_pixels": 200704,
}
QUESTION_KEY_SPLIT_POLICY = (
    "pilot_gate_normalized_questions_absent_from_fit_or_threshold"
)
_FFPROBE_TIMEOUT_SECONDS = 15

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_HEX24_RE = re.compile(r"[0-9a-f]{24}\Z")
_VIDEO_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_CONFIG_KEYS = {
    "schema_version",
    "runner",
    "model",
    "inference",
    "controls",
    "input",
    "gate",
}
_MODEL_KEYS = {
    "id",
    "revision",
    "dtype",
    "attention_implementation",
    "device_map",
    "disable_talker",
}
_INFERENCE_KEYS = {
    "batch_size",
    "maximum_new_tokens",
    "do_sample",
    "seed",
    "conditions",
    "emit_choice_scores_when_single_token",
    "video_preprocessing",
}
_INPUT_KEYS = {
    "official_split",
    "media_extension",
    "selection_policy",
    "validation_split_access",
}
_CONTROL_KEYS = {
    "types",
    "source_orientations",
    "same_question_shuffle_policy",
    "option_order_policy",
    "option_order_seed",
    "outcome_based_cohort_selection",
}
_GATE_KEYS = {
    "evaluation_partition",
    "minimum_strict_parse_rate",
    "minimum_unique_target_accuracy_by_condition",
    "minimum_donor_accuracy_by_role_and_condition",
    "minimum_pair_support_rate_by_role_and_condition",
    "maximum_question_only_source_answer_accuracy_by_orientation",
    "maximum_shuffled_media_source_answer_accuracy_by_condition_and_orientation",
    "minimum_aligned_over_control_margin_by_condition_and_orientation",
    "question_key_sensitivity",
    "minimum_unseen_question_component_count",
    "maximum_unseen_question_margin_attenuation",
    "require_choice_scores",
    "outcome_based_sample_selection",
}
_PILOT_KEYS = {
    "schema_version",
    "status",
    "scope",
    "seed",
    "implementation_sha256",
    "implementation_source_sha256",
    "input_digests",
    "source_use",
    "annotation_use_disclosure",
    "source_attribution",
    "change_note",
    "construction_rules",
    "counts",
    "partition_counts",
    "strata",
    "exclusions",
    "donor_reuse_events",
    "components",
    "gate_contract",
    "candidate_index",
    "attestation_sha256",
}
_PILOT_DIGEST_KEYS = {
    "configuration_sha256",
    "structural_audit_sha256",
    "structural_audit_configuration_sha256",
    "train_archive_sha256",
    "validation_archive_sha256",
}
_SOURCE_USE_KEYS = {
    "official_splits_in_candidate_index",
    "validation_archive_sha256_verified",
    "builder_validation_annotations_loaded",
    "builder_test_annotations_loaded",
    "media_loaded",
    "model_outputs_loaded",
}
_ANNOTATION_USE_KEYS = {
    "eligible_keys_derived_from_authenticated_train_archive",
    "train_eligibility_min_supported_answers",
    "train_eligibility_min_unique_videos_per_supported_answer",
    "structural_audit_authenticated",
    "structural_audit_used_for_candidate_selection",
    "structural_audit_train_labels_inspected",
    "structural_audit_validation_labels_inspected",
    "structural_audit_validation_use",
    "builder_train_labels_loaded",
    "builder_validation_labels_loaded",
    "builder_test_annotations_loaded",
}
_PILOT_IMPLEMENTATION_SOURCE_KEYS = {
    "conflictbench.perception_candidate_audit",
    "conflictbench.perception_media_pilot",
}
_PILOT_COUNT_KEYS = {
    "eligible_key_count",
    "eligible_key_count_by_supported_answer_count",
    "target_count",
    "pair_count",
    "same_answer_pair_count",
    "opposite_answer_pair_count",
    "unique_target_video_count",
    "unique_donor_count",
    "component_count",
    "exclusion_count",
}
_PAIR_KEYS = {
    "pair_id",
    "component_id",
    "official_split",
    "partition",
    "key_sha256",
    "role",
    "anchor",
    "target",
    "donor",
}
_SOURCE_KEYS = {"video_id", "question_id", "question", "options", "answer_id", "answer"}
_COMPONENT_KEYS = {"component_id", "partition", "target_count", "video_ids"}
_CONTRACT_KEYS = {
    "version",
    "status",
    "fit_partition",
    "threshold_partition",
    "evaluation_partition",
    "required_modalities",
    "source_answer_scoring_required",
    "nuisance_detection_required",
    "thresholds_must_be_frozen_before_evaluation",
    "scores_in_builder_forbidden",
    "required_later_records",
    "source_answer_rules",
    "nuisance_detection_rules",
    "decision_rule",
}
_OUTPUT_KEYS = {
    "schema",
    "model",
    "configuration",
    "inference",
    "input_digests",
    "source_files",
    "media_files",
    "runtime",
    "counts",
    "gate",
    "records",
    "candidate_design",
    "source_attribution",
    "change_note",
    "inference_transcript_sha256",
    "payload_sha256",
}
_DESIGN_KEYS = {
    "pair_id",
    "pair_role",
    "component_id",
    "partition",
    "key_sha256",
    "prompt",
    "prompt_sha256",
    "target",
    "donor",
}
_DESIGN_SOURCE_KEYS = {
    "video_id",
    "question_id",
    "normalized_question_sha256",
    "question",
    "expected_choice",
    "media_sha256",
    "balanced_options",
    "option_balance_slot",
    "prompt",
    "prompt_sha256",
    "shuffled_media_video_id",
    "shuffled_media_sha256",
    "shuffled_media_choice",
}
_RECORD_KEYS = {
    "record_id",
    "request_sha256",
    "pair_id",
    "component_id",
    "partition",
    "pair_role",
    "evaluation",
    "source_role",
    "condition",
    "video_id",
    "media_video_id",
    "question_id",
    "normalized_question_sha256",
    "expected_choice",
    "raw_response",
    "predicted_choice",
    "parse_status",
    "is_correct",
    "media_sha256",
    "shuffled_media_choice",
    "prompt_sha256",
    "choice_scores",
}
_CHOICE_SCORE_KEYS = {"status", "method", "values", "token_ids", "used_for_gate"}
_RUNTIME_KEYS = {
    "backend",
    "loaded",
    "torch_version",
    "transformers_version",
    "qwen_omni_utils_version",
    "cuda_runtime_version",
    "gpu_name",
    "model_dtype",
    "deterministic_algorithms",
    "cublas_workspace_config",
    "frozen",
}


class GateValidationError(ValueError):
    """Raised when a gate input, model result, or output fails validation."""


@dataclass(frozen=True)
class InferenceRequest:
    """One outcome-blind model request."""

    record_id: str
    pair_id: str
    pair_role: str
    source_role: str
    condition: str
    video_id: str
    media_path: pathlib.Path | None
    prompt: str
    evaluation: str = "source_sufficiency"
    media_video_id: str | None = None


@dataclass(frozen=True)
class InferenceResult:
    """Raw backend response with optional auditable choice scores."""

    record_id: str
    raw_response: str
    choice_scores: Mapping[str, float] | None = None
    choice_score_method: str | None = None
    choice_token_ids: Mapping[str, int] | None = None


class InferenceBackend(Protocol):
    """Minimal interface used by the deterministic evaluation driver."""

    def infer(
        self, requests: Sequence[InferenceRequest]
    ) -> Sequence[InferenceResult]: ...

    def runtime_details(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class _Task:
    request: InferenceRequest
    component_id: str
    partition: str
    pair_role: str
    question_id: int
    normalized_question_sha256: str
    expected_choice: str
    media_video_id: str | None
    media_sha256: str | None
    shuffled_media_choice: str | None
    prompt_sha256: str
    request_sha256: str


def _reject_json_constant(value: str) -> None:
    raise GateValidationError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GateValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GateValidationError(f"value is not strict JSON: {exc}") from exc


def _digest_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise GateValidationError(f"required file cannot be read: {path.name}") from exc
    return digest.hexdigest()


def _locked_json(
    path: pathlib.Path, expected_sha256: str, label: str
) -> tuple[bytes, Mapping[str, Any]]:
    if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(
        expected_sha256
    ):
        raise GateValidationError(
            f"expected {label} SHA-256 must be lowercase hexadecimal"
        )
    try:
        data = pathlib.Path(path).read_bytes()
    except OSError as exc:
        raise GateValidationError(f"{label} cannot be read") from exc
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise GateValidationError(f"{label} SHA-256 does not match")
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateValidationError(f"{label} is not strict UTF-8 JSON") from exc
    return data, _mapping(value, label)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise GateValidationError(f"{field} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    observed = set(value)
    if observed != expected:
        raise GateValidationError(
            f"{field} keys differ; missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise GateValidationError(f"{field} must be boolean")
    return value


def _positive_int(value: Any, field: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GateValidationError(f"{field} must be a positive integer")
    if maximum is not None and value > maximum:
        raise GateValidationError(f"{field} exceeds {maximum}")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GateValidationError(f"{field} must be a nonnegative integer")
    return value


def _fraction(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateValidationError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise GateValidationError(f"{field} must be finite and between zero and one")
    return result


def _plain_string(value: Any, field: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise GateValidationError(f"{field} must be a string")
    return value


def _sha256_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise GateValidationError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_thresholds(value: Any, field: str) -> dict[str, float]:
    value = _mapping(value, field)
    _exact_keys(value, set(CONDITIONS), field)
    return {
        condition: _fraction(value[condition], f"{field}.{condition}")
        for condition in CONDITIONS
    }


def _validate_role_thresholds(value: Any, field: str) -> dict[str, dict[str, float]]:
    value = _mapping(value, field)
    _exact_keys(value, set(PAIR_ROLES), field)
    return {
        role: _validate_thresholds(value[role], f"{field}.{role}")
        for role in PAIR_ROLES
    }


def _validate_orientation_thresholds(value: Any, field: str) -> dict[str, float]:
    value = _mapping(value, field)
    _exact_keys(value, set(SOURCE_ROLES), field)
    return {
        source_role: _fraction(value[source_role], f"{field}.{source_role}")
        for source_role in SOURCE_ROLES
    }


def _validate_orientation_condition_thresholds(
    value: Any, field: str
) -> dict[str, dict[str, float]]:
    value = _mapping(value, field)
    _exact_keys(value, set(SOURCE_ROLES), field)
    return {
        source_role: _validate_thresholds(value[source_role], f"{field}.{source_role}")
        for source_role in SOURCE_ROLES
    }


def _validate_video_preprocessing(value: Any) -> dict[str, Any]:
    value = _mapping(value, "inference.video_preprocessing")
    _exact_keys(value, set(VIDEO_PREPROCESSING), "inference.video_preprocessing")
    for key, expected in VIDEO_PREPROCESSING.items():
        observed = value[key]
        if type(observed) is not type(expected) or observed != expected:
            raise GateValidationError(
                "video preprocessing differs from the locked bounded-memory profile"
            )
    return dict(VIDEO_PREPROCESSING)


def _validate_config(value: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(value, _CONFIG_KEYS, "configuration")
    if value["schema_version"] != 4:
        raise GateValidationError("configuration.schema_version must equal 4")
    if value["runner"] != "perception_qwen25_omni_source_sufficiency":
        raise GateValidationError("configuration.runner is unexpected")

    model = _mapping(value["model"], "model")
    _exact_keys(model, _MODEL_KEYS, "model")
    if model["id"] != MODEL_ID:
        raise GateValidationError("configuration must use the pinned model ID")
    if model["revision"] != MODEL_REVISION:
        raise GateValidationError("configuration must use the pinned model revision")
    if model["dtype"] != "float16":
        raise GateValidationError("model.dtype must equal float16")
    if model["attention_implementation"] != "sdpa":
        raise GateValidationError("model.attention_implementation must equal sdpa")
    if model["device_map"] != "auto":
        raise GateValidationError("model.device_map must equal auto")
    if _bool(model["disable_talker"], "model.disable_talker") is not True:
        raise GateValidationError("model.disable_talker must be true")

    inference = _mapping(value["inference"], "inference")
    _exact_keys(inference, _INFERENCE_KEYS, "inference")
    batch_size = _positive_int(inference["batch_size"], "inference.batch_size", 32)
    if batch_size != 1:
        raise GateValidationError(
            "inference.batch_size must equal 1 so sampled video FPS is exact"
        )
    maximum_new_tokens = _positive_int(
        inference["maximum_new_tokens"], "inference.maximum_new_tokens", 16
    )
    if _bool(inference["do_sample"], "inference.do_sample") is not False:
        raise GateValidationError("inference.do_sample must be false")
    seed = _nonnegative_int(inference["seed"], "inference.seed")
    if inference["conditions"] != list(CONDITIONS):
        raise GateValidationError(
            "inference.conditions must use the fixed condition order"
        )
    emit_scores = _bool(
        inference["emit_choice_scores_when_single_token"],
        "inference.emit_choice_scores_when_single_token",
    )
    video_preprocessing = _validate_video_preprocessing(
        inference["video_preprocessing"]
    )

    controls = _mapping(value["controls"], "controls")
    _exact_keys(controls, _CONTROL_KEYS, "controls")
    if controls["types"] != list(CONTROL_TYPES):
        raise GateValidationError("controls.types must use the fixed control order")
    if controls["source_orientations"] != list(SOURCE_ROLES):
        raise GateValidationError(
            "controls source orientations must use the fixed target/donor order"
        )
    if (
        controls["same_question_shuffle_policy"]
        != "paired_opposite_answer_within_component"
    ):
        raise GateValidationError("controls shuffle policy is unexpected")
    if (
        controls["option_order_policy"]
        != "deterministic_balanced_by_partition_and_orientation"
    ):
        raise GateValidationError("controls option-order policy is unexpected")
    option_order_seed = _nonnegative_int(
        controls["option_order_seed"], "controls.option_order_seed"
    )
    if option_order_seed != seed:
        raise GateValidationError(
            "controls option-order seed must equal inference.seed"
        )
    if controls["outcome_based_cohort_selection"] != "forbidden":
        raise GateValidationError(
            "outcome-based control cohort selection must be forbidden"
        )

    input_config = _mapping(value["input"], "input")
    _exact_keys(input_config, _INPUT_KEYS, "input")
    expected_input = {
        "official_split": "train",
        "media_extension": ".mp4",
        "selection_policy": "entire_candidate_index",
        "validation_split_access": "forbidden",
    }
    if dict(input_config) != expected_input:
        raise GateValidationError(
            "input configuration violates the training-only full-index contract"
        )

    gate = _mapping(value["gate"], "gate")
    _exact_keys(gate, _GATE_KEYS, "gate")
    if gate["evaluation_partition"] != "pilot_gate":
        raise GateValidationError("gate.evaluation_partition must equal pilot_gate")
    parse_threshold = _fraction(
        gate["minimum_strict_parse_rate"], "gate.minimum_strict_parse_rate"
    )
    target_thresholds = _validate_thresholds(
        gate["minimum_unique_target_accuracy_by_condition"],
        "gate.minimum_unique_target_accuracy_by_condition",
    )
    donor_thresholds = _validate_role_thresholds(
        gate["minimum_donor_accuracy_by_role_and_condition"],
        "gate.minimum_donor_accuracy_by_role_and_condition",
    )
    pair_thresholds = _validate_role_thresholds(
        gate["minimum_pair_support_rate_by_role_and_condition"],
        "gate.minimum_pair_support_rate_by_role_and_condition",
    )
    question_only_maximums = _validate_orientation_thresholds(
        gate["maximum_question_only_source_answer_accuracy_by_orientation"],
        "gate.maximum_question_only_source_answer_accuracy_by_orientation",
    )
    if any(not 1.0 / 3.0 <= value <= 0.5 for value in question_only_maximums.values()):
        raise GateValidationError(
            "question-only maximum must be between chance and 0.5"
        )
    shuffled_maximums = _validate_orientation_condition_thresholds(
        gate[
            "maximum_shuffled_media_source_answer_accuracy_by_condition_and_orientation"
        ],
        "gate.maximum_shuffled_media_source_answer_accuracy_by_condition_and_orientation",
    )
    if any(
        not 1.0 / 3.0 <= threshold <= 0.5
        for by_condition in shuffled_maximums.values()
        for threshold in by_condition.values()
    ):
        raise GateValidationError(
            "shuffled-media maximum must be between chance and 0.5"
        )
    margin_minimums = _validate_orientation_condition_thresholds(
        gate["minimum_aligned_over_control_margin_by_condition_and_orientation"],
        "gate.minimum_aligned_over_control_margin_by_condition_and_orientation",
    )
    if any(
        threshold <= 0.0
        for by_condition in margin_minimums.values()
        for threshold in by_condition.values()
    ):
        raise GateValidationError("aligned-over-control margin must be positive")
    if gate["question_key_sensitivity"] != QUESTION_KEY_SPLIT_POLICY:
        raise GateValidationError("question-key sensitivity split policy is unexpected")
    minimum_unseen_components = _positive_int(
        gate["minimum_unseen_question_component_count"],
        "gate.minimum_unseen_question_component_count",
        EXPECTED_PARTITION_TARGET_COUNTS["pilot_gate"],
    )
    maximum_unseen_attenuation = _fraction(
        gate["maximum_unseen_question_margin_attenuation"],
        "gate.maximum_unseen_question_margin_attenuation",
    )
    if maximum_unseen_attenuation <= 0.0:
        raise GateValidationError(
            "maximum unseen-question margin attenuation must be positive"
        )
    require_scores = _bool(gate["require_choice_scores"], "gate.require_choice_scores")
    if require_scores and not emit_scores:
        raise GateValidationError(
            "required choice scores must be enabled by inference configuration"
        )
    if gate["outcome_based_sample_selection"] != "forbidden":
        raise GateValidationError("outcome-based sample selection must be forbidden")

    return {
        "schema_version": 4,
        "runner": value["runner"],
        "model": dict(model),
        "inference": {
            "batch_size": batch_size,
            "maximum_new_tokens": maximum_new_tokens,
            "do_sample": False,
            "seed": seed,
            "conditions": list(CONDITIONS),
            "emit_choice_scores_when_single_token": emit_scores,
            "video_preprocessing": video_preprocessing,
        },
        "controls": {
            "types": list(CONTROL_TYPES),
            "source_orientations": list(SOURCE_ROLES),
            "same_question_shuffle_policy": ("paired_opposite_answer_within_component"),
            "option_order_policy": (
                "deterministic_balanced_by_partition_and_orientation"
            ),
            "option_order_seed": option_order_seed,
            "outcome_based_cohort_selection": "forbidden",
        },
        "input": expected_input,
        "gate": {
            "evaluation_partition": "pilot_gate",
            "minimum_strict_parse_rate": parse_threshold,
            "minimum_unique_target_accuracy_by_condition": target_thresholds,
            "minimum_donor_accuracy_by_role_and_condition": donor_thresholds,
            "minimum_pair_support_rate_by_role_and_condition": pair_thresholds,
            "maximum_question_only_source_answer_accuracy_by_orientation": (
                question_only_maximums
            ),
            "maximum_shuffled_media_source_answer_accuracy_by_condition_and_orientation": (
                shuffled_maximums
            ),
            "minimum_aligned_over_control_margin_by_condition_and_orientation": (
                margin_minimums
            ),
            "question_key_sensitivity": QUESTION_KEY_SPLIT_POLICY,
            "minimum_unseen_question_component_count": minimum_unseen_components,
            "maximum_unseen_question_margin_attenuation": (maximum_unseen_attenuation),
            "require_choice_scores": require_scores,
            "outcome_based_sample_selection": "forbidden",
        },
    }


def _validate_source(value: Any, field: str) -> dict[str, Any]:
    value = _mapping(value, field)
    _exact_keys(value, _SOURCE_KEYS, field)
    video_id = _plain_string(value["video_id"], f"{field}.video_id")
    if (
        not _VIDEO_ID_RE.fullmatch(video_id)
        or pathlib.PurePosixPath(video_id).name != video_id
    ):
        raise GateValidationError(f"{field}.video_id is not a safe plain identifier")
    question_id = _nonnegative_int(value["question_id"], f"{field}.question_id")
    question = _plain_string(value["question"], f"{field}.question")
    options = value["options"]
    if not isinstance(options, list) or len(options) != 3:
        raise GateValidationError(f"{field}.options must contain exactly three choices")
    normalized_options = [
        _plain_string(option, f"{field}.options[{index}]")
        for index, option in enumerate(options)
    ]
    if len(set(normalized_options)) != 3:
        raise GateValidationError(f"{field}.options must be unique")
    answer_id = _nonnegative_int(value["answer_id"], f"{field}.answer_id")
    if answer_id >= 3:
        raise GateValidationError(f"{field}.answer_id is outside A/B/C")
    answer = _plain_string(value["answer"], f"{field}.answer")
    if answer != normalized_options[answer_id]:
        raise GateValidationError(f"{field}.answer does not match answer_id")
    return {
        "video_id": video_id,
        "question_id": question_id,
        "question": question,
        "options": normalized_options,
        "answer_id": answer_id,
        "answer": answer,
    }


def _validate_pilot(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        validate_media_pilot_attestation(value)
    except PilotConstructionError as exc:
        raise GateValidationError(f"pilot index attestation is invalid: {exc}") from exc
    _exact_keys(value, _PILOT_KEYS, "pilot index")
    if value["schema_version"] != 1:
        raise GateValidationError("pilot index schema_version must equal 1")
    if value["status"] != "construction_complete_scores_not_run":
        raise GateValidationError("pilot index must precede all score generation")
    if value["scope"] != "train_candidate_index_from_authenticated_train_labels":
        raise GateValidationError("pilot index scope is unexpected")
    _nonnegative_int(value["seed"], "pilot index seed")
    _sha256_text(value["implementation_sha256"], "pilot index implementation_sha256")
    implementation_sources = _mapping(
        value["implementation_source_sha256"],
        "pilot index implementation_source_sha256",
    )
    _exact_keys(
        implementation_sources,
        _PILOT_IMPLEMENTATION_SOURCE_KEYS,
        "pilot index implementation_source_sha256",
    )
    for name, digest in implementation_sources.items():
        _sha256_text(digest, f"pilot index implementation_source_sha256.{name}")
    _sha256_text(value["attestation_sha256"], "pilot index attestation_sha256")

    digests = _mapping(value["input_digests"], "pilot index input_digests")
    _exact_keys(digests, _PILOT_DIGEST_KEYS, "pilot index input_digests")
    for name, digest in digests.items():
        _sha256_text(digest, f"pilot index input_digests.{name}")

    source_use = _mapping(value["source_use"], "pilot index source_use")
    _exact_keys(source_use, _SOURCE_USE_KEYS, "pilot index source_use")
    if source_use["official_splits_in_candidate_index"] != ["train"]:
        raise GateValidationError(
            "pilot index must contain only the official train split"
        )
    if source_use["builder_validation_annotations_loaded"] is not False:
        raise GateValidationError(
            "the pilot builder must not load validation annotations"
        )
    if source_use["builder_test_annotations_loaded"] is not False:
        raise GateValidationError("the pilot builder must not load test annotations")
    if (
        source_use["media_loaded"] is not False
        or source_use["model_outputs_loaded"] is not False
    ):
        raise GateValidationError("pilot index must be outcome-blind")
    if (
        _bool(
            source_use["validation_archive_sha256_verified"],
            "source_use.validation_archive_sha256_verified",
        )
        is not True
    ):
        raise GateValidationError("validation archive hash must be verified")

    disclosure = _mapping(
        value["annotation_use_disclosure"],
        "pilot index annotation_use_disclosure",
    )
    _exact_keys(
        disclosure, _ANNOTATION_USE_KEYS, "pilot index annotation_use_disclosure"
    )
    expected_disclosure = {
        "eligible_keys_derived_from_authenticated_train_archive": True,
        "train_eligibility_min_supported_answers": 2,
        "train_eligibility_min_unique_videos_per_supported_answer": 5,
        "structural_audit_authenticated": True,
        "structural_audit_used_for_candidate_selection": False,
        "structural_audit_train_labels_inspected": True,
        "structural_audit_validation_labels_inspected": True,
        "structural_audit_validation_use": "schema_and_candidate_pool_counts_only",
        "builder_train_labels_loaded": True,
        "builder_validation_labels_loaded": False,
        "builder_test_annotations_loaded": False,
    }
    if dict(disclosure) != expected_disclosure:
        raise GateValidationError("pilot index annotation-use disclosure is unexpected")

    attribution = _mapping(
        value["source_attribution"], "pilot index source_attribution"
    )
    _exact_keys(
        attribution,
        {"dataset", "authors", "copyright", "materials_license", "license_url"},
        "pilot index source_attribution",
    )
    for name in ("dataset", "authors", "copyright"):
        _plain_string(attribution[name], f"pilot index source_attribution.{name}")
    if (
        attribution["materials_license"] != "CC-BY-4.0"
        or attribution["license_url"]
        != "https://creativecommons.org/licenses/by/4.0/legalcode"
    ):
        raise GateValidationError("pilot index attribution is not CC-BY-4.0")
    change_note = _plain_string(value["change_note"], "pilot index change_note")
    if "CC-BY-4.0" not in change_note:
        raise GateValidationError(
            "pilot index change note must preserve the CC-BY notice"
        )

    construction = _mapping(
        value["construction_rules"], "pilot index construction_rules"
    )
    expected_construction = {
        "official_split": "train",
        "min_supported_answers_per_key": 2,
        "min_unique_train_videos_per_supported_answer": 5,
        "pair_roles": list(PAIR_ROLES),
        "require_every_eligible_key": True,
        "prefer_unique_donors": True,
        "target_and_donor_video_disjoint_when_possible": True,
        "component_partition_lock": True,
    }
    if dict(construction) != expected_construction:
        raise GateValidationError("pilot index construction rules are unexpected")

    counts = _mapping(value["counts"], "pilot index counts")
    _exact_keys(counts, _PILOT_COUNT_KEYS, "pilot index counts")
    for name, count in counts.items():
        if name == "eligible_key_count_by_supported_answer_count":
            continue
        _nonnegative_int(count, f"pilot index counts.{name}")
    eligible_cardinality = _mapping(
        counts["eligible_key_count_by_supported_answer_count"],
        "pilot index counts.eligible_key_count_by_supported_answer_count",
    )
    _exact_keys(
        eligible_cardinality,
        {"2", "3"},
        "pilot index counts.eligible_key_count_by_supported_answer_count",
    )
    for cardinality, count in eligible_cardinality.items():
        _nonnegative_int(
            count,
            "pilot index counts.eligible_key_count_by_supported_answer_count."
            f"{cardinality}",
        )
    if sum(eligible_cardinality.values()) != counts["eligible_key_count"]:
        raise GateValidationError("pilot index eligible-key cardinalities do not sum")
    if (
        counts["eligible_key_count"] != EXPECTED_ELIGIBLE_KEY_COUNT
        or dict(eligible_cardinality) != EXPECTED_ELIGIBLE_KEY_CARDINALITY
    ):
        raise GateValidationError("pilot index eligible-key counts differ")
    if counts["target_count"] != EXPECTED_TARGET_COUNT:
        raise GateValidationError("pilot index must contain exactly 100 targets")
    if counts["pair_count"] != EXPECTED_PAIR_COUNT:
        raise GateValidationError("pilot index must contain exactly 200 pairs")
    if counts["same_answer_pair_count"] != EXPECTED_TARGET_COUNT:
        raise GateValidationError(
            "pilot index must contain exactly 100 same-answer pairs"
        )
    if counts["opposite_answer_pair_count"] != EXPECTED_TARGET_COUNT:
        raise GateValidationError(
            "pilot index must contain exactly 100 opposite-answer pairs"
        )
    if counts["unique_target_video_count"] != EXPECTED_TARGET_COUNT:
        raise GateValidationError("pilot index must contain 100 unique target videos")

    contract = _mapping(value["gate_contract"], "pilot index gate_contract")
    _exact_keys(contract, _CONTRACT_KEYS, "pilot index gate_contract")
    expected_contract = {
        "version": 1,
        "status": "not_run",
        "fit_partition": "scorer_fit",
        "threshold_partition": "threshold_calibration",
        "evaluation_partition": "pilot_gate",
        "required_modalities": ["audio", "video"],
        "source_answer_scoring_required": True,
        "nuisance_detection_required": True,
        "thresholds_must_be_frozen_before_evaluation": True,
        "scores_in_builder_forbidden": True,
    }
    for name, expected in expected_contract.items():
        if contract[name] != expected:
            raise GateValidationError(f"pilot index gate_contract.{name} is unexpected")
    if not isinstance(contract["required_later_records"], dict):
        raise GateValidationError(
            "pilot index required_later_records must be an object"
        )
    for name in ("source_answer_rules", "nuisance_detection_rules"):
        if not isinstance(contract[name], list) or not all(
            isinstance(item, str) for item in contract[name]
        ):
            raise GateValidationError(f"pilot index {name} must be a string list")
    _plain_string(contract["decision_rule"], "pilot index decision_rule")

    components_value = value["components"]
    if not isinstance(components_value, list) or not components_value:
        raise GateValidationError("pilot index components must be nonempty")
    components: dict[str, dict[str, Any]] = {}
    video_owners: dict[str, tuple[str, str]] = {}
    for index, raw_component in enumerate(components_value):
        component = _mapping(raw_component, f"components[{index}]")
        _exact_keys(component, _COMPONENT_KEYS, f"components[{index}]")
        component_id = _plain_string(component["component_id"], "component_id")
        if not _HEX24_RE.fullmatch(component_id) or component_id in components:
            raise GateValidationError(
                "component IDs must be unique 24-character hex values"
            )
        partition = component["partition"]
        if partition not in PARTITIONS:
            raise GateValidationError("component partition is unexpected")
        _positive_int(component["target_count"], "component target_count")
        video_ids = component["video_ids"]
        if not isinstance(video_ids, list) or not video_ids:
            raise GateValidationError("component video_ids must be nonempty")
        if video_ids != sorted(set(video_ids)):
            raise GateValidationError("component video_ids must be sorted and unique")
        expected_component_id = hashlib.sha256(
            json.dumps(video_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24]
        if component_id != expected_component_id:
            raise GateValidationError(
                "component ID differs from its canonical video set"
            )
        for video_id in video_ids:
            if not isinstance(video_id, str) or not _VIDEO_ID_RE.fullmatch(video_id):
                raise GateValidationError("component has an unsafe video ID")
            if video_id in video_owners:
                raise GateValidationError("one video is listed in multiple components")
            video_owners[video_id] = (component_id, partition)
        components[component_id] = dict(component)
    if list(components.values()) != sorted(
        components.values(), key=lambda item: (item["partition"], item["component_id"])
    ):
        raise GateValidationError("pilot components are not in canonical order")

    pairs_value = value["candidate_index"]
    if not isinstance(pairs_value, list) or not pairs_value:
        raise GateValidationError("pilot candidate index must be nonempty")
    pairs: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    target_roles: dict[tuple[str, int], set[str]] = defaultdict(set)
    target_donors: dict[tuple[str, int], set[str]] = defaultdict(set)
    target_locations: dict[tuple[str, int], tuple[str, str, str]] = {}
    target_payloads: dict[tuple[str, int], dict[str, Any]] = {}
    source_payloads: dict[tuple[str, int], dict[str, Any]] = {}
    component_partitions: dict[str, str] = {}
    component_targets: dict[str, set[tuple[str, int]]] = defaultdict(set)
    graph: dict[str, set[str]] = defaultdict(set)
    used_videos: set[str] = set()
    for index, raw_pair in enumerate(pairs_value):
        pair = _mapping(raw_pair, f"candidate_index[{index}]")
        _exact_keys(pair, _PAIR_KEYS, f"candidate_index[{index}]")
        pair_id = _plain_string(pair["pair_id"], f"candidate_index[{index}].pair_id")
        if not _HEX24_RE.fullmatch(pair_id) or pair_id in pair_ids:
            raise GateValidationError("pair IDs must be unique 24-character hex values")
        pair_ids.add(pair_id)
        component_id = pair["component_id"]
        if component_id not in components:
            raise GateValidationError("pair references an unknown component")
        partition = pair["partition"]
        if (
            partition not in PARTITIONS
            or components[component_id]["partition"] != partition
        ):
            raise GateValidationError("pair partition differs from its component")
        if pair["official_split"] != "train":
            raise GateValidationError("pair must use the official train split")
        _sha256_text(pair["key_sha256"], f"candidate_index[{index}].key_sha256")
        role = pair["role"]
        if role not in {"same_answer_nuisance", "opposite_answer_candidate"}:
            raise GateValidationError("pair role is unexpected")
        anchor = _mapping(pair["anchor"], f"candidate_index[{index}].anchor")
        if set(anchor) != {"question", "options", "answer_id", "answer"}:
            raise GateValidationError("pair anchor keys differ")
        target = _validate_source(pair["target"], f"candidate_index[{index}].target")
        donor = _validate_source(pair["donor"], f"candidate_index[{index}].donor")
        for source in (target, donor):
            source_key = (source["video_id"], source["question_id"])
            prior_source = source_payloads.setdefault(source_key, source)
            if prior_source != source:
                raise GateValidationError(
                    "one video-question source has conflicting payloads"
                )
        expected_anchor = {
            "question": target["question"],
            "options": target["options"],
            "answer_id": target["answer_id"],
            "answer": target["answer"],
        }
        if dict(anchor) != expected_anchor:
            raise GateValidationError("pair anchor differs from target source")
        if donor["question"] != target["question"] or sorted(
            donor["options"]
        ) != sorted(target["options"]):
            raise GateValidationError("target and donor question vocabularies differ")
        if target["video_id"] == donor["video_id"]:
            raise GateValidationError("target and donor must be different videos")
        if role == "same_answer_nuisance" and donor["answer"] != target["answer"]:
            raise GateValidationError("same-answer pair has different answers")
        if role == "opposite_answer_candidate" and donor["answer"] == target["answer"]:
            raise GateValidationError("opposite-answer pair has the same answer")
        component_videos = set(components[component_id]["video_ids"])
        if (
            target["video_id"] not in component_videos
            or donor["video_id"] not in component_videos
        ):
            raise GateValidationError("pair videos are absent from their component")
        prior_partition = component_partitions.setdefault(component_id, partition)
        if prior_partition != partition:
            raise GateValidationError("component crosses partitions")
        target_roles[(target["video_id"], target["question_id"])].add(role)
        target_key = (target["video_id"], target["question_id"])
        target_donors[target_key].add(donor["video_id"])
        location = (component_id, partition, pair["key_sha256"])
        prior_location = target_locations.setdefault(target_key, location)
        if prior_location != location:
            raise GateValidationError(
                "one target is assigned across components or partitions"
            )
        prior_target = target_payloads.setdefault(target_key, target)
        if prior_target != target:
            raise GateValidationError("one target differs across its two pair roles")
        component_targets[component_id].add(target_key)
        graph[target["video_id"]].add(donor["video_id"])
        graph[donor["video_id"]].add(target["video_id"])
        used_videos.update((target["video_id"], donor["video_id"]))
        pairs.append(
            {
                **dict(pair),
                "anchor": dict(anchor),
                "target": target,
                "donor": donor,
            }
        )

    if pairs != sorted(
        pairs,
        key=lambda item: (item["partition"], item["component_id"], item["pair_id"]),
    ):
        raise GateValidationError("pilot candidate index is not in canonical order")
    expected_roles = {"same_answer_nuisance", "opposite_answer_candidate"}
    if any(roles != expected_roles for roles in target_roles.values()):
        raise GateValidationError("each target must have exactly both pair roles")
    if any(len(donors) != len(PAIR_ROLES) for donors in target_donors.values()):
        raise GateValidationError("each target must use two distinct role donors")
    if set(video_owners) != used_videos:
        raise GateValidationError(
            "component video IDs differ from candidate-pair videos"
        )
    for component_id, component in components.items():
        declared_videos = set(component["video_ids"])
        start = next(iter(declared_videos))
        reached: set[str] = set()
        pending = [start]
        while pending:
            video_id = pending.pop()
            if video_id in reached:
                continue
            reached.add(video_id)
            pending.extend(graph[video_id] - reached)
        if reached != declared_videos:
            raise GateValidationError(
                "declared components differ from recomputed donor-connected components"
            )
        if component["target_count"] != len(component_targets[component_id]):
            raise GateValidationError(
                "component target_count differs from candidate data"
            )

    role_counts = Counter(pair["role"] for pair in pairs)
    target_ids = {pair["target"]["video_id"] for pair in pairs}
    donor_ids = {pair["donor"]["video_id"] for pair in pairs}
    observed_counts = {
        "eligible_key_count": len({pair["key_sha256"] for pair in pairs}),
        "target_count": len(target_roles),
        "pair_count": len(pairs),
        "same_answer_pair_count": role_counts["same_answer_nuisance"],
        "opposite_answer_pair_count": role_counts["opposite_answer_candidate"],
        "unique_target_video_count": len(target_ids),
        "unique_donor_count": len(donor_ids),
        "component_count": len(components),
    }
    for name, observed in observed_counts.items():
        if counts[name] != observed:
            raise GateValidationError(
                f"pilot index counts.{name} differs from candidate data"
            )

    partition_counts = _mapping(
        value["partition_counts"], "pilot index partition_counts"
    )
    observed_partitions = {pair["partition"] for pair in pairs}
    if set(partition_counts) != set(PARTITIONS) or observed_partitions != set(
        PARTITIONS
    ):
        raise GateValidationError("pilot index must contain all three partitions")
    for partition, summary_value in partition_counts.items():
        summary = _mapping(summary_value, f"partition_counts.{partition}")
        _exact_keys(
            summary,
            {"component_count", "pair_count", "target_count"},
            f"partition_counts.{partition}",
        )
        partition_pairs = [pair for pair in pairs if pair["partition"] == partition]
        expected_summary = {
            "component_count": len({pair["component_id"] for pair in partition_pairs}),
            "pair_count": len(partition_pairs),
            "target_count": len(
                {
                    (pair["target"]["video_id"], pair["target"]["question_id"])
                    for pair in partition_pairs
                }
            ),
        }
        if dict(summary) != expected_summary:
            raise GateValidationError(f"partition_counts.{partition} differs")
        expected_target_count = EXPECTED_PARTITION_TARGET_COUNTS[partition]
        if summary["target_count"] != expected_target_count:
            raise GateValidationError(
                f"partition_counts.{partition}.target_count must equal "
                f"{expected_target_count}"
            )
        if summary["pair_count"] != expected_target_count * len(PAIR_ROLES):
            raise GateValidationError(
                f"partition_counts.{partition}.pair_count must equal "
                f"{expected_target_count * len(PAIR_ROLES)}"
            )
    return pairs


def parse_choice(text: str) -> str:
    """Return a strict multiple-choice label from a model response."""

    normalized = text.strip() if isinstance(text, str) else ""
    if normalized not in CHOICES:
        raise GateValidationError("response must be exactly one of A, B, or C")
    return normalized


def _prompt(question: str, options: Sequence[str]) -> str:
    lines = [
        "Answer the multiple-choice question using only the supplied media.",
        f"Question: {question}",
        "Choices:",
    ]
    lines.extend(f"{choice}. {option}" for choice, option in zip(CHOICES, options))
    lines.append("Reply with exactly one uppercase letter: A, B, or C. Do not explain.")
    return "\n".join(lines)


def _probe_media_streams(path: pathlib.Path) -> list[str]:
    """Return bounded ffprobe stream types, requiring both audio and video."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=_FFPROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GateValidationError(
            f"required media stream probe failed: {path.name}"
        ) from exc
    if completed.returncode != 0 or len(completed.stdout) > 1_000_000:
        raise GateValidationError(f"required media stream probe failed: {path.name}")
    try:
        payload = json.loads(
            completed.stdout,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise GateValidationError(
            f"required media stream probe returned invalid JSON: {path.name}"
        ) from exc
    streams = _mapping(payload, "ffprobe output").get("streams")
    if not isinstance(streams, list):
        raise GateValidationError(f"required media streams are missing: {path.name}")
    stream_types = sorted(
        {
            stream.get("codec_type")
            for stream in streams
            if isinstance(stream, dict) and isinstance(stream.get("codec_type"), str)
        }
    )
    if not {"audio", "video"} <= set(stream_types):
        raise GateValidationError(
            f"required audio and video streams are missing: {path.name}"
        )
    return stream_types


def _media_inventory(
    pairs: Sequence[Mapping[str, Any]], media_root: pathlib.Path
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    try:
        root = pathlib.Path(media_root).resolve(strict=True)
    except OSError as exc:
        raise GateValidationError("media root cannot be resolved") from exc
    if not root.is_dir():
        raise GateValidationError("media root must be a directory")
    video_ids = sorted(
        {
            pair[source_role]["video_id"]
            for pair in pairs
            for source_role in SOURCE_ROLES
        }
    )
    by_id: dict[str, dict[str, Any]] = {}
    public_records: list[dict[str, Any]] = []
    for video_id in video_ids:
        path = root / f"{video_id}.mp4"
        if path.is_symlink():
            raise GateValidationError(
                f"required media file must not be a symlink: {path.name}"
            )
        try:
            file_stat = path.stat()
        except OSError as exc:
            raise GateValidationError(
                f"required media file is missing: {path.name}"
            ) from exc
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size <= 0:
            raise GateValidationError(
                f"required media file is not a nonempty regular file: {path.name}"
            )
        resolved = path.resolve(strict=True)
        if resolved.parent != root:
            raise GateValidationError(
                f"required media file escapes the media root: {path.name}"
            )
        digest = _sha256_path(resolved)
        stream_types = _probe_media_streams(resolved)
        record = {
            "video_id": video_id,
            "filename": path.name,
            "size_bytes": file_stat.st_size,
            "sha256": digest,
            "stream_types": stream_types,
        }
        public_records.append(record)
        by_id[video_id] = {**record, "path": resolved}
    return by_id, public_records


def _verify_media_unchanged(
    media_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    """Fail if any authenticated media bytes changed while inference ran."""

    for video_id in sorted(media_by_id):
        media = media_by_id[video_id]
        path = pathlib.Path(media["path"])
        try:
            file_stat = path.stat()
        except OSError as exc:
            raise GateValidationError(
                f"media changed during inference: {path.name}"
            ) from exc
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_size != media["size_bytes"]
            or _sha256_path(path) != media["sha256"]
        ):
            raise GateValidationError(f"media changed during inference: {path.name}")


def _source_inventory(paths: Sequence[pathlib.Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    names: set[str] = set()
    for path_value in sorted(
        (pathlib.Path(path) for path in paths), key=lambda item: item.name
    ):
        path = path_value.resolve(strict=True)
        if not path.is_file() or path.name in names:
            raise GateValidationError("source paths must be unique regular files")
        names.add(path.name)
        records.append(
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_path(path),
            }
        )
    return records


def _option_balance_slots(
    pairs: Sequence[Mapping[str, Any]], seed: int
) -> dict[tuple[str, str], int]:
    """Assign near-equal A/B/C answer slots without using model outcomes."""

    slots: dict[tuple[str, str], int] = {}
    offset = seed % len(CHOICES)
    for partition in PARTITIONS:
        target_pairs: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
        for pair in pairs:
            if pair["partition"] == partition:
                source = pair["target"]
                target_pairs[(source["video_id"], source["question_id"])].append(pair)
        for rank, source_key in enumerate(sorted(target_pairs)):
            slot = (rank + offset) % len(CHOICES)
            for pair in target_pairs[source_key]:
                slots[(pair["pair_id"], "target")] = slot
        for pair_role in PAIR_ROLES:
            donor_pairs = sorted(
                (
                    pair
                    for pair in pairs
                    if pair["partition"] == partition and pair["role"] == pair_role
                ),
                key=lambda pair: (
                    pair["donor"]["video_id"],
                    pair["donor"]["question_id"],
                    pair["pair_id"],
                ),
            )
            for rank, pair in enumerate(donor_pairs):
                slots[(pair["pair_id"], "donor")] = (
                    rank + offset + PAIR_ROLES.index(pair_role)
                ) % len(CHOICES)
    if len(slots) != len(pairs) * len(SOURCE_ROLES):
        raise GateValidationError("option-balance assignments are incomplete")
    return slots


def _balanced_options(source: Mapping[str, Any], slot: int) -> list[str]:
    answer = source["answer"]
    remaining = sorted(option for option in source["options"] if option != answer)
    options = list(remaining)
    options.insert(slot, answer)
    return options


def _build_candidate_design(
    pairs: Sequence[Mapping[str, Any]],
    media_by_id: Mapping[str, Mapping[str, Any]],
    option_order_seed: int,
) -> list[dict[str, Any]]:
    """Embed the fixed balanced prompts and counteranswer media assignments."""

    slots = _option_balance_slots(pairs, option_order_seed)
    by_target: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for pair in pairs:
        target_key = (pair["target"]["video_id"], pair["target"]["question_id"])
        by_target[target_key][pair["role"]] = pair
    design: list[dict[str, Any]] = []
    for pair in sorted(pairs, key=lambda item: item["pair_id"]):
        target_key = (pair["target"]["video_id"], pair["target"]["question_id"])
        opposite_source = by_target[target_key]["opposite_answer_candidate"]["donor"]
        sources: dict[str, dict[str, Any]] = {}
        for source_role in SOURCE_ROLES:
            source = pair[source_role]
            normalized_question = normalize_text(source["question"])
            normalized_question_sha256 = hashlib.sha256(
                normalized_question.encode("utf-8")
            ).hexdigest()
            slot = slots[(pair["pair_id"], source_role)]
            options = _balanced_options(source, slot)
            prompt = _prompt(source["question"], options)
            prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if source_role == "target" or pair["role"] == "same_answer_nuisance":
                shuffled_source = opposite_source
            else:
                shuffled_source = pair["target"]
            shuffled_choice = CHOICES[options.index(shuffled_source["answer"])]
            if shuffled_choice == CHOICES[slot]:
                raise GateValidationError(
                    "same-question shuffled media must carry a counteranswer"
                )
            sources[source_role] = {
                "video_id": source["video_id"],
                "question_id": source["question_id"],
                "normalized_question_sha256": normalized_question_sha256,
                "question": source["question"],
                "expected_choice": CHOICES[slot],
                "media_sha256": media_by_id[source["video_id"]]["sha256"],
                "balanced_options": options,
                "option_balance_slot": slot,
                "prompt": prompt,
                "prompt_sha256": prompt_sha256,
                "shuffled_media_video_id": shuffled_source["video_id"],
                "shuffled_media_sha256": media_by_id[shuffled_source["video_id"]][
                    "sha256"
                ],
                "shuffled_media_choice": shuffled_choice,
            }
        design.append(
            {
                "pair_id": pair["pair_id"],
                "pair_role": pair["role"],
                "component_id": pair["component_id"],
                "partition": pair["partition"],
                "key_sha256": pair["key_sha256"],
                "prompt": sources["target"]["prompt"],
                "prompt_sha256": sources["target"]["prompt_sha256"],
                "target": sources["target"],
                "donor": sources["donor"],
            }
        )
    return design


def _build_tasks(
    *,
    design: Sequence[Mapping[str, Any]],
    media_by_id: Mapping[str, Mapping[str, Any]],
    config_sha256: str,
    pilot_sha256: str,
) -> list[_Task]:
    tasks: list[_Task] = []

    def append_task(
        pair: Mapping[str, Any],
        source_role: str,
        evaluation: str,
        condition: str,
    ) -> None:
        source = pair[source_role]
        if evaluation == "question_only":
            media_video_id = None
            media_path = None
            media_sha256 = None
            shuffled_media_choice = None
        elif evaluation == "source_sufficiency":
            media_video_id = source["video_id"]
            media_path = pathlib.Path(media_by_id[media_video_id]["path"])
            media_sha256 = source["media_sha256"]
            shuffled_media_choice = None
        else:
            media_video_id = source["shuffled_media_video_id"]
            media_path = pathlib.Path(media_by_id[media_video_id]["path"])
            media_sha256 = source["shuffled_media_sha256"]
            shuffled_media_choice = source["shuffled_media_choice"]
        identity = {
            "pair_id": pair["pair_id"],
            "pair_role": pair["pair_role"],
            "evaluation": evaluation,
            "source_role": source_role,
            "condition": condition,
        }
        record_id = _digest_value(identity)[:24]
        request_sha256 = _digest_value(
            {
                **identity,
                "record_id": record_id,
                "component_id": pair["component_id"],
                "partition": pair["partition"],
                "video_id": source["video_id"],
                "media_video_id": media_video_id,
                "question_id": source["question_id"],
                "normalized_question_sha256": source["normalized_question_sha256"],
                "expected_choice": source["expected_choice"],
                "media_sha256": media_sha256,
                "shuffled_media_choice": shuffled_media_choice,
                "prompt_sha256": source["prompt_sha256"],
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "configuration_sha256": config_sha256,
                "pilot_index_sha256": pilot_sha256,
            }
        )
        tasks.append(
            _Task(
                request=InferenceRequest(
                    record_id=record_id,
                    pair_id=pair["pair_id"],
                    pair_role=pair["pair_role"],
                    source_role=source_role,
                    condition=condition,
                    video_id=source["video_id"],
                    media_path=media_path,
                    prompt=source["prompt"],
                    evaluation=evaluation,
                    media_video_id=media_video_id,
                ),
                component_id=pair["component_id"],
                partition=pair["partition"],
                pair_role=pair["pair_role"],
                question_id=source["question_id"],
                normalized_question_sha256=source["normalized_question_sha256"],
                expected_choice=source["expected_choice"],
                media_video_id=media_video_id,
                media_sha256=media_sha256,
                shuffled_media_choice=shuffled_media_choice,
                prompt_sha256=source["prompt_sha256"],
                request_sha256=request_sha256,
            )
        )

    ordered_design = sorted(design, key=lambda item: item["pair_id"])
    for condition in CONDITIONS:
        for pair in ordered_design:
            for source_role in SOURCE_ROLES:
                append_task(pair, source_role, "source_sufficiency", condition)
    for pair in ordered_design:
        for source_role in SOURCE_ROLES:
            append_task(pair, source_role, "question_only", "question_only")
    for condition in CONDITIONS:
        for pair in ordered_design:
            for source_role in SOURCE_ROLES:
                append_task(
                    pair,
                    source_role,
                    "same_question_shuffled_media",
                    condition,
                )
    record_ids = [task.request.record_id for task in tasks]
    if len(record_ids) != len(set(record_ids)):
        raise GateValidationError("constructed record IDs are not unique")
    return tasks


def _validate_backend_results(
    results: Sequence[InferenceResult], expected_ids: Sequence[str]
) -> dict[str, InferenceResult]:
    result_by_id: dict[str, InferenceResult] = {}
    for result in results:
        if not isinstance(result, InferenceResult):
            raise GateValidationError("backend returned an invalid result type")
        if result.record_id in result_by_id:
            raise GateValidationError("backend returned duplicate result IDs")
        if not isinstance(result.raw_response, str) or len(result.raw_response) > 256:
            raise GateValidationError("backend raw response must be a bounded string")
        result_by_id[result.record_id] = result
    if set(result_by_id) != set(expected_ids):
        raise GateValidationError("backend result IDs differ from the requested batch")
    return result_by_id


def _choice_score_record(result: InferenceResult) -> dict[str, Any]:
    if result.choice_scores is None:
        if (
            result.choice_score_method is not None
            or result.choice_token_ids is not None
        ):
            raise GateValidationError(
                "unavailable choice scores must not have score metadata"
            )
        return {
            "status": "unavailable",
            "method": None,
            "values": None,
            "token_ids": None,
            "used_for_gate": False,
        }
    scores = _mapping(result.choice_scores, "backend choice_scores")
    token_ids = _mapping(result.choice_token_ids, "backend choice_token_ids")
    if set(scores) != set(CHOICES) or set(token_ids) != set(CHOICES):
        raise GateValidationError("backend choice scores must use exactly A, B, and C")
    values: dict[str, float] = {}
    normalized_ids: dict[str, int] = {}
    for choice in CHOICES:
        score = scores[choice]
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise GateValidationError("backend choice scores must be finite numbers")
        values[choice] = float(score)
        normalized_ids[choice] = _nonnegative_int(
            token_ids[choice], f"token ID for {choice}"
        )
    method = _plain_string(result.choice_score_method, "backend choice score method")
    if len(set(normalized_ids.values())) != len(CHOICES):
        raise GateValidationError("backend choice token IDs must be distinct")
    return {
        "status": "available",
        "method": method,
        "values": values,
        "token_ids": normalized_ids,
        "used_for_gate": False,
    }


def _records_from_backend(
    tasks: Sequence[_Task], backend: InferenceBackend, batch_size: int
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    groups = [
        *(("source_sufficiency", condition) for condition in CONDITIONS),
        ("question_only", "question_only"),
        *(("same_question_shuffled_media", condition) for condition in CONDITIONS),
    ]
    for evaluation, condition in groups:
        grouped_tasks = [
            task
            for task in tasks
            if task.request.evaluation == evaluation
            and task.request.condition == condition
        ]
        for start in range(0, len(grouped_tasks), batch_size):
            batch = grouped_tasks[start : start + batch_size]
            requests = [task.request for task in batch]
            results = _validate_backend_results(
                list(backend.infer(requests)),
                [request.record_id for request in requests],
            )
            for task in batch:
                result = results[task.request.record_id]
                try:
                    predicted_choice = parse_choice(result.raw_response)
                    parse_status = "strict"
                except GateValidationError:
                    predicted_choice = None
                    parse_status = "invalid"
                records.append(
                    {
                        "record_id": task.request.record_id,
                        "request_sha256": task.request_sha256,
                        "pair_id": task.request.pair_id,
                        "component_id": task.component_id,
                        "partition": task.partition,
                        "pair_role": task.pair_role,
                        "evaluation": task.request.evaluation,
                        "source_role": task.request.source_role,
                        "condition": task.request.condition,
                        "video_id": task.request.video_id,
                        "media_video_id": task.media_video_id,
                        "question_id": task.question_id,
                        "normalized_question_sha256": (task.normalized_question_sha256),
                        "expected_choice": task.expected_choice,
                        "raw_response": result.raw_response,
                        "predicted_choice": predicted_choice,
                        "parse_status": parse_status,
                        "is_correct": predicted_choice == task.expected_choice,
                        "media_sha256": task.media_sha256,
                        "shuffled_media_choice": task.shuffled_media_choice,
                        "prompt_sha256": task.prompt_sha256,
                        "choice_scores": _choice_score_record(result),
                    }
                )
    return records


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        raise GateValidationError("gate metric has an empty denominator")
    return numerator / denominator


def _wilson_interval_95(successes: int, count: int) -> list[float]:
    if count <= 0 or not 0 <= successes <= count:
        raise GateValidationError("Wilson interval inputs are invalid")
    z = statistics.NormalDist().inv_cdf(0.975)
    rate = successes / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _summarize_gate(
    records: Sequence[Mapping[str, Any]], gate_config: Mapping[str, Any]
) -> dict[str, Any]:
    evaluation_partition = gate_config["evaluation_partition"]
    partition_records = [
        record for record in records if record["partition"] == evaluation_partition
    ]
    selected = [
        record
        for record in partition_records
        if record["evaluation"] == "source_sufficiency"
    ]
    if not selected:
        raise GateValidationError("evaluation partition has no model records")
    failed: list[str] = []
    metrics: dict[str, Any] = {}
    parse_threshold = float(gate_config["minimum_strict_parse_rate"])
    target_thresholds = gate_config["minimum_unique_target_accuracy_by_condition"]
    donor_thresholds = gate_config["minimum_donor_accuracy_by_role_and_condition"]
    pair_thresholds = gate_config["minimum_pair_support_rate_by_role_and_condition"]
    require_scores = bool(gate_config["require_choice_scores"])
    for condition in CONDITIONS:
        condition_records = [
            record for record in selected if record["condition"] == condition
        ]
        if not condition_records:
            raise GateValidationError(f"evaluation partition lacks {condition} records")
        target_groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = (
            defaultdict(list)
        )
        for record in condition_records:
            if record["source_role"] == "target":
                target_groups[
                    (
                        record["component_id"],
                        record["video_id"],
                        record["question_id"],
                    )
                ].append(record)
        unique_targets: list[Mapping[str, Any]] = []
        duplicate_comparison_fields = (
            "expected_choice",
            "raw_response",
            "predicted_choice",
            "parse_status",
            "is_correct",
            "media_sha256",
            "prompt_sha256",
            "choice_scores",
        )
        for target_records in target_groups.values():
            if len(target_records) != len(PAIR_ROLES) or {
                record["pair_role"] for record in target_records
            } != set(PAIR_ROLES):
                raise GateValidationError(
                    "target-condition records do not cover both pair roles exactly once"
                )
            reference = target_records[0]
            if any(
                any(
                    record[field] != reference[field]
                    for field in duplicate_comparison_fields
                )
                for record in target_records[1:]
            ):
                raise GateValidationError(
                    "duplicate target evaluations disagree across pair roles"
                )
            unique_targets.append(reference)
        unique_target_count = len(unique_targets)
        unique_target_correct_count = sum(
            bool(record["is_correct"]) for record in unique_targets
        )
        unique_target_accuracy = _safe_rate(
            unique_target_correct_count, unique_target_count
        )

        role_metrics: dict[str, Any] = {}
        for pair_role in PAIR_ROLES:
            pair_role_records = [
                record
                for record in condition_records
                if record["pair_role"] == pair_role
            ]
            strict_count = sum(
                record["parse_status"] == "strict" for record in pair_role_records
            )
            scored_count = sum(
                record["choice_scores"]["status"] == "available"
                for record in pair_role_records
            )
            strict_rate = _safe_rate(strict_count, len(pair_role_records))
            score_rate = _safe_rate(scored_count, len(pair_role_records))
            source_accuracy: dict[str, float] = {}
            source_counts: dict[str, int] = {}
            source_correct_counts: dict[str, int] = {}
            for source_role in SOURCE_ROLES:
                source_records = [
                    record
                    for record in pair_role_records
                    if record["source_role"] == source_role
                ]
                source_counts[source_role] = len(source_records)
                source_correct_counts[source_role] = sum(
                    bool(record["is_correct"]) for record in source_records
                )
                source_accuracy[source_role] = _safe_rate(
                    source_correct_counts[source_role],
                    len(source_records),
                )
            by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for record in pair_role_records:
                by_pair[record["pair_id"]].append(record)
            for pair_records in by_pair.values():
                if {record["source_role"] for record in pair_records} != set(
                    SOURCE_ROLES
                ) or len(pair_records) != 2:
                    raise GateValidationError("pair-condition records are incomplete")
            pair_support_count = sum(
                all(bool(record["is_correct"]) for record in pair_records)
                for pair_records in by_pair.values()
            )
            pair_support_rate = _safe_rate(pair_support_count, len(by_pair))
            role_metrics[pair_role] = {
                "record_count": len(pair_role_records),
                "strict_parse_count": strict_count,
                "strict_parse_rate": strict_rate,
                "choice_score_available_count": scored_count,
                "choice_score_available_rate": score_rate,
                "source_counts": source_counts,
                "source_correct_counts": source_correct_counts,
                "source_accuracy": source_accuracy,
                "pair_count": len(by_pair),
                "pair_support_count": pair_support_count,
                "pair_support_rate": pair_support_rate,
            }
            if strict_rate < parse_threshold:
                failed.append(f"{condition}.{pair_role}.strict_parse_rate")
            if source_accuracy["donor"] < float(donor_thresholds[pair_role][condition]):
                failed.append(f"{condition}.{pair_role}.donor_accuracy")
            if pair_support_rate < float(pair_thresholds[pair_role][condition]):
                failed.append(f"{condition}.{pair_role}.pair_support_rate")
            if require_scores and score_rate < 1.0:
                failed.append(f"{condition}.{pair_role}.choice_score_availability")
        metrics[condition] = {
            "record_count": len(condition_records),
            "unique_target_count": unique_target_count,
            "unique_target_correct_count": unique_target_correct_count,
            "unique_target_accuracy": unique_target_accuracy,
            "by_pair_role": role_metrics,
        }
        if unique_target_accuracy < float(target_thresholds[condition]):
            failed.append(f"{condition}.unique_target_accuracy")

    def component_summary(
        condition: str, source_role: str, pair_role: str | None = None
    ) -> dict[str, Any]:
        cell_records = [
            record
            for record in selected
            if record["condition"] == condition
            and record["source_role"] == source_role
            and (pair_role is None or record["pair_role"] == pair_role)
        ]
        by_component: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in cell_records:
            by_component[record["component_id"]].append(record)
        exact_outcomes = [
            {
                "component_id": component_id,
                "aligned_record_ids": sorted(
                    record["record_id"] for record in component_records
                ),
                "success": all(
                    bool(record["is_correct"]) for record in component_records
                ),
            }
            for component_id, component_records in sorted(by_component.items())
        ]
        success_count = sum(item["success"] for item in exact_outcomes)
        component_count = len(exact_outcomes)
        return {
            "component_count": component_count,
            "component_success_count": success_count,
            "component_failure_count": component_count - success_count,
            "component_success_rate": _safe_rate(success_count, component_count),
            "exact_component_outcomes": exact_outcomes,
            "wilson_interval_95": _wilson_interval_95(success_count, component_count),
            "interval_method": "wilson_score",
            "interval_confidence_level": 0.95,
            "interpretation": "descriptive_pilot_feasibility_screen",
        }

    component_source_outcomes: dict[str, Any] = {}
    for condition in CONDITIONS:
        component_source_outcomes[condition] = {
            "target": component_summary(condition, "target"),
            "donor": {
                "by_pair_role": {
                    pair_role: component_summary(condition, "donor", pair_role)
                    for pair_role in PAIR_ROLES
                }
            },
        }

    def unique_orientation_records(
        evaluation: str,
        condition: str,
        source_role: str,
        pair_role: str | None = None,
        record_scope: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[Mapping[str, Any]]:
        scope = partition_records if record_scope is None else record_scope
        matching = [
            record
            for record in scope
            if record["evaluation"] == evaluation
            and record["condition"] == condition
            and record["source_role"] == source_role
            and (pair_role is None or record["pair_role"] == pair_role)
        ]
        grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
        for record in matching:
            key = (
                record["component_id"],
                record["video_id"],
                record["question_id"],
            )
            if source_role == "donor":
                key = (record["pair_id"], *key)
            grouped[key].append(record)
        unique: list[Mapping[str, Any]] = []
        comparison_fields = (
            "expected_choice",
            "raw_response",
            "predicted_choice",
            "parse_status",
            "is_correct",
            "media_video_id",
            "media_sha256",
            "shuffled_media_choice",
            "prompt_sha256",
            "choice_scores",
        )
        for grouped_records in grouped.values():
            reference = grouped_records[0]
            if source_role == "target" and (
                len(grouped_records) != len(PAIR_ROLES)
                or {record["pair_role"] for record in grouped_records}
                != set(PAIR_ROLES)
            ):
                raise GateValidationError(
                    "target control records do not cover both pair roles"
                )
            if any(
                any(record[field] != reference[field] for field in comparison_fields)
                for record in grouped_records[1:]
            ):
                raise GateValidationError(
                    "duplicate target control evaluations disagree across pair roles"
                )
            unique.append(reference)
        if not unique:
            raise GateValidationError("control orientation has no model records")
        return unique

    def control_summary(
        evaluation: str,
        condition: str,
        source_role: str,
        pair_role: str | None = None,
        record_scope: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], int, int]:
        orientation_records = unique_orientation_records(
            evaluation, condition, source_role, pair_role, record_scope
        )
        strict_count = sum(
            record["parse_status"] == "strict" for record in orientation_records
        )
        agreement_count = sum(
            bool(record["is_correct"]) for record in orientation_records
        )
        summary = {
            "unique_record_count": len(orientation_records),
            "strict_parse_count": strict_count,
            "strict_parse_rate": _safe_rate(strict_count, len(orientation_records)),
            "source_answer_agreement_count": agreement_count,
            "source_answer_agreement_rate": _safe_rate(
                agreement_count, len(orientation_records)
            ),
        }
        if evaluation == "same_question_shuffled_media":
            shuffled_agreement_count = sum(
                record["predicted_choice"] == record["shuffled_media_choice"]
                for record in orientation_records
            )
            summary.update(
                {
                    "shuffled_media_answer_agreement_count": (shuffled_agreement_count),
                    "shuffled_media_answer_agreement_rate": _safe_rate(
                        shuffled_agreement_count, len(orientation_records)
                    ),
                }
            )
        return summary, agreement_count, len(orientation_records)

    def passes_margin(
        aligned_count: int,
        aligned_total: int,
        control_count: int,
        control_total: int,
        minimum: float,
    ) -> bool:
        margin = Fraction(aligned_count, aligned_total) - Fraction(
            control_count, control_total
        )
        return margin >= Fraction(str(minimum))

    question_only_maximums = gate_config[
        "maximum_question_only_source_answer_accuracy_by_orientation"
    ]
    shuffled_maximums = gate_config[
        "maximum_shuffled_media_source_answer_accuracy_by_condition_and_orientation"
    ]
    margin_minimums = gate_config[
        "minimum_aligned_over_control_margin_by_condition_and_orientation"
    ]
    control_metrics: dict[str, Any] = {
        "question_only": {"by_source_orientation": {}},
        "same_question_shuffled_media": {},
        "aligned_over_control_margin": {},
    }
    question_only_counts: dict[tuple[str, str | None], tuple[int, int]] = {}
    target_question, target_question_correct, target_question_count = control_summary(
        "question_only", "question_only", "target"
    )
    question_only_counts[("target", None)] = (
        target_question_correct,
        target_question_count,
    )
    control_metrics["question_only"]["by_source_orientation"]["target"] = (
        target_question
    )
    if target_question["strict_parse_rate"] < parse_threshold:
        failed.append("question_only.target.strict_parse_rate")
    if target_question["source_answer_agreement_rate"] > float(
        question_only_maximums["target"]
    ):
        failed.append("question_only.target.source_answer_agreement_rate")

    donor_question_by_role: dict[str, Any] = {}
    for pair_role in PAIR_ROLES:
        summary, agreement_count, record_count = control_summary(
            "question_only", "question_only", "donor", pair_role
        )
        question_only_counts[("donor", pair_role)] = (
            agreement_count,
            record_count,
        )
        donor_question_by_role[pair_role] = summary
        if summary["strict_parse_rate"] < parse_threshold:
            failed.append(f"question_only.donor.{pair_role}.strict_parse_rate")
        if summary["source_answer_agreement_rate"] > float(
            question_only_maximums["donor"]
        ):
            failed.append(
                f"question_only.donor.{pair_role}.source_answer_agreement_rate"
            )
    control_metrics["question_only"]["by_source_orientation"]["donor"] = {
        "by_pair_role": donor_question_by_role
    }

    for condition in CONDITIONS:
        control_metrics["same_question_shuffled_media"][condition] = {
            "by_source_orientation": {}
        }
        control_metrics["aligned_over_control_margin"][condition] = {}
        target_shuffled, target_shuffled_correct, target_shuffled_count = (
            control_summary("same_question_shuffled_media", condition, "target")
        )
        control_metrics["same_question_shuffled_media"][condition][
            "by_source_orientation"
        ]["target"] = target_shuffled
        target_aligned_correct = metrics[condition]["unique_target_correct_count"]
        target_aligned_count = metrics[condition]["unique_target_count"]
        target_question_correct, target_question_total = question_only_counts[
            ("target", None)
        ]
        target_control = max(
            (target_question_correct, target_question_total, "question_only"),
            (
                target_shuffled_correct,
                target_shuffled_count,
                "same_question_shuffled_media",
            ),
            key=lambda item: Fraction(item[0], item[1]),
        )
        target_margin = float(
            Fraction(target_aligned_correct, target_aligned_count)
            - Fraction(target_control[0], target_control[1])
        )
        target_margin_passes = passes_margin(
            target_aligned_correct,
            target_aligned_count,
            target_control[0],
            target_control[1],
            float(margin_minimums["target"][condition]),
        )
        control_metrics["aligned_over_control_margin"][condition]["target"] = {
            "aligned_source_correct_count": target_aligned_correct,
            "aligned_source_count": target_aligned_count,
            "aligned_source_accuracy": _safe_rate(
                target_aligned_correct, target_aligned_count
            ),
            "strongest_control_type": target_control[2],
            "strongest_control_source_answer_agreement_count": target_control[0],
            "strongest_control_count": target_control[1],
            "strongest_control_source_answer_agreement_rate": _safe_rate(
                target_control[0], target_control[1]
            ),
            "margin": target_margin,
            "passes_minimum_margin": target_margin_passes,
        }
        if target_shuffled["strict_parse_rate"] < parse_threshold:
            failed.append(
                f"same_question_shuffled_media.{condition}.target.strict_parse_rate"
            )
        if target_shuffled["source_answer_agreement_rate"] > float(
            shuffled_maximums["target"][condition]
        ):
            failed.append(
                "same_question_shuffled_media."
                f"{condition}.target.source_answer_agreement_rate"
            )
        if not target_margin_passes:
            failed.append(f"aligned_over_control_margin.{condition}.target")

        donor_shuffled_by_role: dict[str, Any] = {}
        donor_margin_by_role: dict[str, Any] = {}
        for pair_role in PAIR_ROLES:
            summary, shuffled_correct, shuffled_count = control_summary(
                "same_question_shuffled_media", condition, "donor", pair_role
            )
            donor_shuffled_by_role[pair_role] = summary
            question_correct, question_count = question_only_counts[
                ("donor", pair_role)
            ]
            strongest_control = max(
                (question_correct, question_count, "question_only"),
                (
                    shuffled_correct,
                    shuffled_count,
                    "same_question_shuffled_media",
                ),
                key=lambda item: Fraction(item[0], item[1]),
            )
            role_metrics = metrics[condition]["by_pair_role"][pair_role]
            aligned_correct = role_metrics["source_correct_counts"]["donor"]
            aligned_count = role_metrics["source_counts"]["donor"]
            margin = float(
                Fraction(aligned_correct, aligned_count)
                - Fraction(strongest_control[0], strongest_control[1])
            )
            margin_passes = passes_margin(
                aligned_correct,
                aligned_count,
                strongest_control[0],
                strongest_control[1],
                float(margin_minimums["donor"][condition]),
            )
            donor_margin_by_role[pair_role] = {
                "aligned_source_correct_count": aligned_correct,
                "aligned_source_count": aligned_count,
                "aligned_source_accuracy": _safe_rate(aligned_correct, aligned_count),
                "strongest_control_type": strongest_control[2],
                "strongest_control_source_answer_agreement_count": (
                    strongest_control[0]
                ),
                "strongest_control_count": strongest_control[1],
                "strongest_control_source_answer_agreement_rate": _safe_rate(
                    strongest_control[0], strongest_control[1]
                ),
                "margin": margin,
                "passes_minimum_margin": margin_passes,
            }
            if summary["strict_parse_rate"] < parse_threshold:
                failed.append(
                    "same_question_shuffled_media."
                    f"{condition}.donor.{pair_role}.strict_parse_rate"
                )
            if summary["source_answer_agreement_rate"] > float(
                shuffled_maximums["donor"][condition]
            ):
                failed.append(
                    "same_question_shuffled_media."
                    f"{condition}.donor.{pair_role}.source_answer_agreement_rate"
                )
            if not margin_passes:
                failed.append(
                    f"aligned_over_control_margin.{condition}.donor.{pair_role}"
                )
        control_metrics["same_question_shuffled_media"][condition][
            "by_source_orientation"
        ]["donor"] = {"by_pair_role": donor_shuffled_by_role}
        control_metrics["aligned_over_control_margin"][condition]["donor"] = {
            "by_pair_role": donor_margin_by_role
        }

    seen_question_keys = sorted(
        {
            record["normalized_question_sha256"]
            for record in records
            if record["partition"] in {"scorer_fit", "threshold_calibration"}
        }
    )
    held_out_question_keys = sorted(
        {
            record["normalized_question_sha256"]
            for record in partition_records
            if record["normalized_question_sha256"] not in seen_question_keys
        }
    )
    if set(seen_question_keys) & set(held_out_question_keys):
        raise GateValidationError("question-key sensitivity split is not disjoint")
    held_out_records = [
        record
        for record in partition_records
        if record["normalized_question_sha256"] in held_out_question_keys
    ]
    held_out_components = sorted(
        {record["component_id"] for record in held_out_records}
    )
    minimum_unseen_components = int(
        gate_config["minimum_unseen_question_component_count"]
    )
    maximum_attenuation = float(
        gate_config["maximum_unseen_question_margin_attenuation"]
    )
    sensitivity_failures: list[str] = []

    def sensitivity_cell(
        condition: str, source_role: str, pair_role: str | None = None
    ) -> dict[str, Any]:
        aligned_records = unique_orientation_records(
            "source_sufficiency",
            condition,
            source_role,
            pair_role,
            held_out_records,
        )
        question_summary, question_correct, question_count = control_summary(
            "question_only",
            "question_only",
            source_role,
            pair_role,
            held_out_records,
        )
        shuffled_summary, shuffled_correct, shuffled_count = control_summary(
            "same_question_shuffled_media",
            condition,
            source_role,
            pair_role,
            held_out_records,
        )
        aligned_correct = sum(bool(record["is_correct"]) for record in aligned_records)
        aligned_count = len(aligned_records)
        strongest_control = max(
            (question_correct, question_count, "question_only"),
            (
                shuffled_correct,
                shuffled_count,
                "same_question_shuffled_media",
            ),
            key=lambda item: Fraction(item[0], item[1]),
        )
        held_out_margin_fraction = Fraction(aligned_correct, aligned_count) - Fraction(
            strongest_control[0], strongest_control[1]
        )
        if source_role == "target":
            primary_cell = control_metrics["aligned_over_control_margin"][condition][
                "target"
            ]
        else:
            if pair_role is None:
                raise GateValidationError("donor sensitivity requires a pair role")
            primary_cell = control_metrics["aligned_over_control_margin"][condition][
                "donor"
            ]["by_pair_role"][pair_role]
        primary_margin_fraction = Fraction(
            primary_cell["aligned_source_correct_count"],
            primary_cell["aligned_source_count"],
        ) - Fraction(
            primary_cell["strongest_control_source_answer_agreement_count"],
            primary_cell["strongest_control_count"],
        )
        attenuation_fraction = primary_margin_fraction - held_out_margin_fraction
        attenuation_passes = attenuation_fraction <= Fraction(str(maximum_attenuation))
        return {
            "aligned_source_correct_count": aligned_correct,
            "aligned_source_count": aligned_count,
            "aligned_source_accuracy": _safe_rate(aligned_correct, aligned_count),
            "question_only_source_answer_agreement_count": question_correct,
            "question_only_count": question_count,
            "question_only_source_answer_agreement_rate": question_summary[
                "source_answer_agreement_rate"
            ],
            "shuffled_media_source_answer_agreement_count": shuffled_correct,
            "shuffled_media_count": shuffled_count,
            "shuffled_media_source_answer_agreement_rate": shuffled_summary[
                "source_answer_agreement_rate"
            ],
            "strongest_control_type": strongest_control[2],
            "strongest_control_source_answer_agreement_rate": _safe_rate(
                strongest_control[0], strongest_control[1]
            ),
            "aligned_over_control_margin": float(held_out_margin_fraction),
            "primary_aligned_over_control_margin": float(primary_margin_fraction),
            "margin_attenuation": float(attenuation_fraction),
            "passes_maximum_margin_attenuation": attenuation_passes,
        }

    question_key_sensitivity: dict[str, Any] = {
        "status": "insufficient_components",
        "split_policy": QUESTION_KEY_SPLIT_POLICY,
        "normalization": "NFC_and_collapsed_whitespace",
        "key_disjoint": True,
        "fit_or_threshold_normalized_question_key_sha256": seen_question_keys,
        "held_out_normalized_question_key_sha256": held_out_question_keys,
        "normalized_question_key_count": len(held_out_question_keys),
        "component_count": len(held_out_components),
        "minimum_component_count": minimum_unseen_components,
        "maximum_margin_attenuation": maximum_attenuation,
        "metrics": {},
    }
    if len(held_out_components) < minimum_unseen_components:
        sensitivity_failures.append("question_key_sensitivity.insufficient_components")
    if held_out_components:
        sensitivity_metrics: dict[str, Any] = {}
        for condition in CONDITIONS:
            target_cell = sensitivity_cell(condition, "target")
            donor_cells = {
                pair_role: sensitivity_cell(condition, "donor", pair_role)
                for pair_role in PAIR_ROLES
            }
            sensitivity_metrics[condition] = {
                "target": target_cell,
                "donor": {"by_pair_role": donor_cells},
            }
            if not target_cell["passes_maximum_margin_attenuation"]:
                sensitivity_failures.append(
                    f"question_key_sensitivity.{condition}.target.margin_attenuation"
                )
            for pair_role, donor_cell in donor_cells.items():
                if not donor_cell["passes_maximum_margin_attenuation"]:
                    sensitivity_failures.append(
                        "question_key_sensitivity."
                        f"{condition}.donor.{pair_role}.margin_attenuation"
                    )
        question_key_sensitivity["metrics"] = sensitivity_metrics
        if len(held_out_components) >= minimum_unseen_components:
            question_key_sensitivity["status"] = (
                "fail" if sensitivity_failures else "pass"
            )
    failed.extend(sensitivity_failures)
    source_status = "pass" if not failed else "fail"
    return {
        "source_sufficiency_status": source_status,
        "nuisance_detection_status": "not_run",
        "overall_pilot_status": (
            "pending_nuisance_detection" if source_status == "pass" else "fail"
        ),
        "evaluation_partition": evaluation_partition,
        "selection_policy": "entire_index_no_outcome_filtering",
        "thresholds_frozen_by_configuration_sha256": True,
        "thresholds": {
            "minimum_strict_parse_rate": parse_threshold,
            "minimum_unique_target_accuracy_by_condition": dict(target_thresholds),
            "minimum_donor_accuracy_by_role_and_condition": {
                role: dict(donor_thresholds[role]) for role in PAIR_ROLES
            },
            "minimum_pair_support_rate_by_role_and_condition": {
                role: dict(pair_thresholds[role]) for role in PAIR_ROLES
            },
            "maximum_question_only_source_answer_accuracy_by_orientation": dict(
                question_only_maximums
            ),
            "maximum_shuffled_media_source_answer_accuracy_by_condition_and_orientation": {
                source_role: dict(shuffled_maximums[source_role])
                for source_role in SOURCE_ROLES
            },
            "minimum_aligned_over_control_margin_by_condition_and_orientation": {
                source_role: dict(margin_minimums[source_role])
                for source_role in SOURCE_ROLES
            },
            "question_key_sensitivity": gate_config["question_key_sensitivity"],
            "minimum_unseen_question_component_count": minimum_unseen_components,
            "maximum_unseen_question_margin_attenuation": maximum_attenuation,
            "require_choice_scores": require_scores,
        },
        "evaluated_record_count": len(selected),
        "evaluated_control_record_count": len(partition_records) - len(selected),
        "evaluated_pair_count": len({record["pair_id"] for record in selected}),
        "evaluated_unique_target_count": len(
            {
                (record["component_id"], record["video_id"], record["question_id"])
                for record in selected
                if record["source_role"] == "target"
            }
        ),
        "metrics": metrics,
        "component_source_outcomes": component_source_outcomes,
        "control_metrics": control_metrics,
        "question_key_sensitivity": question_key_sensitivity,
        "failed_requirements": failed,
    }


def _public_inference_config(config: Mapping[str, Any]) -> dict[str, Any]:
    inference = config["inference"]
    return {
        "batch_size": inference["batch_size"],
        "maximum_new_tokens": inference["maximum_new_tokens"],
        "do_sample": False,
        "seed": inference["seed"],
        "conditions": list(CONDITIONS),
        "emit_choice_scores_when_single_token": inference[
            "emit_choice_scores_when_single_token"
        ],
        "video_preprocessing": dict(inference["video_preprocessing"]),
    }


def _inference_transcript_from_output(output: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": INFERENCE_TRANSCRIPT_SCHEMA,
        "model": dict(output["model"]),
        "input_digests": dict(output["input_digests"]),
        "runtime": dict(output["runtime"]),
        "results": [
            {
                "record_id": record["record_id"],
                "request_sha256": record["request_sha256"],
                "raw_response": record["raw_response"],
                "choice_scores": dict(record["choice_scores"]),
            }
            for record in output["records"]
        ],
    }


def build_inference_transcript(output: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact independently anchorable model-inference transcript."""

    validate_gate_output(output)
    return _inference_transcript_from_output(output)


def run_source_sufficiency_gate(
    *,
    config_path: pathlib.Path,
    expected_config_sha256: str,
    pilot_index_path: pathlib.Path,
    expected_pilot_index_sha256: str,
    media_root: pathlib.Path,
    backend: InferenceBackend,
    source_paths: Sequence[pathlib.Path],
) -> dict[str, Any]:
    """Authenticate inputs, run every source-condition request, and evaluate."""

    config_bytes, raw_config = _locked_json(
        pathlib.Path(config_path), expected_config_sha256, "configuration"
    )
    config = _validate_config(raw_config)
    pilot_bytes, raw_pilot = _locked_json(
        pathlib.Path(pilot_index_path), expected_pilot_index_sha256, "pilot index"
    )
    pairs = _validate_pilot(raw_pilot)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    config_value_sha256 = _digest_value(config)
    pilot_sha256 = hashlib.sha256(pilot_bytes).hexdigest()
    media_by_id, media_files = _media_inventory(pairs, pathlib.Path(media_root))
    source_files = _source_inventory(source_paths)
    candidate_design = _build_candidate_design(
        pairs,
        media_by_id,
        config["controls"]["option_order_seed"],
    )
    tasks = _build_tasks(
        design=candidate_design,
        media_by_id=media_by_id,
        config_sha256=config_sha256,
        pilot_sha256=pilot_sha256,
    )
    records = _records_from_backend(tasks, backend, config["inference"]["batch_size"])
    _verify_media_unchanged(media_by_id)
    source_record_count = len(pairs) * len(SOURCE_ROLES) * len(CONDITIONS)
    control_record_count = len(pairs) * len(SOURCE_ROLES) * (1 + len(CONDITIONS))
    if len(records) != source_record_count + control_record_count:
        raise GateValidationError("record count differs from the complete index design")
    gate = _summarize_gate(records, config["gate"])
    source_set_sha256 = _digest_value(source_files)
    media_set_sha256 = _digest_value(media_files)
    candidate_design_sha256 = _digest_value(candidate_design)
    run_input_sha256 = _digest_value(
        {
            "configuration_sha256": config_sha256,
            "configuration_value_sha256": config_value_sha256,
            "pilot_index_sha256": pilot_sha256,
            "source_set_sha256": source_set_sha256,
            "media_set_sha256": media_set_sha256,
            "candidate_design_sha256": candidate_design_sha256,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
    )
    output: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "frozen": True,
            "talker_disabled": True,
        },
        "configuration": config,
        "inference": _public_inference_config(config),
        "input_digests": {
            "configuration_sha256": config_sha256,
            "configuration_value_sha256": config_value_sha256,
            "pilot_index_sha256": pilot_sha256,
            "source_set_sha256": source_set_sha256,
            "media_set_sha256": media_set_sha256,
            "candidate_design_sha256": candidate_design_sha256,
            "run_input_sha256": run_input_sha256,
        },
        "source_files": source_files,
        "media_files": media_files,
        "runtime": dict(backend.runtime_details()),
        "counts": {
            "condition_count": len(CONDITIONS),
            "control_type_count": len(CONTROL_TYPES),
            "media_file_count": len(media_files),
            "target_count": EXPECTED_TARGET_COUNT,
            "pair_count": len(pairs),
            "pair_role_counts": dict(
                sorted(Counter(pair["role"] for pair in pairs).items())
            ),
            "partition_target_counts": {
                partition: len(
                    {
                        (pair["target"]["video_id"], pair["target"]["question_id"])
                        for pair in pairs
                        if pair["partition"] == partition
                    }
                )
                for partition in PARTITIONS
            },
            "source_record_count": source_record_count,
            "control_record_count": control_record_count,
            "record_count": len(records),
        },
        "gate": gate,
        "records": records,
        "candidate_design": candidate_design,
        "source_attribution": dict(raw_pilot["source_attribution"]),
        "change_note": raw_pilot["change_note"],
    }
    output["inference_transcript_sha256"] = _digest_value(
        _inference_transcript_from_output(output)
    )
    output["payload_sha256"] = _digest_value(output)
    validate_gate_output(
        output,
        expected_config_sha256=config_sha256,
        expected_pilot_index_sha256=pilot_sha256,
    )
    return output


def load_gate_configuration(path: pathlib.Path, expected_sha256: str) -> dict[str, Any]:
    """Load and validate a hash-locked gate configuration."""

    _, raw_config = _locked_json(pathlib.Path(path), expected_sha256, "configuration")
    return _validate_config(raw_config)


def _validate_choice_score_output(value: Any) -> None:
    value = _mapping(value, "output choice_scores")
    _exact_keys(value, _CHOICE_SCORE_KEYS, "output choice_scores")
    if value["used_for_gate"] is not False:
        raise GateValidationError("choice scores must remain diagnostic-only")
    if value["status"] == "unavailable":
        if any(value[name] is not None for name in ("method", "values", "token_ids")):
            raise GateValidationError("unavailable output choice scores have metadata")
        return
    if value["status"] != "available":
        raise GateValidationError("output choice score status is unexpected")
    method = _plain_string(value["method"], "output choice score method")
    if method != CHOICE_SCORE_METHOD:
        raise GateValidationError("output choice score method is unexpected")
    values = _mapping(value["values"], "output choice score values")
    token_ids = _mapping(value["token_ids"], "output choice token IDs")
    normalized = _choice_score_record(
        InferenceResult(
            record_id="validation",
            raw_response="A",
            choice_scores=values,
            choice_score_method=method,
            choice_token_ids=token_ids,
        )
    )
    if dict(value) != normalized:
        raise GateValidationError("output choice scores are not canonical")


def _validate_candidate_design(
    value: Any,
    media_by_id: Mapping[str, Mapping[str, Any]],
    expected_sha256: str,
    option_order_seed: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != EXPECTED_PAIR_COUNT:
        raise GateValidationError("candidate design must contain exactly 200 pairs")
    design: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    role_counts: Counter[str] = Counter()
    target_roles: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    target_definitions: dict[tuple[str, str, int], tuple[Any, ...]] = {}
    partition_targets: dict[str, set[tuple[str, str, int]]] = defaultdict(set)
    video_owners: dict[str, tuple[str, str]] = {}
    graph: dict[str, set[str]] = defaultdict(set)
    component_videos: dict[str, set[str]] = defaultdict(set)
    component_partitions: dict[str, str] = {}
    for index, raw_pair in enumerate(value):
        pair = _mapping(raw_pair, f"candidate_design[{index}]")
        _exact_keys(pair, _DESIGN_KEYS, f"candidate_design[{index}]")
        pair_id = _plain_string(pair["pair_id"], "candidate design pair_id")
        if not _HEX24_RE.fullmatch(pair_id) or pair_id in pair_ids:
            raise GateValidationError(
                "candidate design pair IDs must be unique 24-character hex values"
            )
        pair_ids.add(pair_id)
        pair_role = pair["pair_role"]
        if pair_role not in PAIR_ROLES:
            raise GateValidationError("candidate design pair role is unexpected")
        role_counts[pair_role] += 1
        component_id = _plain_string(
            pair["component_id"], "candidate design component_id"
        )
        if not _HEX24_RE.fullmatch(component_id):
            raise GateValidationError("candidate design component ID is unexpected")
        partition = pair["partition"]
        if partition not in PARTITIONS:
            raise GateValidationError("candidate design partition is unexpected")
        prior_partition = component_partitions.setdefault(component_id, partition)
        if prior_partition != partition:
            raise GateValidationError("candidate design component crosses partitions")
        _sha256_text(pair["key_sha256"], "candidate design key_sha256")
        top_prompt = _plain_string(pair["prompt"], "candidate design prompt")
        prompt_sha256 = _sha256_text(
            pair["prompt_sha256"], "candidate design prompt_sha256"
        )
        if hashlib.sha256(top_prompt.encode("utf-8")).hexdigest() != prompt_sha256:
            raise GateValidationError("candidate design prompt SHA-256 differs")
        normalized_sources: dict[str, dict[str, Any]] = {}
        for source_role in SOURCE_ROLES:
            source = _mapping(pair[source_role], f"candidate design {source_role}")
            _exact_keys(
                source,
                _DESIGN_SOURCE_KEYS,
                f"candidate design {source_role}",
            )
            video_id = _plain_string(source["video_id"], "candidate design video_id")
            if not _VIDEO_ID_RE.fullmatch(video_id):
                raise GateValidationError("candidate design video ID is unsafe")
            question_id = _nonnegative_int(
                source["question_id"], "candidate design question_id"
            )
            question = _plain_string(source["question"], "candidate design question")
            normalized_question_sha256 = _sha256_text(
                source["normalized_question_sha256"],
                "candidate design normalized question SHA-256",
            )
            if hashlib.sha256(normalize_text(question).encode("utf-8")).hexdigest() != (
                normalized_question_sha256
            ):
                raise GateValidationError(
                    "candidate design normalized question SHA-256 differs"
                )
            if source["expected_choice"] not in CHOICES:
                raise GateValidationError(
                    "candidate design expected choice is unexpected"
                )
            media_sha256 = _sha256_text(
                source["media_sha256"], "candidate design media_sha256"
            )
            media = media_by_id.get(video_id)
            if media is None or media["sha256"] != media_sha256:
                raise GateValidationError(
                    "candidate design media differs from inventory"
                )
            balanced_options = source["balanced_options"]
            if (
                not isinstance(balanced_options, list)
                or len(balanced_options) != len(CHOICES)
                or len(set(balanced_options)) != len(CHOICES)
                or not all(
                    isinstance(option, str) and option for option in balanced_options
                )
            ):
                raise GateValidationError(
                    "candidate design balanced options are unexpected"
                )
            option_balance_slot = _nonnegative_int(
                source["option_balance_slot"],
                "candidate design option balance slot",
            )
            if option_balance_slot >= len(CHOICES):
                raise GateValidationError(
                    "candidate design option balance slot is outside A/B/C"
                )
            if source["expected_choice"] != CHOICES[option_balance_slot]:
                raise GateValidationError(
                    "candidate design option balance slot differs from expected choice"
                )
            if (
                _balanced_options(
                    {
                        "options": balanced_options,
                        "answer": balanced_options[option_balance_slot],
                    },
                    option_balance_slot,
                )
                != balanced_options
            ):
                raise GateValidationError(
                    "candidate design option order differs from fixed balancing policy"
                )
            source_prompt = _plain_string(
                source["prompt"], "candidate design source prompt"
            )
            source_prompt_sha256 = _sha256_text(
                source["prompt_sha256"], "candidate design source prompt_sha256"
            )
            if source_prompt != _prompt(question, balanced_options):
                raise GateValidationError(
                    "candidate design source prompt differs from balanced options"
                )
            if (
                hashlib.sha256(source_prompt.encode("utf-8")).hexdigest()
                != source_prompt_sha256
            ):
                raise GateValidationError(
                    "candidate design source prompt SHA-256 differs"
                )
            shuffled_media_video_id = _plain_string(
                source["shuffled_media_video_id"],
                "candidate design shuffled media video_id",
            )
            if not _VIDEO_ID_RE.fullmatch(shuffled_media_video_id):
                raise GateValidationError(
                    "candidate design shuffled media video ID is unsafe"
                )
            shuffled_media_sha256 = _sha256_text(
                source["shuffled_media_sha256"],
                "candidate design shuffled media SHA-256",
            )
            shuffled_media = media_by_id.get(shuffled_media_video_id)
            if (
                shuffled_media is None
                or shuffled_media["sha256"] != shuffled_media_sha256
                or shuffled_media_video_id == video_id
            ):
                raise GateValidationError(
                    "candidate design shuffled media differs from its fixed counteranswer"
                )
            shuffled_media_choice = source["shuffled_media_choice"]
            if (
                shuffled_media_choice not in CHOICES
                or shuffled_media_choice == source["expected_choice"]
            ):
                raise GateValidationError(
                    "candidate design shuffled media choice must be a counteranswer"
                )
            owner = (component_id, partition)
            prior_owner = video_owners.setdefault(video_id, owner)
            if prior_owner != owner:
                raise GateValidationError(
                    "candidate design video crosses components or partitions"
                )
            component_videos[component_id].add(video_id)
            normalized_sources[source_role] = {
                "video_id": video_id,
                "question_id": question_id,
                "normalized_question_sha256": normalized_question_sha256,
                "question": question,
                "expected_choice": source["expected_choice"],
                "media_sha256": media_sha256,
                "balanced_options": list(balanced_options),
                "option_balance_slot": option_balance_slot,
                "prompt": source_prompt,
                "prompt_sha256": source_prompt_sha256,
                "shuffled_media_video_id": shuffled_media_video_id,
                "shuffled_media_sha256": shuffled_media_sha256,
                "shuffled_media_choice": shuffled_media_choice,
            }
        if (
            normalized_sources["target"]["video_id"]
            == normalized_sources["donor"]["video_id"]
        ):
            raise GateValidationError("candidate design target and donor must differ")
        if (
            normalized_sources["target"]["question"]
            != normalized_sources["donor"]["question"]
            or normalized_sources["target"]["normalized_question_sha256"]
            != normalized_sources["donor"]["normalized_question_sha256"]
            or sorted(normalized_sources["target"]["balanced_options"])
            != sorted(normalized_sources["donor"]["balanced_options"])
        ):
            raise GateValidationError(
                "candidate design target and donor question vocabularies differ"
            )
        if (
            pair["prompt"] != normalized_sources["target"]["prompt"]
            or prompt_sha256 != normalized_sources["target"]["prompt_sha256"]
        ):
            raise GateValidationError(
                "candidate design top-level prompt differs from target orientation"
            )
        choices_match = (
            normalized_sources["target"]["balanced_options"][
                CHOICES.index(normalized_sources["target"]["expected_choice"])
            ]
            == normalized_sources["donor"]["balanced_options"][
                CHOICES.index(normalized_sources["donor"]["expected_choice"])
            ]
        )
        if pair_role == "same_answer_nuisance" and not choices_match:
            raise GateValidationError("same-answer candidate design choices differ")
        if pair_role == "opposite_answer_candidate" and choices_match:
            raise GateValidationError("opposite-answer candidate design choices match")
        graph[normalized_sources["target"]["video_id"]].add(
            normalized_sources["donor"]["video_id"]
        )
        graph[normalized_sources["donor"]["video_id"]].add(
            normalized_sources["target"]["video_id"]
        )
        target_key = (
            component_id,
            normalized_sources["target"]["video_id"],
            normalized_sources["target"]["question_id"],
        )
        target_roles[target_key].add(pair_role)
        partition_targets[partition].add(target_key)
        target_definition = (
            pair["key_sha256"],
            normalized_sources["target"]["normalized_question_sha256"],
            prompt_sha256,
            normalized_sources["target"]["expected_choice"],
            normalized_sources["target"]["media_sha256"],
            tuple(normalized_sources["target"]["balanced_options"]),
            normalized_sources["target"]["option_balance_slot"],
            normalized_sources["target"]["prompt_sha256"],
            normalized_sources["target"]["shuffled_media_video_id"],
            normalized_sources["target"]["shuffled_media_sha256"],
            normalized_sources["target"]["shuffled_media_choice"],
        )
        prior_definition = target_definitions.setdefault(target_key, target_definition)
        if prior_definition != target_definition:
            raise GateValidationError(
                "candidate design target differs across pair roles"
            )
        design.append(
            {
                "pair_id": pair_id,
                "pair_role": pair_role,
                "component_id": component_id,
                "partition": partition,
                "key_sha256": pair["key_sha256"],
                "prompt": top_prompt,
                "prompt_sha256": prompt_sha256,
                **normalized_sources,
            }
        )
    if design != sorted(design, key=lambda item: item["pair_id"]):
        raise GateValidationError("candidate design is not in canonical pair-ID order")
    if set(media_by_id) != set(video_owners):
        raise GateValidationError("media inventory differs from candidate design")
    if role_counts != Counter({role: EXPECTED_TARGET_COUNT for role in PAIR_ROLES}):
        raise GateValidationError("candidate design pair-role counts differ")
    if len(target_roles) != EXPECTED_TARGET_COUNT or any(
        roles != set(PAIR_ROLES) for roles in target_roles.values()
    ):
        raise GateValidationError(
            "candidate design targets do not have both pair roles"
        )
    if {
        partition: len(partition_targets[partition]) for partition in PARTITIONS
    } != EXPECTED_PARTITION_TARGET_COUNTS:
        raise GateValidationError("candidate design partition target counts differ")
    slot_pairs = [
        {
            "pair_id": pair["pair_id"],
            "partition": pair["partition"],
            "role": pair["pair_role"],
            "target": pair["target"],
            "donor": pair["donor"],
        }
        for pair in design
    ]
    expected_slots = _option_balance_slots(slot_pairs, option_order_seed)
    for pair in design:
        for source_role in SOURCE_ROLES:
            if (
                pair[source_role]["option_balance_slot"]
                != expected_slots[(pair["pair_id"], source_role)]
            ):
                raise GateValidationError(
                    "candidate design option balance slot differs from fixed assignment"
                )
    pairs_by_target: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(
        dict
    )
    for pair in design:
        target_key = (pair["target"]["video_id"], pair["target"]["question_id"])
        pairs_by_target[target_key][pair["pair_role"]] = pair
    for pair in design:
        target_key = (pair["target"]["video_id"], pair["target"]["question_id"])
        opposite_pair = pairs_by_target[target_key]["opposite_answer_candidate"]
        for source_role in SOURCE_ROLES:
            source = pair[source_role]
            if source_role == "target" or pair["pair_role"] == "same_answer_nuisance":
                shuffled_source = opposite_pair["donor"]
            else:
                shuffled_source = pair["target"]
            shuffled_answer = shuffled_source["balanced_options"][
                CHOICES.index(shuffled_source["expected_choice"])
            ]
            if shuffled_answer not in source["balanced_options"]:
                raise GateValidationError(
                    "candidate design shuffled answer is absent from orientation options"
                )
            expected_shuffled_choice = CHOICES[
                source["balanced_options"].index(shuffled_answer)
            ]
            if (
                source["shuffled_media_video_id"] != shuffled_source["video_id"]
                or source["shuffled_media_sha256"] != shuffled_source["media_sha256"]
                or source["shuffled_media_choice"] != expected_shuffled_choice
            ):
                raise GateValidationError(
                    "candidate design shuffled media differs from fixed assignment"
                )
    for component_id, videos in component_videos.items():
        expected_component_id = hashlib.sha256(
            json.dumps(sorted(videos), separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24]
        if component_id != expected_component_id:
            raise GateValidationError(
                "candidate design component ID differs from its canonical video set"
            )
        start = next(iter(videos))
        reached: set[str] = set()
        pending = [start]
        while pending:
            video_id = pending.pop()
            if video_id in reached:
                continue
            reached.add(video_id)
            pending.extend(graph[video_id] - reached)
        if reached != videos:
            raise GateValidationError(
                "candidate design components differ from the pair graph"
            )
    if _digest_value(design) != expected_sha256:
        raise GateValidationError("candidate design SHA-256 does not match")
    return design


def validate_gate_output(
    value: Any,
    *,
    expected_config_sha256: str | None = None,
    expected_pilot_index_sha256: str | None = None,
) -> None:
    """Validate a complete result and its self-authenticating digest."""

    output = _mapping(value, "gate output")
    _exact_keys(output, _OUTPUT_KEYS, "gate output")
    payload_sha256 = _sha256_text(output["payload_sha256"], "payload_sha256")
    unhashed = {key: output[key] for key in output if key != "payload_sha256"}
    if _digest_value(unhashed) != payload_sha256:
        raise GateValidationError("payload SHA-256 does not match")
    if output["schema"] != OUTPUT_SCHEMA:
        raise GateValidationError("output schema is unexpected")
    model = _mapping(output["model"], "output model")
    if model != {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "frozen": True,
        "talker_disabled": True,
    }:
        raise GateValidationError("output model identity differs")
    embedded_config = _validate_config(
        _mapping(output["configuration"], "output configuration")
    )
    if dict(output["configuration"]) != embedded_config:
        raise GateValidationError("output configuration is not canonical")
    inference = _mapping(output["inference"], "output inference")
    _exact_keys(inference, _INFERENCE_KEYS, "output inference")
    _positive_int(inference["batch_size"], "output inference.batch_size", 32)
    _positive_int(
        inference["maximum_new_tokens"],
        "output inference.maximum_new_tokens",
        16,
    )
    _nonnegative_int(inference["seed"], "output inference.seed")
    _bool(
        inference["emit_choice_scores_when_single_token"],
        "output inference.emit_choice_scores_when_single_token",
    )
    if (
        inference["conditions"] != list(CONDITIONS)
        or inference["do_sample"] is not False
    ):
        raise GateValidationError("output inference settings differ")
    if dict(inference) != _public_inference_config(embedded_config):
        raise GateValidationError(
            "output inference differs from embedded configuration"
        )
    digests = _mapping(output["input_digests"], "output input_digests")
    expected_digest_keys = {
        "configuration_sha256",
        "configuration_value_sha256",
        "pilot_index_sha256",
        "source_set_sha256",
        "media_set_sha256",
        "candidate_design_sha256",
        "run_input_sha256",
    }
    _exact_keys(digests, expected_digest_keys, "output input_digests")
    for name, digest in digests.items():
        _sha256_text(digest, f"output input_digests.{name}")
    if expected_config_sha256 is not None and digests[
        "configuration_sha256"
    ] != _sha256_text(expected_config_sha256, "expected configuration SHA-256"):
        raise GateValidationError("output configuration SHA-256 differs from expected")
    if expected_pilot_index_sha256 is not None and digests[
        "pilot_index_sha256"
    ] != _sha256_text(expected_pilot_index_sha256, "expected pilot index SHA-256"):
        raise GateValidationError("output pilot index SHA-256 differs from expected")
    if _digest_value(embedded_config) != digests["configuration_value_sha256"]:
        raise GateValidationError("configuration value SHA-256 does not match")

    source_files = output["source_files"]
    if not isinstance(source_files, list):
        raise GateValidationError("output source_files must be a list")
    source_names: list[str] = []
    for index, raw_source in enumerate(source_files):
        source = _mapping(raw_source, f"output source_files[{index}]")
        _exact_keys(
            source,
            {"filename", "size_bytes", "sha256"},
            f"output source_files[{index}]",
        )
        filename = _plain_string(source["filename"], "output source filename")
        if pathlib.PurePosixPath(filename).name != filename:
            raise GateValidationError("output source filename must be plain")
        source_names.append(filename)
        _nonnegative_int(source["size_bytes"], "output source size_bytes")
        _sha256_text(source["sha256"], "output source sha256")
    if source_names != sorted(set(source_names)):
        raise GateValidationError("output source files must be sorted and unique")
    if _digest_value(source_files) != digests["source_set_sha256"]:
        raise GateValidationError("source set SHA-256 does not match")

    media_files = output["media_files"]
    if not isinstance(media_files, list) or not media_files:
        raise GateValidationError("output media_files must be a nonempty list")
    media_names: list[str] = []
    media_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw_media in enumerate(media_files):
        media = _mapping(raw_media, f"output media_files[{index}]")
        _exact_keys(
            media,
            {"video_id", "filename", "size_bytes", "sha256", "stream_types"},
            f"output media_files[{index}]",
        )
        video_id = _plain_string(media["video_id"], "output media video_id")
        filename = _plain_string(media["filename"], "output media filename")
        if not _VIDEO_ID_RE.fullmatch(video_id) or filename != f"{video_id}.mp4":
            raise GateValidationError("output media identity is unexpected")
        if video_id in media_by_id:
            raise GateValidationError("output media video IDs must be unique")
        media_by_id[video_id] = media
        media_names.append(video_id)
        _positive_int(media["size_bytes"], "output media size_bytes")
        _sha256_text(media["sha256"], "output media sha256")
        stream_types = media["stream_types"]
        if (
            not isinstance(stream_types, list)
            or stream_types != sorted(set(stream_types))
            or not all(isinstance(item, str) and item for item in stream_types)
            or not {"audio", "video"} <= set(stream_types)
        ):
            raise GateValidationError("output media stream types are unexpected")
    if media_names != sorted(media_names):
        raise GateValidationError("output media files must be sorted")
    if _digest_value(media_files) != digests["media_set_sha256"]:
        raise GateValidationError("media set SHA-256 does not match")
    expected_run_input_sha256 = _digest_value(
        {
            "configuration_sha256": digests["configuration_sha256"],
            "configuration_value_sha256": digests["configuration_value_sha256"],
            "pilot_index_sha256": digests["pilot_index_sha256"],
            "source_set_sha256": digests["source_set_sha256"],
            "media_set_sha256": digests["media_set_sha256"],
            "candidate_design_sha256": digests["candidate_design_sha256"],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
    )
    if expected_run_input_sha256 != digests["run_input_sha256"]:
        raise GateValidationError("run input SHA-256 does not match")

    design = _validate_candidate_design(
        output["candidate_design"],
        media_by_id,
        digests["candidate_design_sha256"],
        embedded_config["controls"]["option_order_seed"],
    )
    expected_record_design: list[dict[str, Any]] = []

    def append_expected_record(
        pair: Mapping[str, Any],
        source_role: str,
        evaluation: str,
        condition: str,
    ) -> None:
        source = pair[source_role]
        if evaluation == "question_only":
            media_video_id = None
            media_sha256 = None
            shuffled_media_choice = None
        elif evaluation == "source_sufficiency":
            media_video_id = source["video_id"]
            media_sha256 = source["media_sha256"]
            shuffled_media_choice = None
        else:
            media_video_id = source["shuffled_media_video_id"]
            media_sha256 = source["shuffled_media_sha256"]
            shuffled_media_choice = source["shuffled_media_choice"]
        identity = {
            "pair_id": pair["pair_id"],
            "pair_role": pair["pair_role"],
            "evaluation": evaluation,
            "source_role": source_role,
            "condition": condition,
        }
        record_id = _digest_value(identity)[:24]
        expected_record_design.append(
            {
                **identity,
                "record_id": record_id,
                "request_sha256": _digest_value(
                    {
                        **identity,
                        "record_id": record_id,
                        "component_id": pair["component_id"],
                        "partition": pair["partition"],
                        "video_id": source["video_id"],
                        "media_video_id": media_video_id,
                        "question_id": source["question_id"],
                        "normalized_question_sha256": source[
                            "normalized_question_sha256"
                        ],
                        "expected_choice": source["expected_choice"],
                        "media_sha256": media_sha256,
                        "shuffled_media_choice": shuffled_media_choice,
                        "prompt_sha256": source["prompt_sha256"],
                        "model_id": MODEL_ID,
                        "model_revision": MODEL_REVISION,
                        "configuration_sha256": digests["configuration_sha256"],
                        "pilot_index_sha256": digests["pilot_index_sha256"],
                    }
                ),
                "component_id": pair["component_id"],
                "partition": pair["partition"],
                "video_id": source["video_id"],
                "media_video_id": media_video_id,
                "question_id": source["question_id"],
                "normalized_question_sha256": source["normalized_question_sha256"],
                "expected_choice": source["expected_choice"],
                "media_sha256": media_sha256,
                "shuffled_media_choice": shuffled_media_choice,
                "prompt_sha256": source["prompt_sha256"],
            }
        )

    for condition in CONDITIONS:
        for pair in design:
            for source_role in SOURCE_ROLES:
                append_expected_record(
                    pair, source_role, "source_sufficiency", condition
                )
    for pair in design:
        for source_role in SOURCE_ROLES:
            append_expected_record(pair, source_role, "question_only", "question_only")
    for condition in CONDITIONS:
        for pair in design:
            for source_role in SOURCE_ROLES:
                append_expected_record(
                    pair,
                    source_role,
                    "same_question_shuffled_media",
                    condition,
                )

    records_value = output["records"]
    if not isinstance(records_value, list) or len(records_value) != len(
        expected_record_design
    ):
        raise GateValidationError("output records differ from the complete design")
    record_ids: set[str] = set()
    records: list[Mapping[str, Any]] = []
    for index, (raw_record, expected_record) in enumerate(
        zip(records_value, expected_record_design)
    ):
        record = _mapping(raw_record, f"output records[{index}]")
        _exact_keys(record, _RECORD_KEYS, f"output records[{index}]")
        record_id = record["record_id"]
        if (
            not isinstance(record_id, str)
            or not _HEX24_RE.fullmatch(record_id)
            or record_id in record_ids
        ):
            raise GateValidationError(
                "output record IDs must be unique 24-character hex values"
            )
        record_ids.add(record_id)
        for name, expected in expected_record.items():
            if record[name] != expected:
                raise GateValidationError(
                    f"output record {name} differs from candidate design"
                )
        for name in ("request_sha256", "prompt_sha256"):
            _sha256_text(record[name], f"output record {name}")
        if (
            record["pair_role"] not in PAIR_ROLES
            or record["source_role"] not in SOURCE_ROLES
            or record["evaluation"] not in EVALUATIONS
        ):
            raise GateValidationError("output record role or evaluation is unexpected")
        if record["evaluation"] == "question_only":
            if (
                record["condition"] != "question_only"
                or record["media_video_id"] is not None
                or record["media_sha256"] is not None
                or record["shuffled_media_choice"] is not None
            ):
                raise GateValidationError(
                    "question-only output record must not contain media"
                )
        else:
            if record["condition"] not in CONDITIONS:
                raise GateValidationError("output record condition is unexpected")
            media_video_id = _plain_string(
                record["media_video_id"], "output record media_video_id"
            )
            media_sha256 = _sha256_text(
                record["media_sha256"], "output record media_sha256"
            )
            media = media_by_id.get(media_video_id)
            if media is None or media["sha256"] != media_sha256:
                raise GateValidationError(
                    "output record media SHA-256 differs from inventory"
                )
            if record["evaluation"] == "source_sufficiency":
                if (
                    media_video_id != record["video_id"]
                    or record["shuffled_media_choice"] is not None
                ):
                    raise GateValidationError(
                        "source-sufficiency output record media differs"
                    )
            elif (
                media_video_id == record["video_id"]
                or record["shuffled_media_choice"] not in CHOICES
                or record["shuffled_media_choice"] == record["expected_choice"]
            ):
                raise GateValidationError(
                    "shuffled-media output record is not a counteranswer control"
                )
        if record["video_id"] not in media_by_id:
            raise GateValidationError(
                "output record source video differs from inventory"
            )
        if record["partition"] not in PARTITIONS:
            raise GateValidationError("output record partition is unexpected")
        if record["expected_choice"] not in CHOICES:
            raise GateValidationError("output expected choice is unexpected")
        _nonnegative_int(record["question_id"], "output record question_id")
        raw_response = _plain_string(
            record["raw_response"], "output raw_response", nonempty=False
        )
        if len(raw_response) > 256:
            raise GateValidationError("output raw_response is too long")
        if record["parse_status"] == "strict":
            if parse_choice(record["raw_response"]) != record["predicted_choice"]:
                raise GateValidationError(
                    "output parsed choice differs from raw response"
                )
        elif record["parse_status"] == "invalid":
            if record["predicted_choice"] is not None:
                raise GateValidationError(
                    "invalid output response has a predicted choice"
                )
            try:
                parse_choice(record["raw_response"])
            except GateValidationError:
                pass
            else:
                raise GateValidationError("invalid output response is actually strict")
        else:
            raise GateValidationError("output parse status is unexpected")
        if record["is_correct"] is not (
            record["predicted_choice"] == record["expected_choice"]
        ):
            raise GateValidationError("output correctness flag differs")
        _validate_choice_score_output(record["choice_scores"])
        if (
            not inference["emit_choice_scores_when_single_token"]
            and record["choice_scores"]["status"] != "unavailable"
        ):
            raise GateValidationError(
                "choice scores are present although inference disabled them"
            )
        records.append(record)

    counts = _mapping(output["counts"], "output counts")
    design_target_count = len(
        {
            (
                pair["component_id"],
                pair["target"]["video_id"],
                pair["target"]["question_id"],
            )
            for pair in design
        }
    )
    expected_counts = {
        "condition_count": len(CONDITIONS),
        "control_type_count": len(CONTROL_TYPES),
        "media_file_count": len(output["media_files"]),
        "target_count": design_target_count,
        "pair_count": len(design),
        "pair_role_counts": dict(
            sorted(Counter(pair["pair_role"] for pair in design).items())
        ),
        "partition_target_counts": {
            partition: len(
                {
                    (
                        pair["component_id"],
                        pair["target"]["video_id"],
                        pair["target"]["question_id"],
                    )
                    for pair in design
                    if pair["partition"] == partition
                }
            )
            for partition in PARTITIONS
        },
        "source_record_count": sum(
            record["evaluation"] == "source_sufficiency" for record in records
        ),
        "control_record_count": sum(
            record["evaluation"] in CONTROL_TYPES for record in records
        ),
        "record_count": len(records),
    }
    if dict(counts) != expected_counts:
        raise GateValidationError("output counts differ from records")
    gate = _mapping(output["gate"], "output gate")
    expected_gate = _summarize_gate(records, embedded_config["gate"])
    if dict(gate) != expected_gate:
        raise GateValidationError("output gate summary differs from records")
    attribution = _mapping(output["source_attribution"], "output source_attribution")
    _exact_keys(
        attribution,
        {"dataset", "authors", "copyright", "materials_license", "license_url"},
        "output source_attribution",
    )
    if (
        attribution["materials_license"] != "CC-BY-4.0"
        or attribution["license_url"]
        != "https://creativecommons.org/licenses/by/4.0/legalcode"
    ):
        raise GateValidationError("output attribution is not CC-BY-4.0")
    for name in ("dataset", "authors", "copyright"):
        _plain_string(attribution[name], f"output source_attribution.{name}")
    change_note = _plain_string(output["change_note"], "output change_note")
    if "CC-BY-4.0" not in change_note:
        raise GateValidationError("output change note must preserve the CC-BY notice")
    runtime = _mapping(output["runtime"], "output runtime")
    _exact_keys(runtime, _RUNTIME_KEYS, "output runtime")
    expected_runtime = {
        "backend": "qwen2_5_omni",
        "loaded": True,
        "torch_version": TORCH_VERSION,
        "transformers_version": TRANSFORMERS_VERSION,
        "qwen_omni_utils_version": QWEN_OMNI_UTILS_VERSION,
        "model_dtype": "torch.float16",
        "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
        "frozen": True,
    }
    for name, expected in expected_runtime.items():
        if runtime[name] != expected:
            raise GateValidationError(f"output runtime.{name} is unexpected")
    cuda_runtime_version = _plain_string(
        runtime["cuda_runtime_version"], "output runtime.cuda_runtime_version"
    )
    gpu_name = _plain_string(runtime["gpu_name"], "output runtime.gpu_name")
    if (
        len(cuda_runtime_version) > 64
        or len(gpu_name) > 128
        or any(character in gpu_name for character in ("/", "\\", "\n", "\r"))
    ):
        raise GateValidationError("output runtime contains unsafe public metadata")
    transcript_sha256 = _sha256_text(
        output["inference_transcript_sha256"],
        "output inference transcript SHA-256",
    )
    if _digest_value(_inference_transcript_from_output(output)) != transcript_sha256:
        raise GateValidationError("inference transcript SHA-256 does not match")


class _TranscriptReplayBackend:
    def __init__(self, transcript: Mapping[str, Any]) -> None:
        self._runtime = dict(_mapping(transcript["runtime"], "transcript runtime"))
        self._results: dict[str, InferenceResult] = {}
        raw_results = transcript["results"]
        if not isinstance(raw_results, list) or not raw_results:
            raise GateValidationError("transcript results must be a nonempty list")
        for index, raw_result in enumerate(raw_results):
            result = _mapping(raw_result, f"transcript results[{index}]")
            _exact_keys(
                result,
                {"record_id", "request_sha256", "raw_response", "choice_scores"},
                f"transcript results[{index}]",
            )
            record_id = _plain_string(result["record_id"], "transcript record ID")
            if not _HEX24_RE.fullmatch(record_id) or record_id in self._results:
                raise GateValidationError(
                    "transcript record IDs must be unique hex values"
                )
            _sha256_text(result["request_sha256"], "transcript request SHA-256")
            raw_response = _plain_string(
                result["raw_response"], "transcript raw response", nonempty=False
            )
            if len(raw_response) > 256:
                raise GateValidationError("transcript raw response is too long")
            _validate_choice_score_output(result["choice_scores"])
            scores = result["choice_scores"]
            self._results[record_id] = InferenceResult(
                record_id=record_id,
                raw_response=raw_response,
                choice_scores=(
                    dict(scores["values"]) if scores["status"] == "available" else None
                ),
                choice_score_method=(
                    scores["method"] if scores["status"] == "available" else None
                ),
                choice_token_ids=(
                    dict(scores["token_ids"])
                    if scores["status"] == "available"
                    else None
                ),
            )

    def infer(self, requests: Sequence[InferenceRequest]) -> Sequence[InferenceResult]:
        try:
            return [self._results[request.record_id] for request in requests]
        except KeyError as exc:
            raise GateValidationError(
                "transcript does not cover every reconstructed request"
            ) from exc

    def runtime_details(self) -> Mapping[str, Any]:
        return self._runtime


def verify_gate_output_replay(
    output: Mapping[str, Any],
    *,
    config_path: pathlib.Path,
    expected_config_sha256: str,
    pilot_index_path: pathlib.Path,
    expected_pilot_index_sha256: str,
    media_root: pathlib.Path,
    source_paths: Sequence[pathlib.Path],
    transcript_path: pathlib.Path,
    expected_transcript_sha256: str,
) -> dict[str, Any]:
    """Rebuild a result from locked inputs and a separately anchored transcript."""

    _, raw_transcript = _locked_json(
        pathlib.Path(transcript_path),
        expected_transcript_sha256,
        "inference transcript",
    )
    transcript = _mapping(raw_transcript, "inference transcript")
    _exact_keys(
        transcript,
        {"schema", "model", "input_digests", "runtime", "results"},
        "inference transcript",
    )
    if transcript["schema"] != INFERENCE_TRANSCRIPT_SCHEMA:
        raise GateValidationError("inference transcript schema is unexpected")
    if transcript["model"] != {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "frozen": True,
        "talker_disabled": True,
    }:
        raise GateValidationError("inference transcript model identity differs")
    backend = _TranscriptReplayBackend(transcript)
    reconstructed = run_source_sufficiency_gate(
        config_path=pathlib.Path(config_path),
        expected_config_sha256=expected_config_sha256,
        pilot_index_path=pathlib.Path(pilot_index_path),
        expected_pilot_index_sha256=expected_pilot_index_sha256,
        media_root=pathlib.Path(media_root),
        backend=backend,
        source_paths=source_paths,
    )
    reconstructed_transcript = _inference_transcript_from_output(reconstructed)
    if transcript != reconstructed_transcript:
        raise GateValidationError(
            "inference transcript differs from replayed requests or inputs"
        )
    validate_gate_output(
        output,
        expected_config_sha256=expected_config_sha256,
        expected_pilot_index_sha256=expected_pilot_index_sha256,
    )
    if dict(output) != reconstructed:
        raise GateValidationError("replay reconstruction differs from gate output")
    return reconstructed


def write_gate_output(
    path: pathlib.Path,
    output: Mapping[str, Any],
    *,
    expected_config_sha256: str | None = None,
    expected_pilot_index_sha256: str | None = None,
) -> None:
    """Atomically write a validated result without overwriting prior work."""

    validate_gate_output(
        output,
        expected_config_sha256=expected_config_sha256,
        expected_pilot_index_sha256=expected_pilot_index_sha256,
    )
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
            raise GateValidationError("output path already exists") from exc
        os.unlink(temporary_name)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


class QwenOmniBackend:
    """Lazy frozen Qwen2.5-Omni backend for media and question-only inputs."""

    def __init__(
        self,
        *,
        model_config: Mapping[str, Any],
        inference_config: Mapping[str, Any],
        cache_dir: pathlib.Path,
    ) -> None:
        if (
            model_config.get("id") != MODEL_ID
            or model_config.get("revision") != MODEL_REVISION
        ):
            raise GateValidationError("backend requires the pinned model identity")
        self._model_config = dict(model_config)
        self._inference_config = dict(inference_config)
        self._video_preprocessing = _validate_video_preprocessing(
            self._inference_config.get("video_preprocessing")
        )
        self._cache_dir = pathlib.Path(cache_dir)
        self._model: Any = None
        self._processor: Any = None
        self._process_mm_info: Any = None
        self._torch: Any = None
        self._runtime: dict[str, Any] = {
            "backend": "qwen2_5_omni",
            "loaded": False,
        }

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if self.loaded:
            return
        try:
            import torch
            from qwen_omni_utils import process_mm_info
            from transformers import (
                Qwen2_5OmniForConditionalGeneration,
                Qwen2_5OmniProcessor,
            )
        except ImportError as exc:
            raise GateValidationError(
                "Qwen2.5-Omni runtime dependencies are unavailable"
            ) from exc
        runtime_versions = {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "qwen-omni-utils": importlib.metadata.version("qwen-omni-utils"),
        }
        expected_versions = {
            "torch": TORCH_VERSION,
            "transformers": TRANSFORMERS_VERSION,
            "qwen-omni-utils": QWEN_OMNI_UTILS_VERSION,
        }
        if runtime_versions != expected_versions:
            raise GateValidationError(
                "Qwen2.5-Omni runtime dependency versions differ from the lock"
            )
        configured_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if configured_workspace not in (None, ":4096:8"):
            raise GateValidationError(
                "CUBLAS_WORKSPACE_CONFIG conflicts with the deterministic lock"
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if not torch.cuda.is_available():
            raise GateValidationError(
                "Qwen2.5-Omni inference requires a visible CUDA device"
            )
        seed = self._inference_config["seed"]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=str(self._cache_dir),
            dtype=torch.float16,
            device_map=self._model_config["device_map"],
            low_cpu_mem_usage=True,
            attn_implementation=self._model_config["attention_implementation"],
        )
        model.disable_talker()
        model.requires_grad_(False)
        model.eval()
        processor = Qwen2_5OmniProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=str(self._cache_dir),
            use_fast=False,
        )
        self._model = model
        self._processor = processor
        self._process_mm_info = process_mm_info
        self._torch = torch
        self._runtime = {
            "backend": "qwen2_5_omni",
            "loaded": True,
            "torch_version": torch.__version__,
            "transformers_version": importlib.metadata.version("transformers"),
            "qwen_omni_utils_version": importlib.metadata.version("qwen-omni-utils"),
            "cuda_runtime_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "model_dtype": str(model.dtype),
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "frozen": all(
                not parameter.requires_grad for parameter in model.parameters()
            ),
        }

    def _conversation(
        self, request: InferenceRequest
    ) -> tuple[list[dict[str, Any]], bool]:
        if request.evaluation == "question_only":
            if (
                request.condition != "question_only"
                or request.media_path is not None
                or request.media_video_id is not None
            ):
                raise GateValidationError(
                    "question-only backend request contains media"
                )
            return [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": request.prompt}],
                }
            ], False
        if request.media_path is None:
            raise GateValidationError("media backend request is missing media")
        if request.condition == "audio_only":
            media = {"type": "audio", "audio": str(request.media_path)}
            use_audio_in_video = False
        elif request.condition == "video_only":
            media = {
                "type": "video",
                "video": str(request.media_path),
                **self._video_preprocessing,
            }
            use_audio_in_video = False
        elif request.condition == "audiovisual":
            media = {
                "type": "video",
                "video": str(request.media_path),
                **self._video_preprocessing,
            }
            use_audio_in_video = True
        else:
            raise GateValidationError("backend request condition is unexpected")
        return [
            {
                "role": "user",
                "content": [media, {"type": "text", "text": request.prompt}],
            }
        ], use_audio_in_video

    @staticmethod
    def _extend_media(destination: list[Any], value: Any) -> None:
        if value is None:
            return
        if isinstance(value, list):
            destination.extend(value)
        else:
            destination.append(value)

    @staticmethod
    def _contextual_choice_token_ids(
        tokenizer: Any, prompt: str
    ) -> dict[str, int] | None:
        """Resolve one-token labels at the exact formatted prompt boundary."""

        prefix_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        token_ids: dict[str, int] = {}
        for choice in CHOICES:
            candidate_ids = list(
                tokenizer.encode(prompt + choice, add_special_tokens=False)
            )
            if (
                len(candidate_ids) != len(prefix_ids) + 1
                or candidate_ids[:-1] != prefix_ids
            ):
                return None
            token_ids[choice] = candidate_ids[-1]
        if len(set(token_ids.values())) != len(CHOICES):
            return None
        return token_ids

    def infer(self, requests: Sequence[InferenceRequest]) -> Sequence[InferenceResult]:
        requests = list(requests)
        if not requests:
            return []
        if len({request.condition for request in requests}) != 1:
            raise GateValidationError(
                "Qwen backend batches must use one media condition"
            )
        if len({request.evaluation for request in requests}) != 1:
            raise GateValidationError(
                "Qwen backend batches must use one evaluation type"
            )
        self._load()
        torch = self._torch
        processor = self._processor
        model = self._model
        use_audio_in_video_values: set[bool] = set()
        prompts: list[str] = []
        audios: list[Any] = []
        images: list[Any] = []
        videos: list[Any] = []
        sampled_video_fps: list[float] = []
        for request in requests:
            conversation, use_audio_in_video = self._conversation(request)
            use_audio_in_video_values.add(use_audio_in_video)
            prompts.append(
                processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            )
            audio_values, image_values, video_values, video_kwargs = (
                self._process_mm_info(
                    conversation,
                    use_audio_in_video=use_audio_in_video,
                    return_video_kwargs=True,
                )
            )
            self._extend_media(audios, audio_values)
            self._extend_media(images, image_values)
            self._extend_media(videos, video_values)
            if not isinstance(video_kwargs, dict) or set(video_kwargs) != {"fps"}:
                raise GateValidationError(
                    "Qwen media utility returned invalid video metadata"
                )
            fps_values = video_kwargs["fps"]
            if not isinstance(fps_values, list):
                raise GateValidationError(
                    "Qwen media utility returned invalid sampled video FPS"
                )
            for fps in fps_values:
                if (
                    isinstance(fps, bool)
                    or not isinstance(fps, (int, float))
                    or not math.isfinite(float(fps))
                    or float(fps) <= 0.0
                ):
                    raise GateValidationError(
                        "Qwen media utility returned invalid sampled video FPS"
                    )
                sampled_video_fps.append(float(fps))
        if len(use_audio_in_video_values) != 1:
            raise GateValidationError(
                "Qwen backend audio-in-video settings differ within a batch"
            )
        processor_kwargs: dict[str, Any] = {
            "text": prompts,
            "return_tensors": "pt",
            "padding": True,
            "use_audio_in_video": next(iter(use_audio_in_video_values)),
        }
        if audios:
            processor_kwargs["audio"] = audios
        if images:
            processor_kwargs["images"] = images
        if videos:
            if len(videos) != 1 or len(sampled_video_fps) != 1:
                raise GateValidationError(
                    "each video batch must have exactly one sampled FPS value"
                )
            processor_kwargs["videos"] = videos
            processor_kwargs["fps"] = sampled_video_fps[0]
        elif sampled_video_fps:
            raise GateValidationError(
                "sampled video FPS was returned without a video input"
            )
        model_inputs = processor(**processor_kwargs)
        model_inputs = model_inputs.to(model.device).to(model.dtype)
        emit_scores = self._inference_config["emit_choice_scores_when_single_token"]
        use_audio_in_video = next(iter(use_audio_in_video_values))
        generate_kwargs = {
            "do_sample": False,
            "thinker_max_new_tokens": self._inference_config["maximum_new_tokens"],
            "use_audio_in_video": use_audio_in_video,
            "return_audio": False,
            "return_dict_in_generate": True,
            "output_logits": emit_scores,
        }
        with torch.inference_mode():
            generated = model.generate(**model_inputs, **generate_kwargs)
        input_width = model_inputs.input_ids.shape[1]
        generated_only = generated.sequences[:, input_width:]
        responses = processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(responses) != len(requests):
            raise GateValidationError(
                "Qwen backend response count differs from its batch"
            )
        token_ids_by_request = (
            [
                self._contextual_choice_token_ids(processor.tokenizer, prompt)
                for prompt in prompts
            ]
            if emit_scores
            else [None] * len(prompts)
        )
        results: list[InferenceResult] = []
        for index, (request, response) in enumerate(zip(requests, responses)):
            choice_scores = None
            method = None
            result_token_ids = None
            token_ids = token_ids_by_request[index]
            generated_logits = getattr(generated, "logits", None)
            if token_ids is not None and generated_logits:
                first_logits = generated_logits[0][index]
                choice_scores = {
                    choice: float(first_logits[token_id].detach().float().cpu().item())
                    for choice, token_id in token_ids.items()
                }
                method = CHOICE_SCORE_METHOD
                result_token_ids = token_ids
            results.append(
                InferenceResult(
                    record_id=request.record_id,
                    raw_response=response,
                    choice_scores=choice_scores,
                    choice_score_method=method,
                    choice_token_ids=result_token_ids,
                )
            )
        return results

    def runtime_details(self) -> Mapping[str, Any]:
        return dict(self._runtime)
