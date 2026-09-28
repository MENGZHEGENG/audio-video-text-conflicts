"""Strict records and policy inputs for the Perception Test raw studies.

The JSON records retain provenance and evaluation fields. The policy view is
constructed separately and exposes only pre-query fields plus the observed
media object returned by an observed-only access boundary.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import struct
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Optional, Protocol

ACQUISITION_EXAMPLE_SCHEMA = "conflictbench.perception-acquisition-example.v2"
CONFLICT_EXAMPLE_SCHEMA = "conflictbench.perception-conflict-example.v2"
POLICY_SAFE_VIEW_SCHEMA = "conflictbench.perception-policy-safe-view.v2"
POLICY_FEATURE_PAYLOAD_SCHEMA = "conflictbench.perception-policy-features.v1"

OFFICIAL_PARTITIONS = ("train", "validation")
ANALYSIS_PARTITIONS = (
    "semantic_fit",
    "router_fit",
    "threshold_calibration",
    "evaluation",
)
LEGAL_PARTITION_PAIRS = frozenset(
    {
        ("train", "semantic_fit"),
        ("train", "router_fit"),
        ("train", "threshold_calibration"),
        ("validation", "evaluation"),
    }
)
MODALITIES = ("audio", "video")
ACQUISITION_CONDITIONS = ("clean_original", "missing_source", "temporal_shift")
CONFLICT_CONDITIONS = (
    "clean_original",
    "remux_only",
    "same_answer",
    "opposite_answer",
    "missing_source",
    "temporal_shift",
)
ORIENTATIONS = ("audio_over_video", "video_over_audio")

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_CHOICES = ("A", "B", "C")

_ACQUISITION_KEYS = {
    "schema",
    "example_id",
    "partitions",
    "bindings",
    "target",
    "question",
    "modalities",
    "observed_condition",
    "outputs",
    "acquisition_cost",
}
_ACQUISITION_QUESTION_KEYS = {
    "text",
    "options",
    "option_permutation",
    "gold_answer",
}
_CONFLICT_KEYS = {
    "schema",
    "example_id",
    "partitions",
    "bindings",
    "condition",
    "orientation",
    "question",
    "physical_sources",
    "source_answers",
    "integrated",
    "output",
}
_CONFLICT_QUESTION_KEYS = {"text", "options", "option_permutation"}
POLICY_FEATURE_ALLOWLIST = (
    "choice_probability_a",
    "choice_probability_b",
    "choice_probability_c",
    "entropy",
    "top_two_margin",
    "duration_seconds",
    "question_token_count",
    "option_token_count_mean",
)
_POLICY_FEATURE_PAYLOAD_KEYS = {
    "schema",
    "pre_query_output_sha256",
    "features",
}
_PRE_QUERY_OUTPUT_KEYS = {
    "output_id",
    "sha256",
    "feature_payload_sha256",
}
_POST_QUERY_OUTPUT_KEYS = {"output_id", "sha256"}


class RawIndexValidationError(ValueError):
    """Raised when a raw-study record violates its frozen contract."""


@dataclass(frozen=True)
class ObservedMediaRequest:
    """Internal lookup key used to load the observed source only."""

    __slots__ = ("condition", "modality", "question_id", "video_id")

    video_id: str
    question_id: int
    modality: str
    condition: str


@dataclass(frozen=True)
class ObservedMedia:
    """Observed-only media bytes crossing the policy access boundary."""

    __slots__ = ("modality", "observed_bytes", "sha256")

    modality: str
    observed_bytes: bytes
    sha256: str


@dataclass(frozen=True)
class PolicyObservedMedia:
    """Observed media exposed to a policy without a stored identity digest."""

    __slots__ = ("modality", "observed_bytes")

    modality: str
    observed_bytes: bytes


@dataclass(frozen=True)
class ObservedPolicyFeatures:
    """Frozen, observed-only scalar features plus their source-output binding."""

    __slots__ = (
        "payload_bytes",
        "payload_sha256",
        "pre_query_output_sha256",
        *POLICY_FEATURE_ALLOWLIST,
    )

    payload_bytes: bytes
    payload_sha256: str
    pre_query_output_sha256: str
    choice_probability_a: float
    choice_probability_b: float
    choice_probability_c: float
    entropy: float
    top_two_margin: float
    duration_seconds: float
    question_token_count: int
    option_token_count_mean: float


class ObservedMediaAccess(Protocol):
    """Access boundary available while creating a pre-query policy input."""

    def load_observed(self, request: ObservedMediaRequest) -> ObservedMedia:
        """Load the already observed source for one validated example."""


@dataclass(frozen=True)
class PolicySafeView:
    """Immutable policy input with no IDs, provenance, or realized outcomes."""

    __slots__ = (
        "schema",
        "question",
        "options",
        "observed_input",
        "observed_modality",
        "candidate_modality",
        "acquisition_cost",
        *POLICY_FEATURE_ALLOWLIST,
    )

    schema: str
    question: str
    options: tuple[str, ...]
    observed_input: PolicyObservedMedia
    observed_modality: str
    candidate_modality: str
    acquisition_cost: float
    choice_probability_a: float
    choice_probability_b: float
    choice_probability_c: float
    entropy: float
    top_two_margin: float
    duration_seconds: float
    question_token_count: int
    option_token_count_mean: float


@dataclass(frozen=True)
class PairedImpossibilityInput:
    """Hash-bound raw inputs used to reconstruct one hidden-role member."""

    __slots__ = (
        "acquisition_example_bytes",
        "acquisition_example_sha256",
        "conflict_example_bytes",
        "conflict_example_sha256",
        "feature_payload_bytes",
        "pair_id",
    )

    pair_id: str
    acquisition_example_bytes: bytes
    acquisition_example_sha256: str
    conflict_example_bytes: bytes
    conflict_example_sha256: str
    feature_payload_bytes: bytes


@dataclass(frozen=True)
class PairedImpossibilityRecord:
    """One validated result from a paired non-identifiability control."""

    __slots__ = (
        "acquisition_example_id",
        "acquisition_example_sha256",
        "config_sha256",
        "conflict_example_id",
        "conflict_example_sha256",
        "feature_payload_sha256",
        "hidden_donor_role",
        "model_sha256",
        "pair_id",
        "policy_view",
        "query_score",
        "router_sha256",
    )

    pair_id: str
    hidden_donor_role: str
    acquisition_example_id: str
    acquisition_example_sha256: str
    conflict_example_id: str
    conflict_example_sha256: str
    feature_payload_sha256: str
    router_sha256: str
    model_sha256: str
    config_sha256: str
    policy_view: PolicySafeView
    query_score: float


class QueryRouter(Protocol):
    """Router boundary that receives only a validated policy-safe view."""

    router_sha256: str
    model_sha256: str
    config_sha256: str

    def score_query(self, view: PolicySafeView) -> float:
        """Return the normalized query score for one policy input."""


_EXACT_DATACLASS_FIELDS: dict[type[Any], tuple[str, ...]] = {
    ObservedMediaRequest: ("video_id", "question_id", "modality", "condition"),
    ObservedMedia: ("modality", "observed_bytes", "sha256"),
    PolicyObservedMedia: ("modality", "observed_bytes"),
    ObservedPolicyFeatures: (
        "payload_bytes",
        "payload_sha256",
        "pre_query_output_sha256",
        *POLICY_FEATURE_ALLOWLIST,
    ),
    PolicySafeView: (
        "schema",
        "question",
        "options",
        "observed_input",
        "observed_modality",
        "candidate_modality",
        "acquisition_cost",
        *POLICY_FEATURE_ALLOWLIST,
    ),
    PairedImpossibilityInput: (
        "pair_id",
        "acquisition_example_bytes",
        "acquisition_example_sha256",
        "conflict_example_bytes",
        "conflict_example_sha256",
        "feature_payload_bytes",
    ),
    PairedImpossibilityRecord: (
        "pair_id",
        "hidden_donor_role",
        "acquisition_example_id",
        "acquisition_example_sha256",
        "conflict_example_id",
        "conflict_example_sha256",
        "feature_payload_sha256",
        "router_sha256",
        "model_sha256",
        "config_sha256",
        "policy_view",
        "query_score",
    ),
}


def _exact_dataclass_state(value: Any, expected_type: type[Any], field: str) -> None:
    if type(value) is not expected_type:
        raise RawIndexValidationError(
            f"{field} must be an exact {expected_type.__name__} frozen slotted record"
        )
    expected_fields = _EXACT_DATACLASS_FIELDS[expected_type]
    actual_fields = (
        tuple(item.name for item in fields(value)) if is_dataclass(value) else ()
    )
    slots = getattr(type(value), "__slots__", ())
    if (
        actual_fields != expected_fields
        or tuple(sorted(slots)) != tuple(sorted(expected_fields))
        or hasattr(value, "__dict__")
        or not value.__dataclass_params__.frozen
    ):
        raise RawIndexValidationError(
            f"{field} must be an exact {expected_type.__name__} frozen slotted record"
        )


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RawIndexValidationError(f"{field} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(str(key) for key in observed - expected)
        raise RawIndexValidationError(
            f"{field} keys differ; missing={missing}, unknown={unknown}"
        )


def _normalized_text(value: Any, field: str) -> str:
    if type(value) is not str:
        raise RawIndexValidationError(f"{field} must be a string")
    normalized = " ".join(unicodedata.normalize("NFC", value).split())
    if not normalized:
        raise RawIndexValidationError(f"{field} must not be empty")
    return normalized


def _canonical_text(value: Any, field: str) -> str:
    normalized = _normalized_text(value, field)
    if value != normalized:
        raise RawIndexValidationError(f"{field} must use canonical text normalization")
    return normalized


def _identifier(value: Any, field: str) -> str:
    if type(value) is not str or _SAFE_IDENTIFIER_RE.fullmatch(value) is None:
        raise RawIndexValidationError(f"{field} must be a safe non-path identifier")
    return value


def canonical_question_key_sha256(question: str, options: Sequence[str]) -> str:
    """Hash canonical question text and permutation-invariant answer options."""

    normalized_question = _normalized_text(question, "question")
    if (
        isinstance(options, (str, bytes, Mapping))
        or not isinstance(options, Sequence)
        or len(options) != len(_CHOICES)
    ):
        raise RawIndexValidationError("options must contain exactly three strings")
    normalized_options = sorted(
        _normalized_text(option, f"options[{index}]")
        for index, option in enumerate(options)
    )
    if len(set(normalized_options)) != len(normalized_options):
        raise RawIndexValidationError("options must contain three unique strings")
    encoded = json.dumps(
        {"options": normalized_options, "question": normalized_question},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: Any, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise RawIndexValidationError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_integer(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise RawIndexValidationError(f"{field} must be a nonnegative integer")
    return value


def _finite_number(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RawIndexValidationError(f"{field} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted < minimum:
        raise RawIndexValidationError(
            f"{field} must be finite and at least {minimum:g}"
        )
    return converted


def _validate_partitions(value: Any, field: str) -> None:
    partitions = _mapping(value, field)
    _exact_keys(partitions, {"official", "analysis"}, field)
    if (
        type(partitions["official"]) is not str
        or type(partitions["analysis"]) is not str
    ):
        raise RawIndexValidationError(f"{field} partition roles must be strings")
    partition_pair = (partitions["official"], partitions["analysis"])
    if partition_pair not in LEGAL_PARTITION_PAIRS:
        raise RawIndexValidationError(
            f"{field} partition pair {partition_pair!r} is not permitted"
        )


def _validate_question_key_binding(
    question: Mapping[str, Any], question_key_sha256: Any, field: str
) -> None:
    supplied = _sha256(question_key_sha256, field)
    expected = canonical_question_key_sha256(question["text"], question["options"])
    if supplied != expected:
        raise RawIndexValidationError(
            f"{field} does not match the canonical question and options"
        )


def _validate_options_and_permutation(question: Mapping[str, Any], field: str) -> None:
    options = question["options"]
    if not isinstance(options, list) or len(options) != len(_CHOICES):
        raise RawIndexValidationError(
            f"{field}.options must have exactly three entries"
        )
    canonical_options = [
        _canonical_text(option, f"{field}.options[{index}]")
        for index, option in enumerate(options)
    ]
    if len(set(canonical_options)) != len(canonical_options):
        raise RawIndexValidationError(f"{field}.options must be unique")

    permutation = question["option_permutation"]
    if (
        not isinstance(permutation, list)
        or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in permutation
        )
        or sorted(permutation) != list(range(len(_CHOICES)))
    ):
        raise RawIndexValidationError(
            f"{field}.option_permutation must be a permutation of [0, 1, 2]"
        )


def _validate_output_reference(
    value: Any, field: str, *, expected_keys: set[str]
) -> None:
    output = _mapping(value, field)
    _exact_keys(output, expected_keys, field)
    _identifier(output["output_id"], f"{field}.output_id")
    _sha256(output["sha256"], f"{field}.sha256")
    if "feature_payload_sha256" in expected_keys:
        _sha256(
            output["feature_payload_sha256"],
            f"{field}.feature_payload_sha256",
        )


def _canonical_question_for_builder(
    value: Mapping[str, Any], expected: set[str], field: str
) -> dict[str, Any]:
    question = _mapping(value, field)
    _exact_keys(question, expected, field)
    result = copy.deepcopy(dict(question))
    result["text"] = _normalized_text(result["text"], f"{field}.text")
    options = result.get("options")
    if not isinstance(options, list):
        raise RawIndexValidationError(f"{field}.options must be a list")
    result["options"] = [
        _normalized_text(option, f"{field}.options[{index}]")
        for index, option in enumerate(options)
    ]
    return result


def validate_acquisition_example(value: Any) -> dict[str, Any]:
    """Validate and copy one acquisition example with exact nested keys."""

    example = _mapping(value, "acquisition example")
    _exact_keys(example, _ACQUISITION_KEYS, "acquisition example")
    if (
        type(example["schema"]) is not str
        or example["schema"] != ACQUISITION_EXAMPLE_SCHEMA
    ):
        raise RawIndexValidationError(
            f"acquisition example schema must be {ACQUISITION_EXAMPLE_SCHEMA}"
        )
    _identifier(example["example_id"], "acquisition example.example_id")
    _validate_partitions(example["partitions"], "acquisition example.partitions")

    bindings = _mapping(example["bindings"], "acquisition example.bindings")
    _exact_keys(
        bindings,
        {
            "component_id",
            "question_key_sha256",
            "observed_media_sha256",
            "candidate_video_id",
            "candidate_media_sha256",
        },
        "acquisition example.bindings",
    )
    _identifier(bindings["component_id"], "acquisition example.bindings.component_id")
    _sha256(
        bindings["observed_media_sha256"],
        "acquisition example.bindings.observed_media_sha256",
    )
    _identifier(
        bindings["candidate_video_id"],
        "acquisition example.bindings.candidate_video_id",
    )
    _sha256(
        bindings["candidate_media_sha256"],
        "acquisition example.bindings.candidate_media_sha256",
    )

    target = _mapping(example["target"], "acquisition example.target")
    _exact_keys(target, {"video_id", "question_id"}, "acquisition example.target")
    _identifier(target["video_id"], "acquisition example.target.video_id")
    if bindings["candidate_video_id"] == target["video_id"]:
        raise RawIndexValidationError(
            "acquisition example candidate video must differ from the target"
        )
    _nonnegative_integer(
        target["question_id"], "acquisition example.target.question_id"
    )

    question = _mapping(example["question"], "acquisition example.question")
    _exact_keys(question, _ACQUISITION_QUESTION_KEYS, "acquisition example.question")
    _canonical_text(question["text"], "acquisition example.question.text")
    _validate_options_and_permutation(question, "acquisition example.question")
    _validate_question_key_binding(
        question,
        bindings["question_key_sha256"],
        "acquisition example.bindings.question_key_sha256",
    )
    if (
        type(question["gold_answer"]) is not str
        or question["gold_answer"] not in _CHOICES
    ):
        raise RawIndexValidationError(
            "acquisition example.question.gold_answer must be A, B, or C"
        )

    modalities = _mapping(example["modalities"], "acquisition example.modalities")
    _exact_keys(modalities, {"observed", "candidate"}, "acquisition example.modalities")
    if (
        type(modalities["observed"]) is not str
        or type(modalities["candidate"]) is not str
        or modalities["observed"] not in MODALITIES
        or modalities["candidate"] not in MODALITIES
    ):
        raise RawIndexValidationError(
            "acquisition example modalities are not recognized"
        )
    if modalities["observed"] == modalities["candidate"]:
        raise RawIndexValidationError("observed and candidate modalities must differ")
    if (
        type(example["observed_condition"]) is not str
        or example["observed_condition"] not in ACQUISITION_CONDITIONS
    ):
        raise RawIndexValidationError(
            "acquisition example observed_condition is not recognized"
        )

    outputs = _mapping(example["outputs"], "acquisition example.outputs")
    _exact_keys(outputs, {"pre_query", "post_query"}, "acquisition example.outputs")
    _validate_output_reference(
        outputs["pre_query"],
        "acquisition example.outputs.pre_query",
        expected_keys=_PRE_QUERY_OUTPUT_KEYS,
    )
    _validate_output_reference(
        outputs["post_query"],
        "acquisition example.outputs.post_query",
        expected_keys=_POST_QUERY_OUTPUT_KEYS,
    )
    _finite_float(example["acquisition_cost"], "acquisition example.acquisition_cost")
    return copy.deepcopy(dict(example))


def build_acquisition_example(
    *,
    example_id: str,
    partitions: Mapping[str, Any],
    bindings: Mapping[str, Any],
    target: Mapping[str, Any],
    question: Mapping[str, Any],
    modalities: Mapping[str, Any],
    observed_condition: str,
    outputs: Mapping[str, Any],
    acquisition_cost: float,
) -> dict[str, Any]:
    """Build one canonical acquisition example and validate the result."""

    record = {
        "schema": ACQUISITION_EXAMPLE_SCHEMA,
        "example_id": example_id,
        "partitions": copy.deepcopy(dict(_mapping(partitions, "partitions"))),
        "bindings": copy.deepcopy(dict(_mapping(bindings, "bindings"))),
        "target": copy.deepcopy(dict(_mapping(target, "target"))),
        "question": _canonical_question_for_builder(
            question, _ACQUISITION_QUESTION_KEYS, "question"
        ),
        "modalities": copy.deepcopy(dict(_mapping(modalities, "modalities"))),
        "observed_condition": observed_condition,
        "outputs": copy.deepcopy(dict(_mapping(outputs, "outputs"))),
        "acquisition_cost": acquisition_cost,
    }
    return validate_acquisition_example(record)


def _validate_source(value: Any, field: str, target_video_id: str) -> None:
    source = _mapping(value, field)
    _exact_keys(source, {"role", "video_id"}, field)
    role = source["role"]
    if type(role) is not str or role not in {"target", "donor", "missing"}:
        raise RawIndexValidationError(f"{field}.role is not recognized")
    video_id = source["video_id"]
    if role == "missing":
        if video_id is not None:
            raise RawIndexValidationError(
                f"{field}.video_id must be null for a missing source"
            )
        return
    validated_video_id = _identifier(video_id, f"{field}.video_id")
    if role == "target" and validated_video_id != target_video_id:
        raise RawIndexValidationError(f"{field} target video binding differs")
    if role == "donor" and validated_video_id == target_video_id:
        raise RawIndexValidationError(f"{field} donor must differ from the target")


def _validate_conflict_sources(
    *,
    condition: str,
    orientation: Any,
    sources: Mapping[str, Any],
    source_answers: Mapping[str, Any],
    integrated: Mapping[str, Any],
) -> None:
    roles = {modality: sources[modality]["role"] for modality in MODALITIES}
    answers = {modality: source_answers[modality] for modality in MODALITIES}

    if condition in {"clean_original", "remux_only"}:
        if orientation is not None:
            raise RawIndexValidationError(f"{condition} orientation must be null")
        if roles != {"audio": "target", "video": "target"}:
            raise RawIndexValidationError(
                f"{condition} must use target audio and video"
            )
    elif orientation not in ORIENTATIONS:
        raise RawIndexValidationError(f"{condition} orientation is not recognized")

    if condition in {"same_answer", "opposite_answer"}:
        expected_roles = (
            {"audio": "donor", "video": "target"}
            if orientation == "audio_over_video"
            else {"audio": "target", "video": "donor"}
        )
        if roles != expected_roles:
            raise RawIndexValidationError(
                f"{condition} physical sources disagree with the orientation"
            )
    elif condition == "missing_source":
        expected_roles = (
            {"audio": "missing", "video": "target"}
            if orientation == "audio_over_video"
            else {"audio": "target", "video": "missing"}
        )
        if roles != expected_roles:
            raise RawIndexValidationError(
                "missing_source physical sources disagree with the orientation"
            )
    elif condition == "temporal_shift" and roles != {
        "audio": "target",
        "video": "target",
    }:
        raise RawIndexValidationError(
            "temporal_shift must retain target audio and video"
        )

    for modality in MODALITIES:
        answer = answers[modality]
        if type(answer) is not str or answer not in (*_CHOICES, "U"):
            raise RawIndexValidationError(
                f"source_answers.{modality} must be A, B, C, or U"
            )
        if roles[modality] == "missing" and answer != "U":
            raise RawIndexValidationError(f"missing {modality} must have answer U")
        if roles[modality] != "missing" and answer == "U":
            raise RawIndexValidationError(f"present {modality} must not have answer U")

    relation = integrated["relation"]
    action = integrated["action"]
    if type(relation) is not str or type(action) is not str:
        raise RawIndexValidationError(
            "integrated relation and action must be exact strings"
        )
    if relation == "AGREE":
        if answers["audio"] not in _CHOICES or answers["audio"] != answers["video"]:
            raise RawIndexValidationError("AGREE requires equal known source answers")
        if action != answers["audio"]:
            raise RawIndexValidationError(
                "AGREE action must equal the shared source answer"
            )
    elif relation == "CONFLICT":
        if (
            answers["audio"] not in _CHOICES
            or answers["video"] not in _CHOICES
            or answers["audio"] == answers["video"]
        ):
            raise RawIndexValidationError(
                "CONFLICT requires different known source answers"
            )
        if action != "ABSTAIN":
            raise RawIndexValidationError("CONFLICT requires ABSTAIN")
    elif relation == "INSUFFICIENT":
        if "U" not in answers.values():
            raise RawIndexValidationError(
                "INSUFFICIENT requires a missing source answer"
            )
        if action != "ABSTAIN":
            raise RawIndexValidationError("INSUFFICIENT requires ABSTAIN")
    else:
        raise RawIndexValidationError("integrated.relation is not recognized")

    expected_relation = {
        "clean_original": "AGREE",
        "remux_only": "AGREE",
        "same_answer": "AGREE",
        "opposite_answer": "CONFLICT",
        "missing_source": "INSUFFICIENT",
        "temporal_shift": "AGREE",
    }[condition]
    if relation != expected_relation:
        raise RawIndexValidationError(
            f"{condition} must have integrated relation {expected_relation}"
        )


def validate_conflict_example(value: Any) -> dict[str, Any]:
    """Validate and copy one conflict example with exact nested keys."""

    example = _mapping(value, "conflict example")
    _exact_keys(example, _CONFLICT_KEYS, "conflict example")
    if (
        type(example["schema"]) is not str
        or example["schema"] != CONFLICT_EXAMPLE_SCHEMA
    ):
        raise RawIndexValidationError(
            f"conflict example schema must be {CONFLICT_EXAMPLE_SCHEMA}"
        )
    _identifier(example["example_id"], "conflict example.example_id")
    _validate_partitions(example["partitions"], "conflict example.partitions")

    bindings = _mapping(example["bindings"], "conflict example.bindings")
    _exact_keys(
        bindings,
        {
            "target_video_id",
            "target_question_id",
            "pair_id",
            "component_id",
            "question_key_sha256",
            "source_media_sha256",
        },
        "conflict example.bindings",
    )
    target_video_id = _identifier(
        bindings["target_video_id"],
        "conflict example.bindings.target_video_id",
    )
    _nonnegative_integer(
        bindings["target_question_id"],
        "conflict example.bindings.target_question_id",
    )
    _identifier(bindings["pair_id"], "conflict example.bindings.pair_id")
    _identifier(bindings["component_id"], "conflict example.bindings.component_id")
    source_media_sha256 = _mapping(
        bindings["source_media_sha256"],
        "conflict example.bindings.source_media_sha256",
    )
    _exact_keys(
        source_media_sha256,
        set(MODALITIES),
        "conflict example.bindings.source_media_sha256",
    )

    condition = example["condition"]
    if type(condition) is not str or condition not in CONFLICT_CONDITIONS:
        raise RawIndexValidationError("conflict example condition is not recognized")
    orientation = example["orientation"]
    if orientation is not None and type(orientation) is not str:
        raise RawIndexValidationError(
            "conflict example orientation must be an exact string or null"
        )

    question = _mapping(example["question"], "conflict example.question")
    _exact_keys(question, _CONFLICT_QUESTION_KEYS, "conflict example.question")
    _canonical_text(question["text"], "conflict example.question.text")
    _validate_options_and_permutation(question, "conflict example.question")
    _validate_question_key_binding(
        question,
        bindings["question_key_sha256"],
        "conflict example.bindings.question_key_sha256",
    )

    sources = _mapping(example["physical_sources"], "conflict example.physical_sources")
    _exact_keys(sources, set(MODALITIES), "conflict example.physical_sources")
    for modality in MODALITIES:
        _validate_source(
            sources[modality],
            f"conflict example.physical_sources.{modality}",
            target_video_id,
        )
        source = sources[modality]
        media_sha256 = source_media_sha256[modality]
        if source["role"] == "missing":
            if media_sha256 is not None:
                raise RawIndexValidationError(
                    f"missing {modality} source media digest must be null"
                )
        else:
            _sha256(
                media_sha256,
                f"conflict example.bindings.source_media_sha256.{modality}",
            )
    present_source_digests: dict[str, str] = {}
    for modality in MODALITIES:
        source = sources[modality]
        if source["video_id"] is None:
            continue
        previous_digest = present_source_digests.setdefault(
            source["video_id"], source_media_sha256[modality]
        )
        if previous_digest != source_media_sha256[modality]:
            raise RawIndexValidationError(
                "one physical source video has inconsistent media digests"
            )

    source_answers = _mapping(
        example["source_answers"], "conflict example.source_answers"
    )
    _exact_keys(source_answers, set(MODALITIES), "conflict example.source_answers")
    integrated = _mapping(example["integrated"], "conflict example.integrated")
    _exact_keys(integrated, {"relation", "action"}, "conflict example.integrated")
    if type(integrated["action"]) is not str or integrated["action"] not in (
        *_CHOICES,
        "ABSTAIN",
    ):
        raise RawIndexValidationError("integrated.action is not recognized")

    output = _mapping(example["output"], "conflict example.output")
    _exact_keys(output, {"sha256"}, "conflict example.output")
    _sha256(output["sha256"], "conflict example.output.sha256")

    _validate_conflict_sources(
        condition=condition,
        orientation=orientation,
        sources=sources,
        source_answers=source_answers,
        integrated=integrated,
    )
    return copy.deepcopy(dict(example))


def build_conflict_example(
    *,
    example_id: str,
    partitions: Mapping[str, Any],
    bindings: Mapping[str, Any],
    condition: str,
    orientation: Any,
    question: Mapping[str, Any],
    physical_sources: Mapping[str, Any],
    source_answers: Mapping[str, Any],
    integrated: Mapping[str, Any],
    output: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one canonical conflict example and validate the result."""

    record = {
        "schema": CONFLICT_EXAMPLE_SCHEMA,
        "example_id": example_id,
        "partitions": copy.deepcopy(dict(_mapping(partitions, "partitions"))),
        "bindings": copy.deepcopy(dict(_mapping(bindings, "bindings"))),
        "condition": condition,
        "orientation": orientation,
        "question": _canonical_question_for_builder(
            question, _CONFLICT_QUESTION_KEYS, "question"
        ),
        "physical_sources": copy.deepcopy(
            dict(_mapping(physical_sources, "physical_sources"))
        ),
        "source_answers": copy.deepcopy(
            dict(_mapping(source_answers, "source_answers"))
        ),
        "integrated": copy.deepcopy(dict(_mapping(integrated, "integrated"))),
        "output": copy.deepcopy(dict(_mapping(output, "output"))),
    }
    return validate_conflict_example(record)


def _finite_float(
    value: Any,
    field: str,
    *,
    minimum: float = 0.0,
    maximum: Optional[float] = None,  # noqa: UP045 -- Python 3.9 reflection
    minimum_exclusive: bool = False,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise RawIndexValidationError(f"{field} must be a finite float")
    below_minimum = value <= minimum if minimum_exclusive else value < minimum
    if below_minimum or (maximum is not None and value > maximum):
        lower_bound = ">" if minimum_exclusive else ">="
        upper_bound = f" and <= {maximum:g}" if maximum is not None else ""
        raise RawIndexValidationError(
            f"{field} must be a finite float {lower_bound} {minimum:g}{upper_bound}"
        )
    return value


def _float_bits(value: Any, field: str) -> bytes:
    if type(value) is not float or not math.isfinite(value):
        raise RawIndexValidationError(f"{field} must be a finite float")
    return struct.pack("!d", value)


def _reject_json_constant(value: str) -> None:
    raise RawIndexValidationError(f"feature payload JSON contains invalid {value}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RawIndexValidationError(
                f"feature payload JSON contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _canonical_json_bytes(value: Any, *, label: str = "feature payload") -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RawIndexValidationError(
            f"{label} must be canonical finite JSON"
        ) from error


def _reject_duplicate_raw_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RawIndexValidationError(
                f"raw example JSON contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _parse_raw_example_bytes(
    payload_bytes: Any,
    *,
    expected_sha256: Any,
    label: str,
    validator: Callable[[Any], dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    if type(payload_bytes) is not bytes or not payload_bytes:
        raise RawIndexValidationError(f"{label} must be nonempty bytes")
    expected = _sha256(expected_sha256, f"{label} digest")
    observed = hashlib.sha256(payload_bytes).hexdigest()
    if observed != expected:
        raise RawIndexValidationError(
            f"{label} digest does not authenticate the supplied bytes"
        )
    try:
        value = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_raw_json_keys,
            parse_constant=_reject_json_constant,
        )
    except RawIndexValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RawIndexValidationError(
            f"{label} must contain valid UTF-8 JSON"
        ) from error
    validated = validator(value)
    if _canonical_json_bytes(validated, label=label) != payload_bytes:
        raise RawIndexValidationError(f"{label} JSON must use canonical encoding")
    return validated, observed


def _validate_feature_scalars(features: Mapping[str, Any]) -> None:
    probabilities = (
        _finite_float(
            features["choice_probability_a"],
            "choice_probability_a",
            maximum=1.0,
        ),
        _finite_float(
            features["choice_probability_b"],
            "choice_probability_b",
            maximum=1.0,
        ),
        _finite_float(
            features["choice_probability_c"],
            "choice_probability_c",
            maximum=1.0,
        ),
    )
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise RawIndexValidationError("choice_probabilities must sum to one")
    entropy = _finite_float(
        features["entropy"],
        "entropy",
        maximum=math.log(len(_CHOICES)),
    )
    top_two_margin = _finite_float(
        features["top_two_margin"],
        "top_two_margin",
        maximum=1.0,
    )
    expected_entropy = -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )
    ordered_probabilities = sorted(probabilities, reverse=True)
    expected_margin = ordered_probabilities[0] - ordered_probabilities[1]
    if not math.isclose(
        entropy, expected_entropy, rel_tol=0.0, abs_tol=1e-8
    ) or not math.isclose(top_two_margin, expected_margin, rel_tol=0.0, abs_tol=1e-8):
        raise RawIndexValidationError(
            "entropy and top_two_margin must be derived from choice_probabilities"
        )
    _finite_float(
        features["duration_seconds"],
        "duration_seconds",
        minimum_exclusive=True,
    )
    question_token_count = features["question_token_count"]
    if (
        isinstance(question_token_count, bool)
        or not isinstance(question_token_count, int)
        or question_token_count < 1
    ):
        raise RawIndexValidationError("question_token_count must be a positive integer")
    _finite_float(
        features["option_token_count_mean"],
        "option_token_count_mean",
        minimum_exclusive=True,
    )


def _parse_feature_payload(
    payload_bytes: Any,
    *,
    expected_payload_sha256: Any,
) -> tuple[dict[str, Any], str]:
    if type(payload_bytes) is not bytes or not payload_bytes:
        raise RawIndexValidationError("feature payload must be nonempty bytes")
    expected = _sha256(expected_payload_sha256, "feature payload digest")
    observed = hashlib.sha256(payload_bytes).hexdigest()
    if observed != expected:
        raise RawIndexValidationError(
            "feature payload digest does not authenticate the supplied bytes"
        )
    try:
        decoded = payload_bytes.decode("utf-8")
        payload = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except RawIndexValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RawIndexValidationError(
            "feature payload must contain valid UTF-8 JSON"
        ) from error
    if _canonical_json_bytes(payload) != payload_bytes:
        raise RawIndexValidationError(
            "feature payload JSON must use canonical encoding"
        )
    payload_mapping = _mapping(payload, "feature payload")
    _exact_keys(payload_mapping, _POLICY_FEATURE_PAYLOAD_KEYS, "feature payload")
    if payload_mapping["schema"] != POLICY_FEATURE_PAYLOAD_SCHEMA:
        raise RawIndexValidationError(
            f"feature payload schema must be {POLICY_FEATURE_PAYLOAD_SCHEMA}"
        )
    _sha256(
        payload_mapping["pre_query_output_sha256"],
        "feature payload.pre_query_output_sha256",
    )
    feature_mapping = _mapping(payload_mapping["features"], "feature payload.features")
    _exact_keys(
        feature_mapping,
        set(POLICY_FEATURE_ALLOWLIST),
        "feature payload.features",
    )
    _validate_feature_scalars(feature_mapping)
    return dict(payload_mapping), observed


def build_observed_policy_features(
    payload_bytes: bytes,
    *,
    expected_payload_sha256: str,
) -> ObservedPolicyFeatures:
    """Parse and authenticate one canonical pre-query feature payload."""

    payload, observed_sha256 = _parse_feature_payload(
        payload_bytes,
        expected_payload_sha256=expected_payload_sha256,
    )
    feature_values = payload["features"]
    return ObservedPolicyFeatures(
        payload_bytes=payload_bytes,
        payload_sha256=observed_sha256,
        pre_query_output_sha256=payload["pre_query_output_sha256"],
        **feature_values,
    )


def _validated_policy_features(
    value: Any,
    *,
    question: Mapping[str, Any],
    pre_query_output_sha256: str,
    feature_payload_sha256: str,
) -> ObservedPolicyFeatures:
    _exact_dataclass_state(value, ObservedPolicyFeatures, "observed_features")
    payload, observed_payload_sha256 = _parse_feature_payload(
        value.payload_bytes,
        expected_payload_sha256=feature_payload_sha256,
    )
    if value.payload_sha256 != observed_payload_sha256:
        raise RawIndexValidationError(
            "observed feature payload digest differs from authenticated bytes"
        )
    payload_features = payload["features"]
    for feature_name in POLICY_FEATURE_ALLOWLIST:
        payload_value = payload_features[feature_name]
        observed_value = getattr(value, feature_name)
        if type(payload_value) is float:
            matches = _float_bits(
                payload_value, f"feature payload.{feature_name}"
            ) == _float_bits(observed_value, feature_name)
        else:
            matches = type(observed_value) is type(payload_value) and (
                observed_value == payload_value
            )
        if not matches:
            raise RawIndexValidationError(
                f"observed feature {feature_name} differs from authenticated payload"
            )
    if value.pre_query_output_sha256 != payload["pre_query_output_sha256"]:
        raise RawIndexValidationError(
            "observed pre-query output digest differs from authenticated payload"
        )
    if value.pre_query_output_sha256 != pre_query_output_sha256:
        raise RawIndexValidationError(
            "observed features do not match the pre-query output digest"
        )
    _sha256(value.pre_query_output_sha256, "pre_query_output_sha256")

    expected_question_tokens = len(question["text"].split())
    expected_option_mean = sum(
        len(option.split()) for option in question["options"]
    ) / len(question["options"])
    if value.question_token_count != expected_question_tokens or not math.isclose(
        value.option_token_count_mean,
        expected_option_mean,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RawIndexValidationError(
            "question feature scalars must be derived from question and options"
        )
    return value


def build_policy_safe_view(
    example: Mapping[str, Any],
    *,
    observed_features: ObservedPolicyFeatures,
    media_access: ObservedMediaAccess,
) -> PolicySafeView:
    """Create a policy input without candidate access or evaluation fields."""

    validated = validate_acquisition_example(example)
    question = validated["question"]
    features = _validated_policy_features(
        observed_features,
        question=question,
        pre_query_output_sha256=validated["outputs"]["pre_query"]["sha256"],
        feature_payload_sha256=validated["outputs"]["pre_query"][
            "feature_payload_sha256"
        ],
    )

    request = ObservedMediaRequest(
        video_id=validated["target"]["video_id"],
        question_id=validated["target"]["question_id"],
        modality=validated["modalities"]["observed"],
        condition=validated["observed_condition"],
    )
    load_observed = getattr(media_access, "load_observed", None)
    if not callable(load_observed):
        raise RawIndexValidationError(
            "media_access must provide a load_observed method"
        )
    observed_input = load_observed(request)
    _exact_dataclass_state(observed_input, ObservedMedia, "load_observed result")
    if type(observed_input.modality) is not str:
        raise RawIndexValidationError("ObservedMedia modality must be an exact string")
    if observed_input.modality != request.modality:
        raise RawIndexValidationError(
            "ObservedMedia modality does not match the requested observed modality"
        )
    if (
        type(observed_input.observed_bytes) is not bytes
        or not observed_input.observed_bytes
    ):
        raise RawIndexValidationError(
            "ObservedMedia observed_bytes must be nonempty bytes"
        )
    observed_sha256 = _sha256(observed_input.sha256, "ObservedMedia.sha256")
    if hashlib.sha256(observed_input.observed_bytes).hexdigest() != observed_sha256:
        raise RawIndexValidationError(
            "ObservedMedia sha256 does not authenticate observed_bytes"
        )
    if observed_sha256 != validated["bindings"]["observed_media_sha256"]:
        raise RawIndexValidationError(
            "ObservedMedia sha256 differs from the acquisition binding"
        )
    observed_snapshot = PolicyObservedMedia(
        modality=observed_input.modality,
        observed_bytes=observed_input.observed_bytes,
    )
    return PolicySafeView(
        schema=POLICY_SAFE_VIEW_SCHEMA,
        question=question["text"],
        options=tuple(question["options"]),
        observed_input=observed_snapshot,
        observed_modality=validated["modalities"]["observed"],
        candidate_modality=validated["modalities"]["candidate"],
        acquisition_cost=float(validated["acquisition_cost"]),
        choice_probability_a=features.choice_probability_a,
        choice_probability_b=features.choice_probability_b,
        choice_probability_c=features.choice_probability_c,
        entropy=features.entropy,
        top_two_margin=features.top_two_margin,
        duration_seconds=features.duration_seconds,
        question_token_count=features.question_token_count,
        option_token_count_mean=features.option_token_count_mean,
    )


def _collection_items(value: Any, field: str) -> list[Any]:
    if (
        isinstance(value, (str, bytes, Mapping))
        or not isinstance(value, Sequence)
        or not value
    ):
        raise RawIndexValidationError(f"{field} must be a nonempty sequence")
    return list(value)


def _trusted_digest_map(value: Any, field: str) -> dict[str, str]:
    mapping = _mapping(value, field)
    if not mapping:
        raise RawIndexValidationError(f"{field} must not be empty")
    trusted: dict[str, str] = {}
    for example_id, digest in mapping.items():
        trusted[_identifier(example_id, f"{field} key")] = _sha256(
            digest, f"{field}[{example_id!r}]"
        )
    return trusted


class _ConnectedComponentRegistry:
    def __init__(self) -> None:
        self._component_partitions: dict[str, tuple[str, str]] = {}
        self._node_components: dict[tuple[str, str], str] = {}

    def add_component(self, component_id: str, partitions: Mapping[str, str]) -> None:
        role = (partitions["official"], partitions["analysis"])
        previous = self._component_partitions.setdefault(component_id, role)
        if previous != role:
            raise RawIndexValidationError(
                f"connected component {component_id!r} has multiple partition roles"
            )

    def add_node(self, kind: str, value: str, component_id: str) -> None:
        node = (kind, value)
        previous = self._node_components.setdefault(node, component_id)
        if previous != component_id:
            raise RawIndexValidationError(
                f"{kind} {value!r} spans more than one connected component"
            )


class _OutputIdentityRegistry:
    def __init__(self) -> None:
        self._ids: dict[str, str] = {}
        self._digests: dict[str, str] = {}
        self._refs: dict[tuple[str, str], str] = {}

    @staticmethod
    def _add_unique(
        registry: dict[Any, str], value: Any, role: str, field: str
    ) -> None:
        previous = registry.setdefault(value, role)
        if previous != role:
            raise RawIndexValidationError(
                f"{field} {value!r} is reused by {previous} and {role}"
            )

    def add_acquisition(self, example: Mapping[str, Any]) -> None:
        component_id = example["bindings"]["component_id"]
        for output_role in ("pre_query", "post_query"):
            output = example["outputs"][output_role]
            role = (
                f"acquisition {example['example_id']} component {component_id} "
                f"{output_role}"
            )
            output_id = output["output_id"]
            digest = output["sha256"]
            self._add_unique(self._ids, output_id, role, "output ID")
            self._add_unique(self._digests, digest, role, "output digest")
            self._add_unique(
                self._refs,
                (output_id, digest),
                role,
                "derived output reference",
            )
            feature_payload_sha256 = output.get("feature_payload_sha256")
            if feature_payload_sha256 is not None:
                self._add_unique(
                    self._digests,
                    feature_payload_sha256,
                    f"{role} feature payload",
                    "output digest",
                )

    def add_conflict(self, example: Mapping[str, Any]) -> None:
        role = (
            f"conflict {example['example_id']} component "
            f"{example['bindings']['component_id']} output"
        )
        self._add_unique(
            self._digests,
            example["output"]["sha256"],
            role,
            "output digest",
        )


class _MediaIdentityRegistry:
    def __init__(self) -> None:
        self._digests: dict[str, str] = {}
        self._video_ids: dict[str, str] = {}

    def add(self, video_id: str, sha256: str) -> None:
        previous = self._digests.setdefault(video_id, sha256)
        if previous != sha256:
            raise RawIndexValidationError(
                f"media digest for video_id {video_id!r} is inconsistent"
            )
        previous_video_id = self._video_ids.setdefault(sha256, video_id)
        if previous_video_id != video_id:
            raise RawIndexValidationError(
                f"media digest {sha256!r} is assigned to multiple video_id values"
            )

    def add_acquisition(self, example: Mapping[str, Any]) -> None:
        self.add(
            example["target"]["video_id"],
            example["bindings"]["observed_media_sha256"],
        )
        self.add(
            example["bindings"]["candidate_video_id"],
            example["bindings"]["candidate_media_sha256"],
        )

    def add_conflict(self, example: Mapping[str, Any]) -> None:
        for modality in MODALITIES:
            source = example["physical_sources"][modality]
            if source["video_id"] is not None:
                self.add(
                    source["video_id"],
                    example["bindings"]["source_media_sha256"][modality],
                )


def _register_acquisition_graph(
    example: Mapping[str, Any], registry: _ConnectedComponentRegistry
) -> None:
    bindings = example["bindings"]
    component_id = bindings["component_id"]
    registry.add_component(component_id, example["partitions"])
    registry.add_node("video_id", example["target"]["video_id"], component_id)
    registry.add_node(
        "video_id", example["bindings"]["candidate_video_id"], component_id
    )
    registry.add_node(
        "question_key_sha256", bindings["question_key_sha256"], component_id
    )


def _conflict_donor_video_id(
    example: Mapping[str, Any],
) -> Optional[str]:  # noqa: UP045 -- Python 3.9 reflection
    donors = [
        source["video_id"]
        for source in example["physical_sources"].values()
        if source["role"] == "donor"
    ]
    if len(donors) > 1:
        raise RawIndexValidationError("a conflict record must have at most one donor")
    return donors[0] if donors else None


def _semantic_answer(question: Mapping[str, Any], answer: str) -> str:
    if answer == "U":
        return answer
    return question["options"][_CHOICES.index(answer)]


def _register_acquisition_gold_answer(
    example: Mapping[str, Any], gold_answers: dict[tuple[str, str], str]
) -> None:
    key = (
        example["target"]["video_id"],
        example["bindings"]["question_key_sha256"],
    )
    answer = _semantic_answer(example["question"], example["question"]["gold_answer"])
    previous = gold_answers.setdefault(key, answer)
    if previous != answer:
        raise RawIndexValidationError(
            f"gold answer for video_id {example['target']['video_id']!r} "
            "is inconsistent"
        )


def _register_conflict_source_answers(
    example: Mapping[str, Any], source_answers: dict[tuple[str, str], str]
) -> None:
    for modality, source in example["physical_sources"].items():
        video_id = source["video_id"]
        if video_id is None:
            continue
        key = (video_id, example["bindings"]["question_key_sha256"])
        answer = _semantic_answer(
            example["question"], example["source_answers"][modality]
        )
        previous = source_answers.setdefault(key, answer)
        if previous != answer:
            raise RawIndexValidationError(
                f"source answer for video_id {video_id!r} is inconsistent"
            )


def _register_conflict_graph(
    example: Mapping[str, Any], registry: _ConnectedComponentRegistry
) -> None:
    bindings = example["bindings"]
    component_id = bindings["component_id"]
    registry.add_component(component_id, example["partitions"])
    registry.add_node("pair_id", bindings["pair_id"], component_id)
    registry.add_node(
        "question_key_sha256", bindings["question_key_sha256"], component_id
    )
    for source in example["physical_sources"].values():
        if source["video_id"] is not None:
            registry.add_node("video_id", source["video_id"], component_id)


def validate_acquisition_collection(value: Any) -> tuple[dict[str, Any], ...]:
    """Validate IDs, split roles, and graph isolation for Study A records."""

    items = _collection_items(value, "acquisition collection")
    validated: list[dict[str, Any]] = []
    example_ids: set[str] = set()
    decision_states: set[tuple[str, int, str, str]] = set()
    gold_answers: dict[tuple[str, str], str] = {}
    registry = _ConnectedComponentRegistry()
    output_registry = _OutputIdentityRegistry()
    media_registry = _MediaIdentityRegistry()
    for item in items:
        example = validate_acquisition_example(item)
        example_id = example["example_id"]
        if example_id in example_ids:
            raise RawIndexValidationError(f"duplicate example_id {example_id!r}")
        example_ids.add(example_id)
        state = (
            example["target"]["video_id"],
            example["target"]["question_id"],
            example["modalities"]["observed"],
            example["observed_condition"],
        )
        if state in decision_states:
            raise RawIndexValidationError(
                f"duplicate acquisition decision state for example_id {example_id!r}"
            )
        decision_states.add(state)
        _register_acquisition_gold_answer(example, gold_answers)
        _register_acquisition_graph(example, registry)
        output_registry.add_acquisition(example)
        media_registry.add_acquisition(example)
        validated.append(example)
    return tuple(validated)


def validate_conflict_collection(value: Any) -> tuple[dict[str, Any], ...]:
    """Validate IDs, pair bindings, and graph isolation for Study B records."""

    items = _collection_items(value, "conflict collection")
    validated: list[dict[str, Any]] = []
    example_ids: set[str] = set()
    condition_states: set[tuple[str, str, Any]] = set()
    pair_bindings: dict[str, tuple[Any, ...]] = {}
    source_answers: dict[tuple[str, str], str] = {}
    registry = _ConnectedComponentRegistry()
    output_registry = _OutputIdentityRegistry()
    media_registry = _MediaIdentityRegistry()
    for item in items:
        example = validate_conflict_example(item)
        example_id = example["example_id"]
        if example_id in example_ids:
            raise RawIndexValidationError(f"duplicate example_id {example_id!r}")
        example_ids.add(example_id)
        bindings = example["bindings"]
        state = (
            bindings["pair_id"],
            example["condition"],
            example["orientation"],
        )
        if state in condition_states:
            raise RawIndexValidationError(
                f"duplicate conflict condition state for example_id {example_id!r}"
            )
        condition_states.add(state)
        donor_video_id = _conflict_donor_video_id(example)
        pair_binding = (
            bindings["target_video_id"],
            bindings["target_question_id"],
            bindings["component_id"],
            bindings["question_key_sha256"],
            example["condition"],
            donor_video_id,
        )
        previous = pair_bindings.setdefault(bindings["pair_id"], pair_binding)
        if previous != pair_binding:
            raise RawIndexValidationError(
                f"pair_id {bindings['pair_id']!r} has inconsistent donor binding"
            )
        _register_conflict_source_answers(example, source_answers)
        _register_conflict_graph(example, registry)
        output_registry.add_conflict(example)
        media_registry.add_conflict(example)
        validated.append(example)
    return tuple(validated)


def validate_raw_index_collections(
    acquisition_examples: Any, conflict_examples: Any
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Validate cross-study ID uniqueness and connected-component isolation."""

    acquisition = validate_acquisition_collection(acquisition_examples)
    conflicts = validate_conflict_collection(conflict_examples)
    all_ids = [example["example_id"] for example in (*acquisition, *conflicts)]
    if len(all_ids) != len(set(all_ids)):
        raise RawIndexValidationError("example_id values must be unique across studies")
    registry = _ConnectedComponentRegistry()
    output_registry = _OutputIdentityRegistry()
    media_registry = _MediaIdentityRegistry()
    semantic_answers: dict[tuple[str, str], str] = {}
    for example in acquisition:
        _register_acquisition_gold_answer(example, semantic_answers)
        _register_acquisition_graph(example, registry)
        output_registry.add_acquisition(example)
        media_registry.add_acquisition(example)
    for example in conflicts:
        _register_conflict_source_answers(example, semantic_answers)
        _register_conflict_graph(example, registry)
        output_registry.add_conflict(example)
        media_registry.add_conflict(example)
    return acquisition, conflicts


def _policy_invariance_signature(view: Any) -> tuple[Any, ...]:
    _exact_dataclass_state(view, PolicySafeView, "policy_view")
    if type(view.schema) is not str or view.schema != POLICY_SAFE_VIEW_SCHEMA:
        raise RawIndexValidationError("policy_view schema is not recognized")
    _canonical_text(view.question, "policy_view.question")
    if (
        type(view.options) is not tuple
        or len(view.options) != len(_CHOICES)
        or any(type(option) is not str for option in view.options)
    ):
        raise RawIndexValidationError(
            "policy_view.options must be exactly three strings"
        )
    for index, option in enumerate(view.options):
        _canonical_text(option, f"policy_view.options[{index}]")
    if len(set(view.options)) != len(view.options):
        raise RawIndexValidationError("policy_view.options must be unique")
    _exact_dataclass_state(
        view.observed_input,
        PolicyObservedMedia,
        "policy_view.observed_input",
    )
    if (
        type(view.observed_input.observed_bytes) is not bytes
        or not view.observed_input.observed_bytes
    ):
        raise RawIndexValidationError(
            "policy_view observed media must contain nonempty bytes"
        )
    observed_sha256 = hashlib.sha256(view.observed_input.observed_bytes).hexdigest()
    if (
        type(view.observed_input.modality) is not str
        or type(view.observed_modality) is not str
        or type(view.candidate_modality) is not str
        or view.observed_modality not in MODALITIES
        or view.candidate_modality not in MODALITIES
        or view.observed_modality == view.candidate_modality
        or view.observed_input.modality != view.observed_modality
    ):
        raise RawIndexValidationError("policy_view modality fields are inconsistent")
    _finite_float(view.acquisition_cost, "policy_view.acquisition_cost")
    visible_features = {
        feature_name: getattr(view, feature_name)
        for feature_name in POLICY_FEATURE_ALLOWLIST
    }
    _validate_feature_scalars(visible_features)
    if view.question_token_count != len(view.question.split()):
        raise RawIndexValidationError(
            "policy_view.question_token_count must be derived from its question"
        )
    expected_option_mean = sum(len(option.split()) for option in view.options) / len(
        view.options
    )
    if not math.isclose(
        view.option_token_count_mean,
        expected_option_mean,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RawIndexValidationError(
            "policy_view.option_token_count_mean must be derived from its options"
        )
    return (
        view.schema,
        view.question,
        view.options,
        view.observed_input.modality,
        view.observed_input.observed_bytes,
        observed_sha256,
        view.observed_modality,
        view.candidate_modality,
        _float_bits(view.acquisition_cost, "policy_view.acquisition_cost"),
        _float_bits(view.choice_probability_a, "policy_view.choice_probability_a"),
        _float_bits(view.choice_probability_b, "policy_view.choice_probability_b"),
        _float_bits(view.choice_probability_c, "policy_view.choice_probability_c"),
        _float_bits(view.entropy, "policy_view.entropy"),
        _float_bits(view.top_two_margin, "policy_view.top_two_margin"),
        _float_bits(view.duration_seconds, "policy_view.duration_seconds"),
        view.question_token_count,
        _float_bits(
            view.option_token_count_mean,
            "policy_view.option_token_count_mean",
        ),
    )


def _snapshot_policy_view(view: PolicySafeView) -> PolicySafeView:
    """Copy a validated view so router-held references cannot alter results."""

    expected_signature = _policy_invariance_signature(view)
    observed_input = PolicyObservedMedia(
        modality=view.observed_input.modality,
        observed_bytes=view.observed_input.observed_bytes,
    )
    snapshot = PolicySafeView(
        schema=view.schema,
        question=view.question,
        options=tuple(view.options),
        observed_input=observed_input,
        observed_modality=view.observed_modality,
        candidate_modality=view.candidate_modality,
        acquisition_cost=view.acquisition_cost,
        choice_probability_a=view.choice_probability_a,
        choice_probability_b=view.choice_probability_b,
        choice_probability_c=view.choice_probability_c,
        entropy=view.entropy,
        top_two_margin=view.top_two_margin,
        duration_seconds=view.duration_seconds,
        question_token_count=view.question_token_count,
        option_token_count_mean=view.option_token_count_mean,
    )
    if _policy_invariance_signature(snapshot) != expected_signature:
        raise RawIndexValidationError("policy_view snapshot differs")
    return snapshot


def _validate_paired_raw_binding(
    acquisition: Mapping[str, Any], conflict: Mapping[str, Any]
) -> str:
    condition = conflict["condition"]
    if condition not in {"same_answer", "opposite_answer"}:
        raise RawIndexValidationError(
            "paired impossibility hidden donor roles are not recognized"
        )
    acquisition_binding = acquisition["bindings"]
    conflict_binding = conflict["bindings"]
    if (
        acquisition["target"]["video_id"] != conflict_binding["target_video_id"]
        or acquisition["target"]["question_id"]
        != conflict_binding["target_question_id"]
        or acquisition_binding["component_id"] != conflict_binding["component_id"]
        or acquisition_binding["question_key_sha256"]
        != conflict_binding["question_key_sha256"]
        or acquisition["partitions"] != conflict["partitions"]
    ):
        raise RawIndexValidationError(
            "paired raw example bindings do not identify the same target state"
        )
    conflict_question = conflict["question"]
    if (
        acquisition["question"]["text"] != conflict_question["text"]
        or acquisition["question"]["options"] != conflict_question["options"]
        or acquisition["question"]["option_permutation"]
        != conflict_question["option_permutation"]
    ):
        raise RawIndexValidationError(
            "paired raw example bindings do not share one displayed question"
        )
    candidate_modality = (
        "audio" if conflict["orientation"] == "audio_over_video" else "video"
    )
    if (
        acquisition["modalities"]["candidate"] != candidate_modality
        or acquisition["modalities"]["observed"]
        == acquisition["modalities"]["candidate"]
        or acquisition["observed_condition"] != "clean_original"
    ):
        raise RawIndexValidationError(
            "paired raw example modalities do not match the hidden donor orientation"
        )
    observed_modality = acquisition["modalities"]["observed"]
    candidate_source = conflict["physical_sources"][candidate_modality]
    observed_source = conflict["physical_sources"][observed_modality]
    if (
        observed_source["role"] != "target"
        or observed_source["video_id"] != acquisition["target"]["video_id"]
        or conflict["source_answers"][observed_modality]
        != acquisition["question"]["gold_answer"]
        or conflict["bindings"]["source_media_sha256"][observed_modality]
        != acquisition_binding["observed_media_sha256"]
    ):
        raise RawIndexValidationError(
            "paired raw example observed source is not the target answer"
        )
    if (
        candidate_source["role"] != "donor"
        or candidate_source["video_id"] != acquisition_binding["candidate_video_id"]
        or conflict["bindings"]["source_media_sha256"][candidate_modality]
        != acquisition_binding["candidate_media_sha256"]
    ):
        raise RawIndexValidationError(
            "paired raw example candidate donor binding differs"
        )
    return condition


def _validate_query_router_bindings(
    query_router: QueryRouter,
    *,
    router_sha256: str,
    model_sha256: str,
    config_sha256: str,
) -> None:
    for digest_name, expected_digest in (
        ("router_sha256", router_sha256),
        ("model_sha256", model_sha256),
        ("config_sha256", config_sha256),
    ):
        if getattr(query_router, digest_name, None) != expected_digest:
            raise RawIndexValidationError(
                f"{digest_name} does not match the query_router binding"
            )


def build_paired_impossibility_input(
    *,
    pair_id: str,
    acquisition_example: Mapping[str, Any],
    conflict_example: Mapping[str, Any],
    feature_payload_bytes: bytes,
) -> PairedImpossibilityInput:
    """Freeze two validated raw records as canonical hash-bound bytes."""

    validated_acquisition = validate_acquisition_example(acquisition_example)
    validated_conflict = validate_conflict_example(conflict_example)
    acquisition_bytes = _canonical_json_bytes(
        validated_acquisition, label="acquisition example"
    )
    conflict_bytes = _canonical_json_bytes(validated_conflict, label="conflict example")
    return PairedImpossibilityInput(
        pair_id=_identifier(pair_id, "paired impossibility pair_id"),
        acquisition_example_bytes=acquisition_bytes,
        acquisition_example_sha256=hashlib.sha256(acquisition_bytes).hexdigest(),
        conflict_example_bytes=conflict_bytes,
        conflict_example_sha256=hashlib.sha256(conflict_bytes).hexdigest(),
        feature_payload_bytes=feature_payload_bytes,
    )


def validate_paired_impossibility_collection(
    value: Any,
    *,
    expected_acquisition_sha256_by_id: Mapping[str, str],
    expected_conflict_sha256_by_id: Mapping[str, str],
    media_access: ObservedMediaAccess,
    query_router: QueryRouter,
    router_sha256: str,
    model_sha256: str,
    config_sha256: str,
) -> tuple[PairedImpossibilityRecord, ...]:
    """Authenticate, rebuild, and compare policy inputs for each hidden pair."""

    items = _collection_items(value, "paired impossibility collection")
    trusted_acquisition = _trusted_digest_map(
        expected_acquisition_sha256_by_id,
        "trusted acquisition digest registry",
    )
    trusted_conflict = _trusted_digest_map(
        expected_conflict_sha256_by_id,
        "trusted conflict digest registry",
    )
    bound_router_sha256 = _sha256(router_sha256, "router_sha256")
    bound_model_sha256 = _sha256(model_sha256, "model_sha256")
    bound_config_sha256 = _sha256(config_sha256, "config_sha256")
    _validate_query_router_bindings(
        query_router,
        router_sha256=bound_router_sha256,
        model_sha256=bound_model_sha256,
        config_sha256=bound_config_sha256,
    )
    score_query = getattr(query_router, "score_query", None)
    if not callable(score_query):
        raise RawIndexValidationError("query_router must provide a score_query method")
    groups: dict[str, dict[str, tuple[tuple[Any, ...], bytes]]] = {}
    group_bindings: dict[str, tuple[Any, ...]] = {}
    raw_example_ids: set[str] = set()
    seen_acquisition_ids: set[str] = set()
    seen_conflict_ids: set[str] = set()
    semantic_answers: dict[tuple[str, str], str] = {}
    component_registry = _ConnectedComponentRegistry()
    output_registry = _OutputIdentityRegistry()
    media_registry = _MediaIdentityRegistry()
    validated: list[PairedImpossibilityRecord] = []
    scored_views: list[tuple[PolicySafeView, tuple[Any, ...]]] = []
    for item in items:
        _exact_dataclass_state(
            item,
            PairedImpossibilityInput,
            "paired impossibility member",
        )
        pair_id = _identifier(item.pair_id, "paired impossibility pair_id")
        acquisition, acquisition_example_sha256 = _parse_raw_example_bytes(
            item.acquisition_example_bytes,
            expected_sha256=item.acquisition_example_sha256,
            label="paired acquisition example",
            validator=validate_acquisition_example,
        )
        conflict, conflict_example_sha256 = _parse_raw_example_bytes(
            item.conflict_example_bytes,
            expected_sha256=item.conflict_example_sha256,
            label="paired conflict example",
            validator=validate_conflict_example,
        )
        acquisition_id = acquisition["example_id"]
        conflict_id = conflict["example_id"]
        if trusted_acquisition.get(acquisition_id) != acquisition_example_sha256:
            raise RawIndexValidationError(
                "paired acquisition bytes differ from the trusted digest registry"
            )
        if trusted_conflict.get(conflict_id) != conflict_example_sha256:
            raise RawIndexValidationError(
                "paired conflict bytes differ from the trusted digest registry"
            )
        seen_acquisition_ids.add(acquisition_id)
        seen_conflict_ids.add(conflict_id)
        if (
            acquisition_id in raw_example_ids
            or conflict_id in raw_example_ids
            or acquisition_id == conflict_id
        ):
            raise RawIndexValidationError(
                "paired impossibility members must use unique raw examples"
            )
        raw_example_ids.update((acquisition_id, conflict_id))
        hidden_donor_role = _validate_paired_raw_binding(acquisition, conflict)
        _register_acquisition_gold_answer(acquisition, semantic_answers)
        _register_conflict_source_answers(conflict, semantic_answers)
        _register_acquisition_graph(acquisition, component_registry)
        _register_conflict_graph(conflict, component_registry)
        output_registry.add_acquisition(acquisition)
        output_registry.add_conflict(conflict)
        media_registry.add_acquisition(acquisition)
        media_registry.add_conflict(conflict)

        feature_payload_sha256 = acquisition["outputs"]["pre_query"][
            "feature_payload_sha256"
        ]
        features = build_observed_policy_features(
            item.feature_payload_bytes,
            expected_payload_sha256=feature_payload_sha256,
        )
        policy_view = build_policy_safe_view(
            acquisition,
            observed_features=features,
            media_access=media_access,
        )
        signature = _policy_invariance_signature(policy_view)
        query_score = score_query(policy_view)
        _validate_query_router_bindings(
            query_router,
            router_sha256=bound_router_sha256,
            model_sha256=bound_model_sha256,
            config_sha256=bound_config_sha256,
        )
        if _policy_invariance_signature(policy_view) != signature:
            raise RawIndexValidationError("policy_view changed during scoring")
        scored_views.append((policy_view, signature))
        _finite_float(query_score, "query_score", maximum=1.0)
        score_bits = _float_bits(query_score, "query_score")
        result_view = _snapshot_policy_view(policy_view)

        pair_binding = (
            acquisition["target"]["video_id"],
            acquisition["target"]["question_id"],
            acquisition["bindings"]["component_id"],
            acquisition["bindings"]["question_key_sha256"],
            acquisition["partitions"]["official"],
            acquisition["partitions"]["analysis"],
        )
        previous_binding = group_bindings.setdefault(pair_id, pair_binding)
        if previous_binding != pair_binding:
            raise RawIndexValidationError(
                f"pair_id {pair_id!r} has inconsistent raw example bindings"
            )
        members = groups.setdefault(pair_id, {})
        if hidden_donor_role in members:
            raise RawIndexValidationError(
                f"pair_id {pair_id!r} must have both hidden donor roles exactly once"
            )
        members[hidden_donor_role] = (signature, score_bits)
        record = PairedImpossibilityRecord(
            pair_id=pair_id,
            hidden_donor_role=hidden_donor_role,
            acquisition_example_id=acquisition_id,
            acquisition_example_sha256=acquisition_example_sha256,
            conflict_example_id=conflict_id,
            conflict_example_sha256=conflict_example_sha256,
            feature_payload_sha256=feature_payload_sha256,
            router_sha256=bound_router_sha256,
            model_sha256=bound_model_sha256,
            config_sha256=bound_config_sha256,
            policy_view=result_view,
            query_score=query_score,
        )
        _exact_dataclass_state(
            record,
            PairedImpossibilityRecord,
            "paired impossibility result",
        )
        validated.append(record)

    if seen_acquisition_ids != set(trusted_acquisition):
        raise RawIndexValidationError(
            "trusted acquisition digest registry does not exactly cover the collection"
        )
    if seen_conflict_ids != set(trusted_conflict):
        raise RawIndexValidationError(
            "trusted conflict digest registry does not exactly cover the collection"
        )
    for record, (scored_view, expected_signature) in zip(validated, scored_views):
        if _policy_invariance_signature(scored_view) != expected_signature:
            raise RawIndexValidationError(
                "a previously scored policy_view changed during collection scoring"
            )
        _exact_dataclass_state(
            record,
            PairedImpossibilityRecord,
            "paired impossibility result",
        )
        if _policy_invariance_signature(record.policy_view) != expected_signature:
            raise RawIndexValidationError("stored policy_view changed after scoring")

    required_roles = {"same_answer", "opposite_answer"}
    for pair_id, members in groups.items():
        if set(members) != required_roles:
            raise RawIndexValidationError(
                f"pair_id {pair_id!r} must have both hidden donor roles exactly once"
            )
        same_signature = members["same_answer"]
        opposite_signature = members["opposite_answer"]
        if same_signature != opposite_signature:
            raise RawIndexValidationError(
                "paired impossibility inputs and query scores must be bitwise identical"
            )
    return tuple(validated)
