"""Deterministic controlled-media edits for Perception Test pilot pairs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from conflictbench.perception_media_pilot import (
    PilotConstructionError,
    validate_media_pilot_attestation,
)
from conflictbench.perception_nuisance_gate import (
    IMPLEMENTATION_SOURCE_ROLES,
    NuisanceValidationError,
    evaluate_post_edit_diagnostics,
    probe_media_file,
    validate_media_receipt,
    validate_nuisance_contract,
)
from conflictbench.perception_omni_gate import (
    GateValidationError,
    verify_gate_output_replay,
)


class MediaEditValidationError(ValueError):
    """Raised when edit construction or validation must stop."""


CONDITIONS = (
    "clean_original",
    "remux_only",
    "same_answer",
    "opposite_answer",
    "missing_source",
    "temporal_shift",
)
SOURCE_ORIENTATIONS = ("audio_over_video", "video_over_audio")
PARTITIONS = ("scorer_fit", "threshold_calibration", "pilot_gate")
PAIR_ROLES = ("same_answer_nuisance", "opposite_answer_candidate")
SOURCE_GATE_IMPLEMENTATION_FILENAMES = {
    "gate": "perception_omni_gate.py",
    "runner": "run_perception_omni_gate.py",
}
TEMPORAL_GENERATOR_REQUEST_SCHEMA = (
    "conflictbench.perception-temporal-generator-request.v1"
)
TEMPORAL_SCORER_REQUEST_SCHEMA = "conflictbench.perception-temporal-scorer-request.v1"
TEMPORAL_SCORER_RESPONSE_SCHEMA = "conflictbench.perception-temporal-scorer-response.v1"
_ROLE_CONDITIONS = {
    "same_answer_nuisance": "same_answer",
    "opposite_answer_candidate": "opposite_answer",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9_.:-]+\Z")
_CONFIG_KEYS = {
    "schema_version",
    "runner",
    "conditions",
    "requested_source_orientations",
    "source_gate_authorization",
    "missing_source_fill",
    "temporal_calibration",
    "audio_encoding",
    "video_encoding",
    "container",
    "validation",
    "human_evaluation",
}
_EDIT_INDEX_KEYS = {
    "schema",
    "status",
    "continuation_allowed",
    "nuisance_implementation_source_sha256",
    "source_gate_implementation_source_sha256",
    "temporal_implementation_source_sha256",
    "input_digests",
    "tool_identities",
    "configuration",
    "counts",
    "output_set_sha256",
    "edits",
    "attestation_sha256",
}
_EDIT_KEYS = {
    "output_id",
    "filename",
    "condition",
    "source_orientation",
    "pair_id",
    "target_id",
    "component_id",
    "partition",
    "key_sha256",
    "target_video_id",
    "donor_video_id",
    "anchor",
    "target",
    "donor",
    "operation",
    "source_files",
    "output",
}
_EDIT_OUTPUT_KEYS = {
    "relative_path",
    "size_bytes",
    "sha256",
    "diagnostic_sha256",
    "profile_sha256",
}
_EDIT_SOURCE_FILE_KEYS = {
    "source_role",
    "video_id",
    "filename",
    "size_bytes",
    "sha256",
}
_POST_EDIT_KEYS = {
    "pair_id",
    "target_id",
    "component_id",
    "partition",
    "key_sha256",
    "target_video_id",
    "output_video_id",
    "role",
    "source_orientation",
}
_VALIDATION_KEYS = {
    "schema",
    "status",
    "continuation_allowed",
    "human_evaluation_used",
    "nuisance_implementation_source_sha256",
    "source_gate_implementation_source_sha256",
    "temporal_implementation_source_sha256",
    "input_digests",
    "checks",
    "nuisance_reports",
    "attestation_sha256",
}
_VALIDATION_CHECK_KEYS = {
    "edit_count",
    "post_edit_pair_count",
    "target_anchor_integrity",
    "component_partition_integrity",
    "source_hash_integrity",
    "output_hash_integrity",
    "stream_and_duration_integrity",
}
_RUN_KEYS = {
    "schema",
    "status",
    "continuation_allowed",
    "nuisance_implementation_source_sha256",
    "source_gate_implementation_source_sha256",
    "temporal_implementation_source_sha256",
    "input_digests",
    "tool_identities",
    "implementation_source_sha256",
    "counts",
    "output_digests",
    "attestation_sha256",
}
_TOOL_IDENTITY_KEYS = {"filename", "sha256", "version"}
_RUN_COUNT_KEYS = {
    "edit_count",
    "post_edit_pair_count",
    "source_video_count",
    "target_count",
    "clean_original_count",
}
_EDIT_COUNT_KEYS = {"target_count", "edit_count", "clean_original_count"}
_RUN_OUTPUT_DIGEST_KEYS = {
    "edit_index_sha256",
    "post_edit_index_sha256",
    "validation_sha256",
    "output_set_sha256",
}
_INPUT_DIGEST_KEYS = {
    "configuration_sha256",
    "pilot_index_sha256",
    "media_receipt_sha256",
    "media_set_sha256",
    "nuisance_configuration_sha256",
    "nuisance_implementation_bundle_sha256",
    "nuisance_contract_sha256",
    "source_gate_configuration_sha256",
    "source_gate_output_sha256",
    "source_gate_implementation_bundle_sha256",
    "source_gate_transcript_sha256",
    "temporal_implementation_bundle_sha256",
    "temporal_calibration_sha256",
}
_VALIDATION_INPUT_DIGEST_KEYS = _INPUT_DIGEST_KEYS | {
    "edit_index_sha256",
    "post_edit_index_sha256",
    "output_set_sha256",
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MediaEditValidationError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise MediaEditValidationError(
            f"{label} fields differ; "
            f"missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MediaEditValidationError(f"{label} must be nonempty text")
    return value.strip()


def _canonical_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _identifier(value: Any, label: str) -> str:
    text = _text(value, label)
    if _IDENTIFIER_RE.fullmatch(text) is None:
        raise MediaEditValidationError(f"{label} contains unsafe characters")
    return text


def _sha256_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MediaEditValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MediaEditValidationError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MediaEditValidationError(f"{label} must be a nonnegative integer")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise MediaEditValidationError(f"{label} must be a boolean")
    return value


def _exact_positive_int(value: Any, expected: int, label: str) -> int:
    number = _positive_int(value, label)
    if number != expected:
        raise MediaEditValidationError(f"{label} differs")
    return number


def _positive_int_text(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise MediaEditValidationError(f"{label} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise MediaEditValidationError(f"{label} must be a positive integer") from exc
    return _positive_int(number, label)


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MediaEditValidationError(f"{label} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise MediaEditValidationError(f"{label} must be a positive finite number")
    return number


def _positive_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise MediaEditValidationError(
            f"{label} must be a positive finite float (finite number)"
        )
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MediaEditValidationError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise MediaEditValidationError(f"{label} must be a finite number")
    return number


def _bounded_number(value: Any, label: str) -> float:
    number = _finite_number(value, label)
    if number < 0.0 or number > 1.0:
        raise MediaEditValidationError(f"{label} must be between zero and one")
    return number


def _same_json_types_and_values(observed: Any, expected: Any) -> bool:
    """Compare decoded JSON without Python's bool/int and int/float coercion."""

    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            _same_json_types_and_values(observed[key], expected[key])
            for key in expected
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _same_json_types_and_values(left, right)
            for left, right in zip(observed, expected)
        )
    return observed == expected


def _canonical_bytes(value: Any, *, pretty: bool = False) -> bytes:
    try:
        if pretty:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            return (encoded + "\n").encode("utf-8")
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MediaEditValidationError("value is not strict JSON") from exc


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise MediaEditValidationError(
            f"required file cannot be read: {path.name}"
        ) from exc
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise MediaEditValidationError(f"non-finite JSON number is forbidden: {value}")


def _unique_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise MediaEditValidationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _decode_json_bytes(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MediaEditValidationError(f"{label} is not strict UTF-8 JSON") from exc


def _load_locked_json(
    path_value: pathlib.Path, expected_sha256: str, label: str
) -> tuple[bytes, Any]:
    path = pathlib.Path(path_value)
    if path.is_symlink() or not path.is_file():
        raise MediaEditValidationError(f"{label} must be a regular non-symlink file")
    expected = _sha256_text(expected_sha256, f"expected {label} digest")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise MediaEditValidationError(f"{label} cannot be read") from exc
    if hashlib.sha256(payload).hexdigest() != expected:
        raise MediaEditValidationError(f"{label} digest differs")
    value = _decode_json_bytes(payload, label)
    return payload, value


def _read_json(path: pathlib.Path, label: str) -> tuple[bytes, Any]:
    if path.is_symlink() or not path.is_file():
        raise MediaEditValidationError(f"{label} must be a regular non-symlink file")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise MediaEditValidationError(f"{label} cannot be read") from exc
    value = _decode_json_bytes(payload, label)
    return payload, value


def _validate_configuration(value: Any) -> dict[str, Any]:
    config = _mapping(value, "configuration")
    _exact_keys(config, _CONFIG_KEYS, "configuration")
    _exact_positive_int(config.get("schema_version"), 4, "configuration schema version")
    if config.get("runner") != "perception_test_controlled_media_edits":
        raise MediaEditValidationError("configuration runner differs")
    if config.get("conditions") != list(CONDITIONS):
        raise MediaEditValidationError("configuration conditions or order differ")
    requested = config.get("requested_source_orientations")
    if (
        not isinstance(requested, list)
        or not requested
        or requested != list(dict.fromkeys(requested))
        or any(item not in SOURCE_ORIENTATIONS for item in requested)
    ):
        raise MediaEditValidationError("requested source orientations are invalid")
    source_gate = _mapping(
        config.get("source_gate_authorization"), "source-gate authorization"
    )
    _exact_keys(
        source_gate,
        {
            "verification",
            "required_source_sufficiency_status",
            "required_nuisance_detection_status",
            "required_overall_pilot_status",
            "orientation_conditions",
        },
        "source-gate authorization",
    )
    expected_source_gate = {
        "verification": "exact_output_from_authenticated_transcript_replay",
        "required_source_sufficiency_status": "pass",
        "required_nuisance_detection_status": "not_run",
        "required_overall_pilot_status": "pending_nuisance_detection",
        "orientation_conditions": {
            "audio_over_video": "audio_only",
            "video_over_audio": "video_only",
        },
    }
    if not _same_json_types_and_values(dict(source_gate), expected_source_gate):
        raise MediaEditValidationError("source-gate authorization contract differs")
    fill = _mapping(config.get("missing_source_fill"), "missing-source fill")
    _exact_keys(fill, {"audio", "video"}, "missing-source fill")
    if not _same_json_types_and_values(
        dict(fill), {"audio": "silence", "video": "black"}
    ):
        raise MediaEditValidationError("missing-source fill profile differs")
    temporal = _mapping(config.get("temporal_calibration"), "temporal calibration")
    _exact_keys(
        temporal,
        {
            "record_schema",
            "replay_protocol",
            "selection_rule",
            "event_scoring_rule",
            "implementation_roles",
            "candidate_shifts_seconds",
            "official_split",
            "calibration_partition",
        },
        "temporal calibration",
    )
    expected_temporal = {
        "record_schema": "conflictbench.perception-temporal-shift-events.v2",
        "replay_protocol": "pinned_python_json_cli_v1",
        "selection_rule": (
            "maximum_mean_absolute_score_change_then_smallest_shift_seconds"
        ),
        "event_scoring_rule": "absolute_reference_minus_shifted",
        "implementation_roles": ["generator", "scorer"],
        "candidate_shifts_seconds": [0.25, 0.5, 1.0],
        "official_split": "train",
        "calibration_partition": "scorer_fit",
    }
    if not _same_json_types_and_values(dict(temporal), expected_temporal):
        raise MediaEditValidationError("temporal calibration contract differs")

    audio = _mapping(config.get("audio_encoding"), "audio encoding")
    _exact_keys(
        audio,
        {
            "codec",
            "bitrate",
            "bitrate_tolerance_fraction",
            "sample_rate",
            "channels",
            "sample_format",
        },
        "audio encoding",
    )
    if audio.get("codec") != "aac":
        raise MediaEditValidationError("audio codec must be aac")
    bitrate = _text(audio.get("bitrate"), "audio bitrate")
    if re.fullmatch(r"[1-9][0-9]*k", bitrate) is None:
        raise MediaEditValidationError("audio bitrate is invalid")
    sample_rate = _positive_int(audio.get("sample_rate"), "audio sample rate")
    channels = _positive_int(audio.get("channels"), "audio channel count")
    if channels not in {1, 2}:
        raise MediaEditValidationError("audio channel count must be one or two")
    bitrate_tolerance = _positive_number(
        audio.get("bitrate_tolerance_fraction"), "audio bitrate tolerance"
    )
    if bitrate_tolerance >= 1.0:
        raise MediaEditValidationError("audio bitrate tolerance must be below one")
    if audio.get("sample_format") != "fltp":
        raise MediaEditValidationError("audio sample format must be fltp")

    video = _mapping(config.get("video_encoding"), "video encoding")
    _exact_keys(
        video,
        {"codec", "encoder", "pixel_format", "preset", "crf"},
        "video encoding",
    )
    _exact_positive_int(video.get("crf"), 18, "video CRF")
    if not _same_json_types_and_values(
        dict(video),
        {
            "codec": "h264",
            "encoder": "libx264",
            "pixel_format": "yuv420p",
            "preset": "medium",
            "crf": 18,
        },
    ):
        raise MediaEditValidationError("video encoding profile differs")

    container = _mapping(config.get("container"), "container")
    _exact_keys(
        container,
        {"format", "probe_format_names", "strip_metadata"},
        "container",
    )
    _boolean(container.get("strip_metadata"), "container strip-metadata setting")
    if not _same_json_types_and_values(
        dict(container),
        {
            "format": "mp4",
            "probe_format_names": ["mov", "mp4", "m4a", "3gp", "3g2", "mj2"],
            "strip_metadata": True,
        },
    ):
        raise MediaEditValidationError(
            "container settings differ from the fixed profile"
        )

    validation = _mapping(config.get("validation"), "validation")
    _exact_keys(
        validation,
        {
            "duration_tolerance_seconds",
            "ffmpeg_timeout_seconds",
            "require_exact_stream_types",
            "reject_extra_streams",
        },
        "validation",
    )
    tolerance = _positive_float(
        validation.get("duration_tolerance_seconds"), "duration tolerance"
    )
    timeout = _positive_int(validation.get("ffmpeg_timeout_seconds"), "ffmpeg timeout")
    if validation.get("require_exact_stream_types") != ["audio", "video"]:
        raise MediaEditValidationError("exactly one audio and video stream is required")
    if (
        _boolean(
            validation.get("reject_extra_streams"), "extra-stream rejection setting"
        )
        is not True
    ):
        raise MediaEditValidationError("extra streams must be rejected")
    if config.get("human_evaluation") != "forbidden":
        raise MediaEditValidationError("configuration permits human evaluation")
    return {
        **dict(config),
        "requested_source_orientations": list(requested),
        "source_gate_authorization": expected_source_gate,
        "missing_source_fill": dict(fill),
        "temporal_calibration": expected_temporal,
        "audio_encoding": {
            "codec": "aac",
            "bitrate": bitrate,
            "bitrate_tolerance_fraction": bitrate_tolerance,
            "sample_rate": sample_rate,
            "channels": channels,
            "sample_format": "fltp",
        },
        "video_encoding": dict(video),
        "container": dict(container),
        "validation": {
            **dict(validation),
            "duration_tolerance_seconds": tolerance,
            "ffmpeg_timeout_seconds": timeout,
        },
    }


def _validate_source_entry(value: Any, label: str) -> dict[str, Any]:
    source = _mapping(value, label)
    _exact_keys(
        source,
        {"video_id", "question_id", "question", "options", "answer_id", "answer"},
        label,
    )
    video_id = _identifier(source.get("video_id"), f"{label} video ID")
    question_id = source.get("question_id")
    if (
        isinstance(question_id, bool)
        or not isinstance(question_id, int)
        or question_id < 0
    ):
        raise MediaEditValidationError(f"{label} question ID is invalid")
    question = _text(source.get("question"), f"{label} question")
    options_value = source.get("options")
    if not isinstance(options_value, list) or len(options_value) != 3:
        raise MediaEditValidationError(f"{label} options must contain three strings")
    options = [_text(option, f"{label} option") for option in options_value]
    if len(set(options)) != len(options):
        raise MediaEditValidationError(f"{label} options must be unique")
    if len({_canonical_text(option) for option in options}) != len(options):
        raise MediaEditValidationError(
            f"{label} options must be unique after text normalization"
        )
    answer_id = source.get("answer_id")
    if (
        isinstance(answer_id, bool)
        or not isinstance(answer_id, int)
        or answer_id not in range(3)
    ):
        raise MediaEditValidationError(f"{label} answer ID is invalid")
    answer = _text(source.get("answer"), f"{label} answer")
    if answer != options[answer_id]:
        raise MediaEditValidationError(f"{label} answer does not match its option")
    return {
        "video_id": video_id,
        "question_id": question_id,
        "question": question,
        "options": options,
        "answer_id": answer_id,
        "answer": answer,
    }


def _validate_pilot(value: Any) -> tuple[list[dict[str, Any]], set[str]]:
    pilot = _mapping(value, "pilot index")
    _exact_positive_int(pilot.get("schema_version"), 1, "pilot-index schema version")
    try:
        validate_media_pilot_attestation(pilot)
    except (PilotConstructionError, TypeError, ValueError) as exc:
        raise MediaEditValidationError("pilot-index attestation differs") from exc

    components_value = pilot.get("components")
    if not isinstance(components_value, list) or not components_value:
        raise MediaEditValidationError("pilot components must be a nonempty list")
    components: dict[str, dict[str, Any]] = {}
    video_components: dict[str, tuple[str, str]] = {}
    for index, raw in enumerate(components_value):
        component = _mapping(raw, f"component {index}")
        _exact_keys(
            component,
            {"component_id", "partition", "target_count", "video_ids"},
            f"component {index}",
        )
        component_id = _identifier(component.get("component_id"), "component ID")
        if component_id in components:
            raise MediaEditValidationError("component IDs must be unique")
        partition = component.get("partition")
        if partition not in PARTITIONS:
            raise MediaEditValidationError("component partition differs")
        target_count = _positive_int(
            component.get("target_count"), "component target count"
        )
        video_ids_value = component.get("video_ids")
        if not isinstance(video_ids_value, list) or not video_ids_value:
            raise MediaEditValidationError("component video IDs must be nonempty")
        video_ids = [
            _identifier(item, "component video ID") for item in video_ids_value
        ]
        if len(set(video_ids)) != len(video_ids):
            raise MediaEditValidationError("component video IDs must be unique")
        for video_id in video_ids:
            if video_id in video_components:
                raise MediaEditValidationError(
                    "one video appears in more than one connected component or partition"
                )
            video_components[video_id] = (component_id, partition)
        components[component_id] = {
            "component_id": component_id,
            "partition": partition,
            "target_count": target_count,
            "video_ids": video_ids,
        }

    pairs_value = pilot.get("candidate_index")
    if not isinstance(pairs_value, list) or not pairs_value:
        raise MediaEditValidationError("pilot candidate index must be a nonempty list")
    pairs: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    targets: dict[str, list[dict[str, Any]]] = {}
    target_components: dict[str, str] = {}
    for index, raw in enumerate(pairs_value):
        pair = _mapping(raw, f"candidate pair {index}")
        _exact_keys(
            pair,
            {
                "pair_id",
                "component_id",
                "official_split",
                "partition",
                "key_sha256",
                "role",
                "anchor",
                "target",
                "donor",
            },
            f"candidate pair {index}",
        )
        pair_id = _identifier(pair.get("pair_id"), "pair ID")
        if pair_id in pair_ids:
            raise MediaEditValidationError("pair IDs must be unique")
        pair_ids.add(pair_id)
        if pair.get("official_split") != "train":
            raise MediaEditValidationError("candidate pair is outside the train split")
        partition = pair.get("partition")
        if partition not in PARTITIONS:
            raise MediaEditValidationError("candidate pair partition differs")
        component_id = _identifier(pair.get("component_id"), "pair component ID")
        if (
            component_id not in components
            or components[component_id]["partition"] != partition
        ):
            raise MediaEditValidationError(
                "candidate pair component or partition differs"
            )
        role = pair.get("role")
        if role not in PAIR_ROLES:
            raise MediaEditValidationError("candidate pair role differs")
        key_sha256 = _sha256_text(pair.get("key_sha256"), "question-key digest")
        target = _validate_source_entry(pair.get("target"), "candidate target")
        donor = _validate_source_entry(pair.get("donor"), "candidate donor")
        if target["video_id"] == donor["video_id"]:
            raise MediaEditValidationError("candidate target and donor must differ")
        for source in (target, donor):
            if video_components.get(source["video_id"]) != (component_id, partition):
                raise MediaEditValidationError(
                    "candidate source crosses its connected component or partition"
                )
        anchor = _mapping(pair.get("anchor"), "candidate anchor")
        _exact_keys(
            anchor, {"question", "options", "answer_id", "answer"}, "candidate anchor"
        )
        _validate_anchor_schema(anchor, "candidate anchor")
        expected_anchor = {
            "question": target["question"],
            "options": target["options"],
            "answer_id": target["answer_id"],
            "answer": target["answer"],
        }
        if dict(anchor) != expected_anchor:
            raise MediaEditValidationError("candidate question/answer anchor differs")
        if _canonical_text(donor["question"]) != _canonical_text(target["question"]):
            raise MediaEditValidationError("candidate donor question differs")
        if {_canonical_text(option) for option in donor["options"]} != {
            _canonical_text(option) for option in target["options"]
        }:
            raise MediaEditValidationError("candidate donor question/options differ")
        if role == "same_answer_nuisance" and _canonical_text(
            donor["answer"]
        ) != _canonical_text(target["answer"]):
            raise MediaEditValidationError("same-answer donor answer differs")
        if role == "opposite_answer_candidate" and _canonical_text(
            donor["answer"]
        ) == _canonical_text(target["answer"]):
            raise MediaEditValidationError(
                "opposite-answer donor answer does not differ"
            )
        target_id = f"{target['video_id']}:{target['question_id']}"
        prior_component = target_components.setdefault(target_id, component_id)
        if prior_component != component_id:
            raise MediaEditValidationError("one target crosses connected components")
        normalized = {
            "pair_id": pair_id,
            "component_id": component_id,
            "partition": partition,
            "key_sha256": key_sha256,
            "role": role,
            "anchor": expected_anchor,
            "target": target,
            "donor": donor,
            "target_id": target_id,
        }
        pairs.append(normalized)
        targets.setdefault(target_id, []).append(normalized)

    observed_target_count: dict[str, int] = {
        component_id: 0 for component_id in components
    }
    for target_id, target_pairs in targets.items():
        if len(target_pairs) != 2 or {item["role"] for item in target_pairs} != set(
            PAIR_ROLES
        ):
            raise MediaEditValidationError(
                f"target {target_id} must have exactly one pair for each role"
            )
        for key in ("component_id", "partition", "key_sha256", "anchor", "target"):
            if any(item[key] != target_pairs[0][key] for item in target_pairs[1:]):
                raise MediaEditValidationError(
                    f"target {target_id} has inconsistent {key}"
                )
        observed_target_count[target_pairs[0]["component_id"]] += 1
    for component_id, component in components.items():
        if observed_target_count[component_id] != component["target_count"]:
            raise MediaEditValidationError("component target count differs")
    media_ids = {
        source["video_id"]
        for pair in pairs
        for source in (pair["target"], pair["donor"])
    }
    if media_ids != set(video_components):
        raise MediaEditValidationError("connected-component media membership differs")
    pairs.sort(
        key=lambda item: (item["partition"], item["component_id"], item["pair_id"])
    )
    return pairs, media_ids


def _media_map(
    value: Sequence[Mapping[str, Any]], expected_ids: set[str]
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MediaEditValidationError("media records must be a sequence")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(value):
        record = _mapping(raw, f"media record {index}")
        _exact_keys(
            record,
            {"video_id", "filename", "size_bytes", "sha256", "stream_types"},
            f"media record {index}",
        )
        video_id = _identifier(record.get("video_id"), "media video ID")
        filename = _text(record.get("filename"), "media filename")
        if (
            pathlib.PurePath(filename).name != filename
            or pathlib.Path(filename).stem != video_id
        ):
            raise MediaEditValidationError("media filename differs from its video ID")
        if video_id in result:
            raise MediaEditValidationError("media video IDs must be unique")
        if record.get("stream_types") != ["audio", "video"]:
            raise MediaEditValidationError("media source stream types differ")
        result[video_id] = {
            "video_id": video_id,
            "filename": filename,
            "size_bytes": _positive_int(record.get("size_bytes"), "media size"),
            "sha256": _sha256_text(record.get("sha256"), "media digest"),
            "stream_types": ["audio", "video"],
        }
    if set(result) != expected_ids:
        raise MediaEditValidationError("media records differ from pilot membership")
    return result


def _validate_media_receipt_scalar_types(value: Any) -> None:
    receipt = _mapping(value, "media receipt")
    _positive_int(receipt.get("media_file_count"), "media file count")
    files = receipt.get("media_files")
    if not isinstance(files, list) or not files:
        raise MediaEditValidationError("media receipt file list must be nonempty")
    for index, raw_file in enumerate(files):
        item = _mapping(raw_file, f"media receipt file {index}")
        _positive_int(item.get("size_bytes"), f"media receipt file {index} size")
        _boolean(
            item.get("resumed_existing"),
            f"media receipt file {index} resumed-existing flag",
        )


def _source_file(record: Mapping[str, Any], source_role: str) -> dict[str, Any]:
    return {
        "source_role": source_role,
        "video_id": record["video_id"],
        "filename": record["filename"],
        "size_bytes": record["size_bytes"],
        "sha256": record["sha256"],
    }


def build_edit_plan(
    pilot_value: Any,
    media_records: Sequence[Mapping[str, Any]],
    configuration_value: Any,
    *,
    selected_shift_seconds: float,
    authorized_orientations: Sequence[str],
) -> list[dict[str, Any]]:
    """Build the complete deterministic edit plan without touching media."""

    config = _validate_configuration(configuration_value)
    shift_seconds = _positive_number(selected_shift_seconds, "selected temporal shift")
    if not isinstance(authorized_orientations, Sequence) or isinstance(
        authorized_orientations, (str, bytes)
    ):
        raise MediaEditValidationError(
            "authorized source orientations must be a sequence"
        )
    allowed = list(authorized_orientations)
    if (
        not allowed
        or allowed != list(dict.fromkeys(allowed))
        or any(item not in SOURCE_ORIENTATIONS for item in allowed)
    ):
        raise MediaEditValidationError("authorized source orientations are invalid")
    orientations = [
        orientation
        for orientation in config["requested_source_orientations"]
        if orientation in allowed
    ]
    if not orientations:
        raise MediaEditValidationError("no requested source orientation is authorized")
    pairs, expected_media_ids = _validate_pilot(pilot_value)
    media = _media_map(media_records, expected_media_ids)
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for pair in pairs:
        grouped.setdefault(pair["target_id"], {})[pair["role"]] = pair
    condition_rank = {condition: index for index, condition in enumerate(CONDITIONS)}
    config_sha256 = _canonical_digest(config)
    edits: list[dict[str, Any]] = []
    for target_id in sorted(grouped):
        role_pairs = grouped[target_id]
        same_pair = role_pairs["same_answer_nuisance"]
        opposite_pair = role_pairs["opposite_answer_candidate"]
        base = same_pair
        target_record = media[base["target"]["video_id"]]
        specifications: list[
            tuple[
                str,
                str | None,
                dict[str, Any] | None,
                dict[str, Any] | None,
                list[dict[str, Any]],
            ]
        ] = [
            (
                "clean_original",
                None,
                None,
                None,
                [_source_file(target_record, "target")],
            ),
            (
                "remux_only",
                None,
                None,
                None,
                [_source_file(target_record, "target")],
            ),
        ]
        for orientation in orientations:
            specifications.extend(
                [
                    (
                        "same_answer",
                        orientation,
                        same_pair,
                        same_pair["donor"],
                        [
                            _source_file(target_record, "target"),
                            _source_file(
                                media[same_pair["donor"]["video_id"]], "donor"
                            ),
                        ],
                    ),
                    (
                        "opposite_answer",
                        orientation,
                        opposite_pair,
                        opposite_pair["donor"],
                        [
                            _source_file(target_record, "target"),
                            _source_file(
                                media[opposite_pair["donor"]["video_id"]], "donor"
                            ),
                        ],
                    ),
                    (
                        "missing_source",
                        orientation,
                        None,
                        None,
                        [_source_file(target_record, "target")],
                    ),
                    (
                        "temporal_shift",
                        orientation,
                        None,
                        None,
                        [_source_file(target_record, "target")],
                    ),
                ]
            )
        for condition, orientation, pair, donor, source_files in specifications:
            identity = {
                "condition": condition,
                "source_orientation": orientation,
                "configuration_sha256": config_sha256,
                "selected_shift_seconds": shift_seconds,
                "target_id": target_id,
                "component_id": base["component_id"],
                "partition": base["partition"],
                "key_sha256": base["key_sha256"],
                "pair_id": pair["pair_id"] if pair is not None else None,
                "source_sha256": [source["sha256"] for source in source_files],
            }
            output_id = f"edit_{_canonical_digest(identity)[:24]}"
            if condition in {"clean_original", "remux_only"}:
                operation = {"audio_source": "target", "video_source": "target"}
            elif condition in {"same_answer", "opposite_answer"}:
                if orientation == "audio_over_video":
                    operation = {"audio_source": "donor", "video_source": "target"}
                else:
                    operation = {"audio_source": "target", "video_source": "donor"}
            elif condition == "missing_source":
                operation = (
                    {
                        "audio_source": "synthetic_silence",
                        "video_source": "target",
                    }
                    if orientation == "audio_over_video"
                    else {
                        "audio_source": "target",
                        "video_source": "synthetic_black",
                    }
                )
            else:
                operation = {"audio_source": "target", "video_source": "target"}
                if orientation == "audio_over_video":
                    operation["audio_delay_seconds"] = shift_seconds
                else:
                    operation["video_delay_seconds"] = shift_seconds
            edits.append(
                {
                    "output_id": output_id,
                    "filename": f"{output_id}.mp4",
                    "condition": condition,
                    "source_orientation": orientation,
                    "pair_id": pair["pair_id"] if pair is not None else None,
                    "target_id": target_id,
                    "component_id": base["component_id"],
                    "partition": base["partition"],
                    "key_sha256": base["key_sha256"],
                    "target_video_id": base["target"]["video_id"],
                    "donor_video_id": donor["video_id"] if donor is not None else None,
                    "anchor": dict(base["anchor"]),
                    "target": dict(base["target"]),
                    "donor": dict(donor) if donor is not None else None,
                    "operation": operation,
                    "source_files": source_files,
                }
            )
    edits.sort(
        key=lambda item: (
            item["partition"],
            item["component_id"],
            item["target_id"],
            condition_rank[item["condition"]],
            SOURCE_ORIENTATIONS.index(item["source_orientation"])
            if item["source_orientation"] is not None
            else -1,
        )
    )
    if len({edit["output_id"] for edit in edits}) != len(edits):
        raise MediaEditValidationError("edit output IDs are not unique")
    expected_edits_per_target = 2 + 4 * len(orientations)
    if len(edits) != len(grouped) * expected_edits_per_target:
        raise MediaEditValidationError("edit-plan condition coverage differs")
    return edits


def _tool_identity(
    path_value: pathlib.Path, expected_name: str
) -> tuple[pathlib.Path, dict[str, str]]:
    path = pathlib.Path(path_value)
    if path.is_symlink() or not path.is_file() or not os.access(path, os.X_OK):
        raise MediaEditValidationError(
            f"{expected_name} must be an executable regular non-symlink file"
        )
    path = path.resolve(strict=True)
    if path.name != expected_name:
        raise MediaEditValidationError(
            f"explicit {expected_name} path has a different filename"
        )
    try:
        completed = subprocess.run(
            [str(path), "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaEditValidationError(
            f"{expected_name} identity check failed"
        ) from exc
    first_line = completed.stdout.splitlines()[0] if completed.stdout else ""
    if completed.returncode != 0 or not first_line or len(first_line) > 500:
        raise MediaEditValidationError(f"{expected_name} identity check failed")
    return path, {
        "filename": expected_name,
        "sha256": _sha256_path(path),
        "version": first_line,
    }


def _probe_sources(
    *,
    ffprobe_path: pathlib.Path,
    media_root: pathlib.Path,
    media_records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    diagnostics = {}
    for record in media_records:
        video_id = str(record["video_id"])
        diagnostics[video_id] = probe_media_file(
            ffprobe_path,
            media_root / str(record["filename"]),
            expected_size_bytes=int(record["size_bytes"]),
        )
    return diagnostics


def _probe_media_profile(
    ffprobe_path: pathlib.Path,
    media_path: pathlib.Path,
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                str(ffprobe_path),
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(media_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaEditValidationError("ffprobe profile check failed") from exc
    if completed.returncode != 0 or len(completed.stdout.encode("utf-8")) > 5_000_000:
        raise MediaEditValidationError(
            "ffprobe profile response is missing or too large"
        )
    try:
        payload = json.loads(
            completed.stdout,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise MediaEditValidationError("ffprobe profile response is invalid") from exc
    record = _mapping(payload, "ffprobe profile")
    if set(record) != {"format", "streams"}:
        raise MediaEditValidationError("ffprobe profile fields differ")
    streams = record.get("streams")
    if not isinstance(streams, list):
        raise MediaEditValidationError("ffprobe profile streams must be a list")
    stream_types = [
        stream.get("codec_type") if isinstance(stream, Mapping) else None
        for stream in streams
    ]
    expected_types = configuration["validation"]["require_exact_stream_types"]
    if sorted(stream_types) != sorted(expected_types) or len(stream_types) != len(
        expected_types
    ):
        raise MediaEditValidationError("edited-media stream membership differs")
    by_type = {
        str(stream["codec_type"]): _mapping(stream, "ffprobe stream")
        for stream in streams
    }
    format_value = _mapping(record.get("format"), "ffprobe format")
    format_names = _text(format_value.get("format_name"), "container format").split(",")
    if format_names != configuration["container"]["probe_format_names"]:
        raise MediaEditValidationError("edited-media container format differs")
    audio = by_type["audio"]
    video = by_type["video"]
    return {
        "container_format_names": format_names,
        "audio": {
            "codec": _text(audio.get("codec_name"), "audio codec"),
            "bitrate": _positive_int_text(audio.get("bit_rate"), "audio bitrate"),
            "sample_rate": _positive_int_text(
                audio.get("sample_rate"), "audio sample rate"
            ),
            "channels": _positive_int(audio.get("channels"), "audio channel count"),
            "sample_format": _text(audio.get("sample_fmt"), "audio sample format"),
        },
        "video": {
            "codec": _text(video.get("codec_name"), "video codec"),
            "bitrate": _positive_int_text(video.get("bit_rate"), "video bitrate"),
            "pixel_format": _text(video.get("pix_fmt"), "video pixel format"),
            "width": _positive_int(video.get("width"), "video width"),
            "height": _positive_int(video.get("height"), "video height"),
            "frame_rate": _text(video.get("avg_frame_rate"), "video frame rate"),
        },
    }


def _probe_source_profiles(
    *,
    ffprobe_path: pathlib.Path,
    media_root: pathlib.Path,
    media_records: Sequence[Mapping[str, Any]],
    configuration: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        str(record["video_id"]): _probe_media_profile(
            ffprobe_path,
            media_root / str(record["filename"]),
            configuration,
        )
        for record in media_records
    }


def _validate_output_profile(
    profile: Mapping[str, Any],
    *,
    target_profile: Mapping[str, Any],
    condition: str,
    source_orientation: str | None,
    target_stream_padded: bool,
    configuration: Mapping[str, Any],
) -> None:
    if (
        profile.get("container_format_names")
        != configuration["container"]["probe_format_names"]
    ):
        raise MediaEditValidationError("edited-media container format differs")
    output_audio = _mapping(profile.get("audio"), "output audio profile")
    output_video = _mapping(profile.get("video"), "output video profile")
    target_audio = _mapping(target_profile.get("audio"), "target audio profile")
    target_video = _mapping(target_profile.get("video"), "target video profile")
    edited_condition = condition not in {"clean_original", "remux_only"}
    paired_condition = condition in {"same_answer", "opposite_answer"}
    audio_is_encoded = edited_condition and (
        source_orientation == "audio_over_video"
        or (
            paired_condition
            and source_orientation == "video_over_audio"
            and target_stream_padded
        )
    )
    video_is_encoded = edited_condition and (
        source_orientation == "video_over_audio"
        or (
            paired_condition
            and source_orientation == "audio_over_video"
            and target_stream_padded
        )
    )
    if audio_is_encoded:
        expected_audio = configuration["audio_encoding"]
        for field in ("codec", "sample_rate", "channels", "sample_format"):
            if output_audio.get(field) != expected_audio[field]:
                raise MediaEditValidationError(
                    f"edited audio {field.replace('_', ' ')} differs"
                )
        requested_bitrate = int(str(expected_audio["bitrate"])[:-1]) * 1000
        tolerance = float(expected_audio["bitrate_tolerance_fraction"])
        observed_bitrate = _positive_int(output_audio.get("bitrate"), "audio bitrate")
        if abs(observed_bitrate - requested_bitrate) / requested_bitrate > tolerance:
            raise MediaEditValidationError("edited audio bitrate differs")
    else:
        for field in ("codec", "bitrate", "sample_rate", "channels", "sample_format"):
            if output_audio.get(field) != target_audio.get(field):
                raise MediaEditValidationError(
                    f"copied audio {field.replace('_', ' ')} differs from target"
                )
    if video_is_encoded:
        expected_video = configuration["video_encoding"]
        for field in ("codec", "pixel_format"):
            if output_video.get(field) != expected_video[field]:
                raise MediaEditValidationError(
                    f"edited video {field.replace('_', ' ')} differs"
                )
        for field in ("width", "height", "frame_rate"):
            if output_video.get(field) != target_video.get(field):
                raise MediaEditValidationError(
                    f"edited video {field.replace('_', ' ')} differs from target"
                )
        _positive_int(output_video.get("bitrate"), "video bitrate")
    else:
        for field in (
            "codec",
            "bitrate",
            "pixel_format",
            "width",
            "height",
            "frame_rate",
        ):
            if output_video.get(field) != target_video.get(field):
                raise MediaEditValidationError(
                    f"copied video {field.replace('_', ' ')} differs from target"
                )


def _duration_text(value: float) -> str:
    return f"{value:.6f}"


def _output_duration_for_edit(
    edit: Mapping[str, Any],
    source_diagnostics: Mapping[str, Mapping[str, Any]],
) -> float:
    """Return the locked v4 duration without truncating paired donor media."""

    target_id = _identifier(edit.get("target_video_id"), "edit target video ID")
    if target_id not in source_diagnostics:
        raise MediaEditValidationError("edit target diagnostics are missing")
    target_duration = _positive_number(
        source_diagnostics[target_id].get("duration_seconds"), "target duration"
    )
    if edit.get("condition") not in {"same_answer", "opposite_answer"}:
        return target_duration
    donor_id = _identifier(edit.get("donor_video_id"), "edit donor video ID")
    if donor_id not in source_diagnostics:
        raise MediaEditValidationError("edit donor diagnostics are missing")
    donor_duration = _positive_number(
        source_diagnostics[donor_id].get("duration_seconds"), "donor duration"
    )
    return max(target_duration, donor_duration)


def _target_stream_needs_padding(
    edit: Mapping[str, Any],
    target_diagnostic: Mapping[str, Any],
    output_duration: float,
) -> bool:
    if edit.get("condition") not in {"same_answer", "opposite_answer"}:
        return False
    orientation = edit.get("source_orientation")
    if orientation == "audio_over_video":
        stream_name = "video"
    elif orientation == "video_over_audio":
        stream_name = "audio"
    else:
        raise MediaEditValidationError("paired edit source orientation differs")
    stream = _mapping(
        target_diagnostic.get(stream_name), f"target {stream_name} diagnostics"
    )
    stream_duration = _positive_number(
        stream.get("duration_seconds"), f"target {stream_name} duration"
    )
    return stream_duration < _positive_number(output_duration, "output duration")


def _ffmpeg_arguments(
    *,
    ffmpeg_path: pathlib.Path,
    edit: Mapping[str, Any],
    media_root: pathlib.Path,
    output_path: pathlib.Path,
    target_duration: float,
    target_stream_duration: float,
    donor_duration: float | None,
    output_duration: float,
    target_diagnostic: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> list[str]:
    audio = configuration["audio_encoding"]
    video = configuration["video_encoding"]
    target_path = media_root / edit["source_files"][0]["filename"]
    common = [
        str(ffmpeg_path),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-i",
        str(target_path),
    ]
    target_duration = _positive_number(target_duration, "target duration")
    target_stream_duration = _positive_number(
        target_stream_duration, "target selected-stream duration"
    )
    output_duration = _positive_number(output_duration, "output duration")
    condition = edit["condition"]
    orientation = edit["source_orientation"]
    if condition in {"same_answer", "opposite_answer"}:
        donor_duration = _positive_number(donor_duration, "donor duration")
        expected_duration = max(target_duration, donor_duration)
    else:
        if donor_duration is not None:
            raise MediaEditValidationError(
                "non-paired edits must not provide a donor duration"
            )
        expected_duration = target_duration
    if not math.isclose(output_duration, expected_duration, rel_tol=0.0, abs_tol=1e-12):
        raise MediaEditValidationError("output duration differs from the v4 policy")
    duration = _duration_text(output_duration)
    target_video = _mapping(target_diagnostic.get("video"), "target video diagnostics")
    frame_rate = _positive_number(target_video.get("frame_rate"), "target frame rate")
    width = _positive_int(target_video.get("width"), "target video width")
    height = _positive_int(target_video.get("height"), "target video height")
    video_encode = [
        "-c:v",
        video["encoder"],
        "-preset",
        video["preset"],
        "-crf",
        str(video["crf"]),
        "-pix_fmt",
        video["pixel_format"],
    ]
    audio_encode = [
        "-c:a",
        audio["codec"],
        "-b:a",
        audio["bitrate"],
        "-ar",
        str(audio["sample_rate"]),
        "-ac",
        str(audio["channels"]),
    ]
    if condition == "remux_only":
        media_arguments = [
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-c",
            "copy",
        ]
    elif condition in {"same_answer", "opposite_answer"}:
        donor_path = media_root / edit["source_files"][1]["filename"]
        common.extend(["-i", str(donor_path)])
        if orientation == "audio_over_video":
            video_arguments = ["-c:v", "copy"]
            if target_stream_duration < output_duration:
                video_arguments = [
                    *video_encode,
                    "-filter:v",
                    (
                        "tpad=stop_mode=clone:"
                        f"stop_duration={duration},trim=duration={duration},"
                        "setpts=PTS-STARTPTS"
                    ),
                ]
            media_arguments = [
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                *video_arguments,
                *audio_encode,
                "-filter:a",
                f"apad,atrim=duration={duration}",
                "-t",
                duration,
            ]
        else:
            video_filter = (
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
                f"fps={frame_rate:.8f},tpad=stop_mode=clone:stop_duration={duration},"
                f"trim=duration={duration},setpts=PTS-STARTPTS"
            )
            audio_arguments = ["-c:a", "copy"]
            if target_stream_duration < output_duration:
                audio_arguments = [
                    *audio_encode,
                    "-filter:a",
                    f"apad,atrim=duration={duration}",
                ]
            media_arguments = [
                "-map",
                "1:v:0",
                "-map",
                "0:a:0",
                *video_encode,
                "-filter:v",
                video_filter,
                *audio_arguments,
                "-t",
                duration,
            ]
    elif condition == "missing_source":
        if orientation == "audio_over_video":
            layout = "mono" if audio["channels"] == 1 else "stereo"
            common.extend(
                [
                    "-f",
                    "lavfi",
                    "-i",
                    f"anullsrc=r={audio['sample_rate']}:cl={layout}",
                ]
            )
            media_arguments = [
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                *audio_encode,
                "-t",
                duration,
            ]
        else:
            common.extend(
                [
                    "-f",
                    "lavfi",
                    "-i",
                    f"color=c=black:s={width}x{height}:r={frame_rate:.8f}:d={duration}",
                ]
            )
            media_arguments = [
                "-map",
                "1:v:0",
                "-map",
                "0:a:0",
                *video_encode,
                "-c:a",
                "copy",
                "-t",
                duration,
            ]
    elif condition == "temporal_shift":
        shift_seconds = float(
            edit["operation"].get(
                "audio_delay_seconds", edit["operation"].get("video_delay_seconds")
            )
        )
        if orientation == "audio_over_video":
            delay_ms = round(shift_seconds * 1000.0)
            media_arguments = [
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-c:v",
                "copy",
                *audio_encode,
                "-filter:a",
                f"adelay={delay_ms}:all=1,apad,atrim=duration={duration}",
                "-t",
                duration,
            ]
        else:
            media_arguments = [
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                *video_encode,
                "-filter:v",
                (
                    f"tpad=start_mode=clone:start_duration={shift_seconds:.6f},"
                    f"trim=duration={duration},setpts=PTS-STARTPTS"
                ),
                "-c:a",
                "copy",
                "-t",
                duration,
            ]
    else:
        raise MediaEditValidationError("edit condition is unsupported")
    deterministic_muxing = [
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-metadata",
        "creation_time=1970-01-01T00:00:00Z",
        "-fflags",
        "+bitexact",
        "-movflags",
        "+faststart",
        "-threads",
        "1",
        str(output_path),
    ]
    return common + media_arguments + deterministic_muxing


def _validate_output_probe(
    diagnostic: Mapping[str, Any],
    *,
    target_diagnostic: Mapping[str, Any],
    expected_duration: float,
    condition: str,
    source_orientation: str | None,
    target_stream_padded: bool,
    configuration: Mapping[str, Any],
) -> None:
    _positive_number(target_diagnostic.get("duration_seconds"), "target duration")
    tolerance = float(configuration["validation"]["duration_tolerance_seconds"])
    expected_duration = _positive_number(expected_duration, "expected output duration")
    expected_target_padding = False
    if condition in {"same_answer", "opposite_answer"}:
        selected_stream = (
            "video" if source_orientation == "audio_over_video" else "audio"
        )
        selected_diagnostic = _mapping(
            target_diagnostic.get(selected_stream),
            f"target {selected_stream} diagnostics",
        )
        selected_duration = _positive_number(
            selected_diagnostic.get("duration_seconds"),
            f"target {selected_stream} duration",
        )
        expected_target_padding = selected_duration < expected_duration
    if target_stream_padded is not expected_target_padding:
        raise MediaEditValidationError("target-stream padding decision differs")
    duration = _positive_number(diagnostic.get("duration_seconds"), "output duration")
    if abs(duration - expected_duration) > tolerance:
        raise MediaEditValidationError("edited output duration differs from v4 policy")
    target_video = _mapping(target_diagnostic.get("video"), "target video diagnostics")
    output_video = _mapping(diagnostic.get("video"), "output video diagnostics")
    for field in ("width", "height", "frame_rate"):
        if output_video.get(field) != target_video.get(field):
            raise MediaEditValidationError(
                f"edited output video {field.replace('_', ' ')} differs from target"
            )
    video_is_encoded = condition not in {"clean_original", "remux_only"} and (
        source_orientation == "video_over_audio"
        or (
            condition in {"same_answer", "opposite_answer"}
            and source_orientation == "audio_over_video"
            and target_stream_padded
        )
    )
    if video_is_encoded:
        video = configuration["video_encoding"]
        if output_video.get("codec") != video["codec"]:
            raise MediaEditValidationError("edited video codec differs")
        if output_video.get("pixel_format") != video["pixel_format"]:
            raise MediaEditValidationError("edited video pixel format differs")
    else:
        for field in ("codec", "pixel_format"):
            if output_video.get(field) != target_video.get(field):
                raise MediaEditValidationError(
                    f"edited output video {field.replace('_', ' ')} differs from target"
                )
    output_audio = _mapping(diagnostic.get("audio"), "output audio diagnostics")
    audio_is_encoded = condition not in {"clean_original", "remux_only"} and (
        source_orientation == "audio_over_video"
        or (
            condition in {"same_answer", "opposite_answer"}
            and source_orientation == "video_over_audio"
            and target_stream_padded
        )
    )
    if not audio_is_encoded:
        target_audio = _mapping(
            target_diagnostic.get("audio"), "target audio diagnostics"
        )
        for field in ("sample_rate", "channels", "codec", "sample_format"):
            if output_audio.get(field) != target_audio.get(field):
                raise MediaEditValidationError(
                    f"remux audio {field.replace('_', ' ')} differs from target"
                )
    else:
        audio = configuration["audio_encoding"]
        if output_audio.get("codec") != audio["codec"]:
            raise MediaEditValidationError("edited audio codec differs")
        if output_audio.get("sample_rate") != audio["sample_rate"]:
            raise MediaEditValidationError("edited audio sample rate differs")
        if output_audio.get("channels") != audio["channels"]:
            raise MediaEditValidationError("edited audio channel count differs")
    audio_duration = _positive_number(
        output_audio.get("duration_seconds"), "output audio duration"
    )
    video_duration = _positive_number(
        output_video.get("duration_seconds"), "output video duration"
    )
    if (
        abs(audio_duration - expected_duration) > tolerance
        or abs(video_duration - expected_duration) > tolerance
    ):
        raise MediaEditValidationError("edited stream duration differs from v4 policy")


def _run_one_edit(
    *,
    ffmpeg_path: pathlib.Path,
    ffprobe_path: pathlib.Path,
    edit: Mapping[str, Any],
    media_root: pathlib.Path,
    media_output_root: pathlib.Path,
    source_diagnostics: Mapping[str, Mapping[str, Any]],
    source_profiles: Mapping[str, Mapping[str, Any]],
    configuration: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    target_video_id = str(edit["target_video_id"])
    target_diagnostic = source_diagnostics[target_video_id]
    target_duration = float(target_diagnostic["duration_seconds"])
    output_duration = _output_duration_for_edit(edit, source_diagnostics)
    donor_duration: float | None = None
    target_stream_duration = target_duration
    if edit["condition"] in {"same_answer", "opposite_answer"}:
        donor_duration = float(
            source_diagnostics[str(edit["donor_video_id"])]["duration_seconds"]
        )
        selected_stream = (
            "video" if edit["source_orientation"] == "audio_over_video" else "audio"
        )
        target_stream_duration = float(
            _mapping(
                target_diagnostic.get(selected_stream),
                f"target {selected_stream} diagnostics",
            )["duration_seconds"]
        )
    target_stream_padded = _target_stream_needs_padding(
        edit, target_diagnostic, output_duration
    )
    if (
        edit["condition"] == "temporal_shift"
        and float(
            edit["operation"].get(
                "audio_delay_seconds", edit["operation"].get("video_delay_seconds")
            )
        )
        >= target_duration
    ):
        raise MediaEditValidationError("temporal shift must be shorter than the target")
    output_path = media_output_root / str(edit["filename"])
    if edit["condition"] == "clean_original":
        source_path = media_root / str(edit["source_files"][0]["filename"])
        try:
            shutil.copyfile(source_path, output_path)
        except OSError as exc:
            raise MediaEditValidationError(
                f"clean-original copy failed for {edit['output_id']}"
            ) from exc
    else:
        arguments = _ffmpeg_arguments(
            ffmpeg_path=ffmpeg_path,
            edit=edit,
            media_root=media_root,
            output_path=output_path,
            target_duration=target_duration,
            target_stream_duration=target_stream_duration,
            donor_duration=donor_duration,
            output_duration=output_duration,
            target_diagnostic=target_diagnostic,
            configuration=configuration,
        )
        try:
            completed = subprocess.run(
                arguments,
                check=False,
                capture_output=True,
                text=True,
                timeout=int(configuration["validation"]["ffmpeg_timeout_seconds"]),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MediaEditValidationError(
                f"ffmpeg failed for {edit['output_id']}"
            ) from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip().splitlines()
            suffix = f": {detail[-1][:300]}" if detail else ""
            raise MediaEditValidationError(
                f"ffmpeg failed for {edit['output_id']}{suffix}"
            )
    if output_path.is_symlink() or not output_path.is_file():
        raise MediaEditValidationError("ffmpeg did not create a regular output file")
    size_bytes = output_path.stat().st_size
    if size_bytes <= 0:
        raise MediaEditValidationError("ffmpeg created an empty output file")
    output_path.chmod(0o400)
    output_sha256 = _sha256_path(output_path)
    diagnostic = probe_media_file(
        ffprobe_path, output_path, expected_size_bytes=size_bytes
    )
    profile = _probe_media_profile(ffprobe_path, output_path, configuration)
    _validate_output_probe(
        diagnostic,
        target_diagnostic=target_diagnostic,
        expected_duration=output_duration,
        condition=str(edit["condition"]),
        source_orientation=edit["source_orientation"],
        target_stream_padded=target_stream_padded,
        configuration=configuration,
    )
    _validate_output_profile(
        profile,
        target_profile=source_profiles[target_video_id],
        condition=str(edit["condition"]),
        source_orientation=edit["source_orientation"],
        target_stream_padded=target_stream_padded,
        configuration=configuration,
    )
    output = {
        "relative_path": f"media/{edit['filename']}",
        "size_bytes": size_bytes,
        "sha256": output_sha256,
        "diagnostic_sha256": _canonical_digest(diagnostic),
        "profile_sha256": _canonical_digest(profile),
    }
    return {**dict(edit), "output": output}, diagnostic, profile


def _post_edit_index(edits: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    entries = []
    condition_to_role = {
        "same_answer": "same_answer_nuisance",
        "opposite_answer": "opposite_answer_candidate",
    }
    for edit in edits:
        role = condition_to_role.get(edit["condition"])
        if role is None:
            continue
        entries.append(
            {
                "pair_id": edit["pair_id"],
                "target_id": edit["target_id"],
                "component_id": edit["component_id"],
                "partition": edit["partition"],
                "key_sha256": edit["key_sha256"],
                "target_video_id": edit["target_video_id"],
                "output_video_id": edit["output_id"],
                "role": role,
                "source_orientation": edit["source_orientation"],
            }
        )
    entries.sort(
        key=lambda item: (
            item["source_orientation"],
            item["partition"],
            item["component_id"],
            item["pair_id"],
        )
    )
    return entries


def _evaluate_nuisance_by_orientation(
    post_edit_index: Sequence[Mapping[str, Any]],
    source_diagnostics: Mapping[str, Mapping[str, Any]],
    output_diagnostics: Mapping[str, Mapping[str, Any]],
    nuisance_configuration: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for orientation in SOURCE_ORIENTATIONS:
        selected = [
            entry
            for entry in post_edit_index
            if entry["source_orientation"] == orientation
        ]
        if not selected:
            continue
        detector_rows = [
            {key: value for key, value in entry.items() if key != "source_orientation"}
            for entry in selected
        ]
        selected_diagnostics = {
            entry["output_video_id"]: output_diagnostics[entry["output_video_id"]]
            for entry in selected
        }
        reports[orientation] = evaluate_post_edit_diagnostics(
            detector_rows,
            source_diagnostics,
            selected_diagnostics,
            nuisance_configuration,
        )
    if not reports:
        raise MediaEditValidationError("no nuisance report orientation was evaluated")
    return reports


def _output_set_digest(edits: Sequence[Mapping[str, Any]]) -> str:
    records = [
        {
            "output_id": edit["output_id"],
            "relative_path": edit["output"]["relative_path"],
            "size_bytes": edit["output"]["size_bytes"],
            "sha256": edit["output"]["sha256"],
        }
        for edit in edits
    ]
    return _canonical_digest(records)


def _implementation_digest() -> str:
    return _sha256_path(pathlib.Path(__file__).resolve())


def _attest(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["attestation_sha256"] = _canonical_digest(result)
    return result


def _validation_record(
    *,
    edit_index_sha256: str,
    post_edit_index_sha256: str,
    binding_digests: Mapping[str, str],
    nuisance_implementation_source_sha256: Mapping[str, str],
    source_gate_implementation_source_sha256: Mapping[str, str],
    temporal_implementation_source_sha256: Mapping[str, str],
    output_set_sha256: str,
    edit_count: int,
    post_edit_pair_count: int,
    nuisance_reports: Mapping[str, Mapping[str, Any]],
    source_hash_integrity: str,
) -> dict[str, Any]:
    continuation_allowed = all(
        report.get("decision", {}).get("nuisance_detection_status") == "pass"
        for report in nuisance_reports.values()
    )
    return _attest(
        {
            "schema": "conflictbench.perception-controlled-media-validation.v4",
            "status": "complete",
            "continuation_allowed": continuation_allowed,
            "human_evaluation_used": False,
            "nuisance_implementation_source_sha256": dict(
                nuisance_implementation_source_sha256
            ),
            "source_gate_implementation_source_sha256": dict(
                source_gate_implementation_source_sha256
            ),
            "temporal_implementation_source_sha256": dict(
                temporal_implementation_source_sha256
            ),
            "input_digests": {
                "edit_index_sha256": edit_index_sha256,
                "post_edit_index_sha256": post_edit_index_sha256,
                **dict(binding_digests),
                "output_set_sha256": output_set_sha256,
            },
            "checks": {
                "edit_count": edit_count,
                "post_edit_pair_count": post_edit_pair_count,
                "target_anchor_integrity": "pass",
                "component_partition_integrity": "pass",
                "source_hash_integrity": source_hash_integrity,
                "output_hash_integrity": "pass",
                "stream_and_duration_integrity": "pass",
            },
            "nuisance_reports": {
                orientation: dict(report)
                for orientation, report in nuisance_reports.items()
            },
        }
    )


def _run_record(
    *,
    input_digests: Mapping[str, str],
    tool_identities: Mapping[str, Mapping[str, str]],
    counts: Mapping[str, int],
    output_digests: Mapping[str, str],
    continuation_allowed: bool,
    nuisance_implementation_source_sha256: Mapping[str, str],
    source_gate_implementation_source_sha256: Mapping[str, str],
    temporal_implementation_source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    return _attest(
        {
            "schema": "conflictbench.perception-controlled-media-run.v4",
            "status": "complete",
            "continuation_allowed": continuation_allowed,
            "nuisance_implementation_source_sha256": dict(
                nuisance_implementation_source_sha256
            ),
            "source_gate_implementation_source_sha256": dict(
                source_gate_implementation_source_sha256
            ),
            "temporal_implementation_source_sha256": dict(
                temporal_implementation_source_sha256
            ),
            "input_digests": dict(input_digests),
            "tool_identities": {
                name: dict(identity) for name, identity in tool_identities.items()
            },
            "implementation_source_sha256": _implementation_digest(),
            "counts": dict(counts),
            "output_digests": dict(output_digests),
        }
    )


def _write_staged_json(path: pathlib.Path, value: Any) -> bytes:
    payload = _canonical_bytes(value, pretty=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o400)
    return payload


def _fsync_directory(path: pathlib.Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_output_root_separation(
    media_root: pathlib.Path, output_root: pathlib.Path
) -> None:
    """Reject output locations that could modify authenticated source media."""

    try:
        source_root = pathlib.Path(media_root).resolve(strict=True)
        destination = pathlib.Path(output_root).resolve(strict=False)
    except OSError as exc:
        raise MediaEditValidationError(
            "media or output root cannot be resolved"
        ) from exc
    if (
        destination == source_root
        or source_root in destination.parents
        or destination in source_root.parents
    ):
        raise MediaEditValidationError(
            "output root and media root must be mutually disjoint"
        )


def _reauthenticate_source_media(
    *,
    media_root: pathlib.Path,
    media_records: Sequence[Mapping[str, Any]],
    expected_media_set_sha256: str,
) -> str:
    """Rehash every authenticated source after the last source-dependent step."""

    root_input = pathlib.Path(media_root)
    if root_input.is_symlink() or not root_input.is_dir():
        raise MediaEditValidationError(
            "source media root must be a non-symlink directory"
        )
    try:
        root = root_input.resolve(strict=True)
        observed_names = {entry.name for entry in root.iterdir()}
    except OSError as exc:
        raise MediaEditValidationError("source media root cannot be read") from exc

    normalized: list[dict[str, Any]] = []
    expected_names: set[str] = set()
    observed_video_ids: set[str] = set()
    for index, raw_record in enumerate(media_records):
        record = _mapping(raw_record, f"source media record {index}")
        _exact_keys(
            record,
            {"video_id", "filename", "size_bytes", "sha256", "stream_types"},
            f"source media record {index}",
        )
        video_id = _identifier(record.get("video_id"), "source media video ID")
        filename = _text(record.get("filename"), "source media filename")
        if pathlib.PurePath(filename).name != filename or not filename.endswith(".mp4"):
            raise MediaEditValidationError("source media filename is unsafe")
        if pathlib.Path(filename).stem != video_id:
            raise MediaEditValidationError("source media video ID differs")
        if filename in expected_names or video_id in observed_video_ids:
            raise MediaEditValidationError("source media membership differs")
        expected_names.add(filename)
        observed_video_ids.add(video_id)

        source_path = root / filename
        if source_path.is_symlink():
            raise MediaEditValidationError(f"source media file is unsafe: {filename}")
        try:
            file_stat = source_path.stat()
        except OSError as exc:
            raise MediaEditValidationError(
                f"source media file is missing: {filename}"
            ) from exc
        if not stat.S_ISREG(file_stat.st_mode):
            raise MediaEditValidationError(
                f"source media file is not regular: {filename}"
            )
        if stat.S_IMODE(file_stat.st_mode) & 0o222:
            raise MediaEditValidationError(f"source media file is writable: {filename}")
        expected_size = _positive_int(
            record.get("size_bytes"), f"source media size for {filename}"
        )
        if file_stat.st_size != expected_size:
            raise MediaEditValidationError(f"source media size differs: {filename}")
        expected_digest = _sha256_text(
            record.get("sha256"), f"source media digest for {filename}"
        )
        if _sha256_path(source_path) != expected_digest:
            raise MediaEditValidationError(f"source media digest differs: {filename}")
        if record.get("stream_types") != ["audio", "video"]:
            raise MediaEditValidationError(
                f"source media stream record differs: {filename}"
            )
        normalized.append(
            {
                "video_id": video_id,
                "filename": filename,
                "size_bytes": expected_size,
                "sha256": expected_digest,
                "stream_types": ["audio", "video"],
            }
        )

    if observed_names != expected_names:
        raise MediaEditValidationError("source media membership differs")
    normalized.sort(key=lambda item: item["video_id"])
    if _canonical_digest(normalized) != _sha256_text(
        expected_media_set_sha256, "expected source media-set digest"
    ):
        raise MediaEditValidationError("source media-set digest differs")
    return "pass"


def _validate_record_attestation(value: Mapping[str, Any], label: str) -> None:
    attestation = _sha256_text(value.get("attestation_sha256"), f"{label} attestation")
    unsigned = dict(value)
    unsigned.pop("attestation_sha256")
    if _canonical_digest(unsigned) != attestation:
        raise MediaEditValidationError(f"{label} attestation differs")


def _bind_named_sources(
    source_paths: Mapping[str, pathlib.Path],
    expected_source_sha256: Mapping[str, str],
    expected_bundle_sha256: str,
    *,
    label: str,
    required_roles: set[str] | None = None,
    required_filenames: Mapping[str, str] | None = None,
) -> dict[str, str]:
    roles = set(source_paths)
    if not roles or roles != set(expected_source_sha256):
        raise MediaEditValidationError(f"{label} source and digest roles differ")
    if required_roles is not None and roles != required_roles:
        raise MediaEditValidationError(f"{label} source roles differ")
    if required_filenames is not None and roles != set(required_filenames):
        raise MediaEditValidationError(f"{label} source roles differ")
    source_digests: dict[str, str] = {}
    filenames: set[str] = set()
    for role in sorted(roles):
        _identifier(role, f"{label} source role")
        source = pathlib.Path(source_paths[role])
        if source.is_symlink() or not source.is_file():
            raise MediaEditValidationError(f"{label} source is missing or unsafe")
        if required_filenames is not None and source.name != required_filenames[role]:
            raise MediaEditValidationError(f"{label} source filename differs")
        if source.name in filenames:
            raise MediaEditValidationError(f"{label} source filenames must be unique")
        filenames.add(source.name)
        expected = _sha256_text(
            expected_source_sha256[role], f"expected {label} source digest"
        )
        observed = _sha256_path(source)
        if observed != expected:
            raise MediaEditValidationError(f"{label} source digest differs")
        source_digests[role] = observed
    observed_bundle = _canonical_digest(dict(sorted(source_digests.items())))
    if observed_bundle != _sha256_text(
        expected_bundle_sha256, f"expected {label} source-bundle digest"
    ):
        raise MediaEditValidationError(f"{label} source-bundle digest differs")
    return dict(sorted(source_digests.items()))


def _validate_source_gate_evidence(
    *,
    configuration: Mapping[str, Any],
    source_gate_output_path: pathlib.Path,
    expected_source_gate_output_sha256: str,
    source_gate_configuration_path: pathlib.Path,
    expected_source_gate_configuration_sha256: str,
    pilot_index_path: pathlib.Path,
    expected_pilot_index_sha256: str,
    media_root: pathlib.Path,
    expected_media_set_sha256: str,
    media_records: Sequence[Mapping[str, Any]],
    source_gate_implementation_sources: Mapping[str, pathlib.Path],
    expected_source_gate_implementation_source_sha256: Mapping[str, str],
    expected_source_gate_implementation_bundle_sha256: str,
    source_gate_transcript_path: pathlib.Path,
    expected_source_gate_transcript_sha256: str,
) -> dict[str, Any]:
    _load_locked_json(
        source_gate_configuration_path,
        expected_source_gate_configuration_sha256,
        "source-gate configuration",
    )
    _, transcript_value = _load_locked_json(
        source_gate_transcript_path,
        expected_source_gate_transcript_sha256,
        "source-gate transcript",
    )
    _, output_value = _load_locked_json(
        source_gate_output_path,
        expected_source_gate_output_sha256,
        "source-gate output",
    )
    source_digests = _bind_named_sources(
        source_gate_implementation_sources,
        expected_source_gate_implementation_source_sha256,
        expected_source_gate_implementation_bundle_sha256,
        label="source-gate implementation",
        required_roles=set(SOURCE_GATE_IMPLEMENTATION_FILENAMES),
        required_filenames=SOURCE_GATE_IMPLEMENTATION_FILENAMES,
    )
    try:
        replayed = verify_gate_output_replay(
            _mapping(output_value, "source-gate output"),
            config_path=pathlib.Path(source_gate_configuration_path),
            expected_config_sha256=_sha256_text(
                expected_source_gate_configuration_sha256,
                "expected source-gate configuration digest",
            ),
            pilot_index_path=pathlib.Path(pilot_index_path),
            expected_pilot_index_sha256=_sha256_text(
                expected_pilot_index_sha256,
                "expected source-gate pilot-index digest",
            ),
            media_root=pathlib.Path(media_root),
            source_paths=[
                pathlib.Path(source_gate_implementation_sources[role])
                for role in sorted(source_gate_implementation_sources)
            ],
            transcript_path=pathlib.Path(source_gate_transcript_path),
            expected_transcript_sha256=_sha256_text(
                expected_source_gate_transcript_sha256,
                "expected source-gate transcript digest",
            ),
        )
    except (GateValidationError, TypeError, ValueError) as exc:
        raise MediaEditValidationError(
            "source-gate replay verification failed"
        ) from exc
    gate = _mapping(replayed.get("gate"), "replayed source-gate decision")
    authorization = configuration["source_gate_authorization"]
    if (
        gate.get("source_sufficiency_status")
        != authorization["required_source_sufficiency_status"]
        or gate.get("nuisance_detection_status")
        != authorization["required_nuisance_detection_status"]
        or gate.get("overall_pilot_status")
        != authorization["required_overall_pilot_status"]
    ):
        raise MediaEditValidationError(
            "replayed source gate does not authorize controlled-media edits"
        )
    digests = _mapping(replayed.get("input_digests"), "source-gate input digests")
    expected_replayed_digests = {
        "configuration_sha256": _sha256_text(
            expected_source_gate_configuration_sha256,
            "expected source-gate configuration digest",
        ),
        "pilot_index_sha256": _sha256_text(
            expected_pilot_index_sha256, "expected source-gate pilot-index digest"
        ),
        "media_set_sha256": _sha256_text(
            expected_media_set_sha256, "expected media-set digest"
        ),
    }
    for name, expected in expected_replayed_digests.items():
        if digests.get(name) != expected:
            raise MediaEditValidationError(f"source-gate {name} binding differs")
    expected_media = sorted(
        [dict(record) for record in media_records], key=lambda item: item["video_id"]
    )
    if replayed.get("media_files") != expected_media:
        raise MediaEditValidationError("source-gate media receipt binding differs")
    expected_sources = sorted(
        [
            {
                "filename": pathlib.Path(source_gate_implementation_sources[role]).name,
                "size_bytes": pathlib.Path(source_gate_implementation_sources[role])
                .stat()
                .st_size,
                "sha256": source_digests[role],
            }
            for role in sorted(source_gate_implementation_sources)
        ],
        key=lambda item: item["filename"],
    )
    if replayed.get("source_files") != expected_sources:
        raise MediaEditValidationError("source-gate implementation binding differs")
    if digests.get("source_set_sha256") != _canonical_digest(expected_sources):
        raise MediaEditValidationError("source-gate source-set binding differs")
    if replayed.get("inference_transcript_sha256") != _canonical_digest(
        transcript_value
    ):
        raise MediaEditValidationError("source-gate transcript binding differs")
    if (
        _bind_named_sources(
            source_gate_implementation_sources,
            expected_source_gate_implementation_source_sha256,
            expected_source_gate_implementation_bundle_sha256,
            label="source-gate implementation",
            required_roles=set(SOURCE_GATE_IMPLEMENTATION_FILENAMES),
            required_filenames=SOURCE_GATE_IMPLEMENTATION_FILENAMES,
        )
        != source_digests
    ):
        raise MediaEditValidationError(
            "source-gate implementation changed during replay"
        )
    metrics = _mapping(gate.get("metrics"), "replayed source-gate metrics")
    allowed_orientations = [
        orientation
        for orientation in configuration["requested_source_orientations"]
        if authorization["orientation_conditions"][orientation] in metrics
    ]
    if allowed_orientations != configuration["requested_source_orientations"]:
        raise MediaEditValidationError(
            "replayed source gate does not cover every requested orientation"
        )
    return {
        "allowed_orientations": allowed_orientations,
        "implementation_source_sha256": source_digests,
        "output_file_sha256": _sha256_text(
            expected_source_gate_output_sha256, "expected source-gate output digest"
        ),
    }


def _calibration_sources(
    pairs: Sequence[Mapping[str, Any]], media_records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    media_by_id = {str(record["video_id"]): record for record in media_records}
    sources_by_id: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        if pair["partition"] != "scorer_fit":
            continue
        for source_role in ("target", "donor"):
            source = pair[source_role]
            video_id = str(source["video_id"])
            record = {
                "component_id": pair["component_id"],
                "partition": "scorer_fit",
                "video_id": video_id,
                "source_role": source_role,
                "sha256": media_by_id[video_id]["sha256"],
            }
            prior = sources_by_id.setdefault(video_id, record)
            if prior != record:
                raise MediaEditValidationError(
                    "temporal calibration source has inconsistent ownership"
                )
    if not sources_by_id:
        raise MediaEditValidationError("temporal calibration source set is empty")
    return sorted(
        sources_by_id.values(),
        key=lambda item: (
            item["component_id"],
            item["source_role"],
            item["video_id"],
        ),
    )


def _run_temporal_command(
    arguments: Sequence[str], *, label: str, timeout_seconds: int
) -> subprocess.CompletedProcess[bytes]:
    environment = dict(os.environ)
    environment.update({"LANG": "C", "LC_ALL": "C", "PYTHONHASHSEED": "0"})
    try:
        completed = subprocess.run(
            list(arguments),
            check=False,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout_seconds,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaEditValidationError(f"{label} failed") from exc
    if completed.returncode != 0:
        raise MediaEditValidationError(f"{label} failed")
    return completed


def _replay_temporal_events(
    *,
    calibration_sources: Sequence[Mapping[str, Any]],
    candidate_shifts: Sequence[float],
    media_records: Sequence[Mapping[str, Any]],
    media_root: pathlib.Path,
    temporal_implementation_sources: Mapping[str, pathlib.Path],
    temporal_implementation_source_sha256: Mapping[str, str],
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    media_by_id = {str(record["video_id"]): record for record in media_records}
    source_root = pathlib.Path(media_root).resolve(strict=True)
    replayed_events: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="conflictbench-temporal-replay-") as value:
        replay_root = pathlib.Path(value)
        replay_sources: dict[str, pathlib.Path] = {}
        for role in ("generator", "scorer"):
            source_path = pathlib.Path(temporal_implementation_sources[role])
            try:
                source_payload = source_path.read_bytes()
            except OSError as exc:
                raise MediaEditValidationError(
                    "temporal implementation cannot be read for replay"
                ) from exc
            if (
                hashlib.sha256(source_payload).hexdigest()
                != (temporal_implementation_source_sha256[role])
            ):
                raise MediaEditValidationError(
                    "temporal implementation changed before replay"
                )
            replay_source = replay_root / f"{role}.py"
            replay_source.write_bytes(source_payload)
            replay_source.chmod(0o500)
            replay_sources[role] = replay_source
        for shift in candidate_shifts:
            for source in calibration_sources:
                identity = {
                    "component_id": source["component_id"],
                    "partition": "scorer_fit",
                    "video_id": source["video_id"],
                    "source_role": source["source_role"],
                    "shift_seconds": shift,
                }
                event_id = _canonical_digest(identity)[:24]
                media_record = media_by_id[str(source["video_id"])]
                source_path = (source_root / str(media_record["filename"])).resolve(
                    strict=True
                )
                try:
                    source_path.relative_to(source_root)
                except ValueError as exc:
                    raise MediaEditValidationError(
                        "temporal calibration source escapes media root"
                    ) from exc
                source_sha256 = _sha256_path(source_path)
                if source_sha256 != source["sha256"]:
                    raise MediaEditValidationError(
                        "temporal calibration source digest differs"
                    )
                generator_request = {
                    "schema": TEMPORAL_GENERATOR_REQUEST_SCHEMA,
                    "event_id": event_id,
                    **identity,
                    "source_filename": source_path.name,
                    "source_sha256": source_sha256,
                }
                generator_request_payload = _canonical_bytes(generator_request)
                generator_request_sha256 = hashlib.sha256(
                    generator_request_payload
                ).hexdigest()
                generator_request_path = replay_root / f"{event_id}.generator.json"
                shifted_media_path = replay_root / f"{event_id}.mp4"
                generator_request_path.write_bytes(generator_request_payload)
                generator_request_path.chmod(0o400)
                _run_temporal_command(
                    [
                        sys.executable,
                        str(replay_sources["generator"]),
                        "--request",
                        str(generator_request_path),
                        "--source",
                        str(source_path),
                        "--output",
                        str(shifted_media_path),
                    ],
                    label="temporal generator replay",
                    timeout_seconds=timeout_seconds,
                )
                if _sha256_path(generator_request_path) != generator_request_sha256:
                    raise MediaEditValidationError(
                        "temporal generator request changed during replay"
                    )
                if shifted_media_path.is_symlink() or not shifted_media_path.is_file():
                    raise MediaEditValidationError(
                        "temporal generator replay did not produce regular media"
                    )
                if shifted_media_path.stat().st_size <= 0:
                    raise MediaEditValidationError(
                        "temporal generator replay produced empty media"
                    )
                shifted_media_sha256 = _sha256_path(shifted_media_path)
                scorer_request = {
                    "schema": TEMPORAL_SCORER_REQUEST_SCHEMA,
                    "event_id": event_id,
                    **identity,
                    "generator_request_sha256": generator_request_sha256,
                    "reference_media_sha256": source_sha256,
                    "shifted_media_sha256": shifted_media_sha256,
                }
                scorer_request_payload = _canonical_bytes(scorer_request)
                scorer_request_sha256 = hashlib.sha256(
                    scorer_request_payload
                ).hexdigest()
                scorer_request_path = replay_root / f"{event_id}.scorer.json"
                scorer_request_path.write_bytes(scorer_request_payload)
                scorer_request_path.chmod(0o400)
                completed = _run_temporal_command(
                    [
                        sys.executable,
                        str(replay_sources["scorer"]),
                        "--request",
                        str(scorer_request_path),
                        "--reference",
                        str(source_path),
                        "--shifted",
                        str(shifted_media_path),
                    ],
                    label="temporal scorer replay",
                    timeout_seconds=timeout_seconds,
                )
                if len(completed.stdout) > 1024 * 1024:
                    raise MediaEditValidationError(
                        "temporal scorer replay response is too large"
                    )
                if _sha256_path(scorer_request_path) != scorer_request_sha256:
                    raise MediaEditValidationError(
                        "temporal scorer request changed during replay"
                    )
                if _sha256_path(shifted_media_path) != shifted_media_sha256:
                    raise MediaEditValidationError(
                        "temporal shifted media changed during scoring"
                    )
                response = _mapping(
                    _decode_json_bytes(
                        completed.stdout, "temporal scorer replay response"
                    ),
                    "temporal scorer replay response",
                )
                _exact_keys(
                    response,
                    {
                        "schema",
                        "request_sha256",
                        "reference_score",
                        "shifted_score",
                    },
                    "temporal scorer replay response",
                )
                if response.get("schema") != TEMPORAL_SCORER_RESPONSE_SCHEMA:
                    raise MediaEditValidationError(
                        "temporal scorer replay response schema differs"
                    )
                if response.get("request_sha256") != scorer_request_sha256:
                    raise MediaEditValidationError(
                        "temporal scorer replay request binding differs"
                    )
                reference_score = _bounded_number(
                    response.get("reference_score"),
                    "temporal scorer replay reference score",
                )
                shifted_score = _bounded_number(
                    response.get("shifted_score"),
                    "temporal scorer replay shifted score",
                )
                if _sha256_path(source_path) != source_sha256:
                    raise MediaEditValidationError(
                        "temporal calibration source changed during replay"
                    )
                replayed_events.append(
                    {
                        "event_id": event_id,
                        **identity,
                        "source_sha256": source_sha256,
                        "generator_request_sha256": generator_request_sha256,
                        "shifted_media_sha256": shifted_media_sha256,
                        "scorer_request_sha256": scorer_request_sha256,
                        "scorer_response_sha256": _canonical_digest(dict(response)),
                        "reference_score": reference_score,
                        "shifted_score": shifted_score,
                    }
                )
    return replayed_events


def _validate_temporal_calibration(
    value: Any,
    *,
    configuration: Mapping[str, Any],
    expected_pilot_index_sha256: str,
    expected_media_receipt_sha256: str,
    expected_media_set_sha256: str,
    calibration_sources: Sequence[Mapping[str, Any]],
    expected_source_gate_output_sha256: str,
    expected_source_gate_transcript_sha256: str,
    temporal_implementation_source_sha256: Mapping[str, str],
    expected_temporal_implementation_bundle_sha256: str,
    temporal_implementation_sources: Mapping[str, pathlib.Path],
    media_records: Sequence[Mapping[str, Any]],
    media_root: pathlib.Path,
) -> dict[str, Any]:
    record = _mapping(value, "temporal calibration")
    _exact_keys(
        record,
        {
            "schema",
            "status",
            "official_split",
            "calibration_partition",
            "replay_protocol",
            "selection_rule",
            "event_scoring_rule",
            "input_digests",
            "implementation_source_sha256",
            "events",
            "candidate_scores",
            "selected_shift_seconds",
            "attestation_sha256",
        },
        "temporal calibration",
    )
    _validate_record_attestation(record, "temporal calibration")
    locked = configuration["temporal_calibration"]
    if (
        record.get("schema") != locked["record_schema"]
        or record.get("status") != "complete"
        or record.get("official_split") != locked["official_split"]
        or record.get("calibration_partition") != locked["calibration_partition"]
        or record.get("replay_protocol") != locked["replay_protocol"]
        or record.get("selection_rule") != locked["selection_rule"]
        or record.get("event_scoring_rule") != locked["event_scoring_rule"]
    ):
        raise MediaEditValidationError("temporal calibration contract differs")
    digests = _mapping(record.get("input_digests"), "temporal calibration digests")
    expected_digests = {
        "pilot_index_sha256": expected_pilot_index_sha256,
        "media_receipt_sha256": expected_media_receipt_sha256,
        "media_set_sha256": expected_media_set_sha256,
        "calibration_source_set_sha256": _canonical_digest(calibration_sources),
        "source_gate_output_sha256": expected_source_gate_output_sha256,
        "source_gate_transcript_sha256": expected_source_gate_transcript_sha256,
        "temporal_implementation_bundle_sha256": (
            expected_temporal_implementation_bundle_sha256
        ),
    }
    if dict(digests) != expected_digests:
        raise MediaEditValidationError("temporal calibration input binding differs")
    if record.get("implementation_source_sha256") != dict(
        temporal_implementation_source_sha256
    ):
        raise MediaEditValidationError(
            "temporal calibration implementation binding differs"
        )
    candidate_shifts = [
        _positive_number(value, "configured temporal candidate shift")
        for value in locked["candidate_shifts_seconds"]
    ]
    if candidate_shifts != sorted(set(candidate_shifts)):
        raise MediaEditValidationError("configured temporal candidate shifts differ")
    expected_sources = {
        (source["component_id"], source["video_id"], source["source_role"])
        for source in calibration_sources
    }
    events = record.get("events")
    if not isinstance(events, list) or not events:
        raise MediaEditValidationError("temporal calibration events are empty")
    normalized_events: list[dict[str, Any]] = []
    observed: set[tuple[str, str, str, float]] = set()
    scores_by_shift: dict[float, list[float]] = {
        shift: [] for shift in candidate_shifts
    }
    for index, raw_event in enumerate(events):
        event = _mapping(raw_event, f"temporal event {index}")
        _exact_keys(
            event,
            {
                "event_id",
                "component_id",
                "partition",
                "video_id",
                "source_role",
                "shift_seconds",
                "source_sha256",
                "generator_request_sha256",
                "shifted_media_sha256",
                "scorer_request_sha256",
                "scorer_response_sha256",
                "reference_score",
                "shifted_score",
            },
            f"temporal event {index}",
        )
        shift = _positive_number(
            event.get("shift_seconds"), f"temporal event {index} shift"
        )
        if shift not in scores_by_shift:
            raise MediaEditValidationError("temporal event shift is not preregistered")
        component_id = _identifier(
            event.get("component_id"), f"temporal event {index} component ID"
        )
        if event.get("partition") != "scorer_fit":
            raise MediaEditValidationError(
                "temporal calibration events must be scorer_fit only"
            )
        video_id = _identifier(
            event.get("video_id"), f"temporal event {index} video ID"
        )
        source_role = event.get("source_role")
        if source_role not in {"target", "donor"}:
            raise MediaEditValidationError("temporal event source role differs")
        source_key = (component_id, video_id, source_role)
        if source_key not in expected_sources:
            raise MediaEditValidationError(
                "temporal event source is outside the scorer_fit cohort"
            )
        event_key = (*source_key, shift)
        if event_key in observed:
            raise MediaEditValidationError("temporal calibration event is duplicated")
        observed.add(event_key)
        identity = {
            "component_id": component_id,
            "partition": "scorer_fit",
            "video_id": video_id,
            "source_role": source_role,
            "shift_seconds": shift,
        }
        if event.get("event_id") != _canonical_digest(identity)[:24]:
            raise MediaEditValidationError("temporal event ID differs")
        source_sha256 = _sha256_text(
            event.get("source_sha256"), f"temporal event {index} source digest"
        )
        expected_source_sha256 = next(
            str(source["sha256"])
            for source in calibration_sources
            if (
                source["component_id"],
                source["video_id"],
                source["source_role"],
            )
            == source_key
        )
        if source_sha256 != expected_source_sha256:
            raise MediaEditValidationError("temporal event source digest differs")
        generator_request_sha256 = _sha256_text(
            event.get("generator_request_sha256"),
            f"temporal event {index} generator-request digest",
        )
        shifted_media_sha256 = _sha256_text(
            event.get("shifted_media_sha256"),
            f"temporal event {index} shifted-media digest",
        )
        scorer_request_sha256 = _sha256_text(
            event.get("scorer_request_sha256"),
            f"temporal event {index} scorer-request digest",
        )
        scorer_response_sha256 = _sha256_text(
            event.get("scorer_response_sha256"),
            f"temporal event {index} scorer-response digest",
        )
        reference_score = _bounded_number(
            event.get("reference_score"), f"temporal event {index} reference score"
        )
        shifted_score = _bounded_number(
            event.get("shifted_score"), f"temporal event {index} shifted score"
        )
        event_score = abs(reference_score - shifted_score)
        scores_by_shift[shift].append(event_score)
        normalized_events.append(
            {
                "event_id": event["event_id"],
                **identity,
                "source_sha256": source_sha256,
                "generator_request_sha256": generator_request_sha256,
                "shifted_media_sha256": shifted_media_sha256,
                "scorer_request_sha256": scorer_request_sha256,
                "scorer_response_sha256": scorer_response_sha256,
                "reference_score": reference_score,
                "shifted_score": shifted_score,
            }
        )
    expected_event_keys = {
        (*source, shift) for source in expected_sources for shift in candidate_shifts
    }
    if observed != expected_event_keys:
        raise MediaEditValidationError(
            "temporal calibration event coverage differs from scorer_fit sources"
        )
    canonical_events = sorted(
        normalized_events,
        key=lambda item: (
            item["shift_seconds"],
            item["component_id"],
            item["source_role"],
            item["video_id"],
        ),
    )
    if normalized_events != canonical_events:
        raise MediaEditValidationError("temporal calibration events are not canonical")
    replayed_events = _replay_temporal_events(
        calibration_sources=calibration_sources,
        candidate_shifts=candidate_shifts,
        media_records=media_records,
        media_root=media_root,
        temporal_implementation_sources=temporal_implementation_sources,
        temporal_implementation_source_sha256=(temporal_implementation_source_sha256),
        timeout_seconds=configuration["validation"]["ffmpeg_timeout_seconds"],
    )
    if normalized_events != replayed_events:
        raise MediaEditValidationError(
            "temporal calibration differs from pinned replay"
        )
    recomputed_candidates = [
        {
            "shift_seconds": shift,
            "event_count": len(scores_by_shift[shift]),
            "mean_absolute_score_change": math.fsum(scores_by_shift[shift])
            / len(scores_by_shift[shift]),
        }
        for shift in candidate_shifts
    ]
    candidates = record.get("candidate_scores")
    if not isinstance(candidates, list) or len(candidates) != len(
        recomputed_candidates
    ):
        raise MediaEditValidationError("temporal candidate scores are incomplete")
    normalized_candidates: list[dict[str, Any]] = []
    for index, (raw_candidate, recomputed) in enumerate(
        zip(candidates, recomputed_candidates)
    ):
        candidate = _mapping(raw_candidate, f"temporal candidate score {index}")
        _exact_keys(
            candidate,
            {"shift_seconds", "event_count", "mean_absolute_score_change"},
            f"temporal candidate score {index}",
        )
        shift = _positive_number(
            candidate.get("shift_seconds"), f"temporal candidate score {index} shift"
        )
        event_count = _positive_int(
            candidate.get("event_count"), f"temporal candidate score {index} count"
        )
        score = _bounded_number(
            candidate.get("mean_absolute_score_change"),
            f"temporal candidate score {index} value",
        )
        if (
            shift != recomputed["shift_seconds"]
            or event_count != recomputed["event_count"]
            or not math.isclose(
                score,
                recomputed["mean_absolute_score_change"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise MediaEditValidationError(
                "temporal candidate score differs from event-level recomputation"
            )
        normalized_candidates.append(
            {
                "shift_seconds": shift,
                "event_count": event_count,
                "mean_absolute_score_change": score,
            }
        )
    winner = max(
        recomputed_candidates,
        key=lambda item: (
            item["mean_absolute_score_change"],
            -item["shift_seconds"],
        ),
    )
    selected = _positive_number(
        record.get("selected_shift_seconds"), "selected temporal shift"
    )
    if selected != winner["shift_seconds"]:
        raise MediaEditValidationError(
            "selected temporal shift differs from the deterministic rule"
        )
    return {
        **dict(record),
        "input_digests": dict(digests),
        "implementation_source_sha256": dict(temporal_implementation_source_sha256),
        "events": normalized_events,
        "candidate_scores": normalized_candidates,
        "selected_shift_seconds": selected,
    }


def _bind_nuisance_implementation(
    source_paths: Mapping[str, pathlib.Path], expected_bundle_sha256: str
) -> dict[str, str]:
    if set(source_paths) != set(IMPLEMENTATION_SOURCE_ROLES):
        raise MediaEditValidationError("nuisance implementation source set differs")
    source_digests: dict[str, str] = {}
    for role in IMPLEMENTATION_SOURCE_ROLES:
        source = pathlib.Path(source_paths[role])
        if source.is_symlink() or not source.is_file():
            raise MediaEditValidationError(
                "nuisance implementation source is missing or unsafe"
            )
        source_digests[role] = _sha256_path(source)
    observed_bundle = _canonical_digest(dict(sorted(source_digests.items())))
    if observed_bundle != _sha256_text(
        expected_bundle_sha256, "nuisance implementation-bundle digest"
    ):
        raise MediaEditValidationError("nuisance implementation-bundle digest differs")
    return dict(sorted(source_digests.items()))


def _load_context(
    *,
    configuration_path: pathlib.Path,
    expected_configuration_sha256: str,
    pilot_index_path: pathlib.Path,
    expected_pilot_index_sha256: str,
    media_receipt_path: pathlib.Path,
    expected_media_receipt_sha256: str,
    expected_media_set_sha256: str,
    media_root: pathlib.Path,
    nuisance_configuration_path: pathlib.Path,
    expected_nuisance_configuration_sha256: str,
    nuisance_implementation_sources: Mapping[str, pathlib.Path],
    expected_nuisance_implementation_bundle_sha256: str,
    nuisance_contract_path: pathlib.Path,
    expected_nuisance_contract_sha256: str,
    source_gate_configuration_path: pathlib.Path,
    expected_source_gate_configuration_sha256: str,
    source_gate_output_path: pathlib.Path,
    expected_source_gate_output_sha256: str,
    source_gate_implementation_sources: Mapping[str, pathlib.Path],
    expected_source_gate_implementation_source_sha256: Mapping[str, str],
    expected_source_gate_implementation_bundle_sha256: str,
    source_gate_transcript_path: pathlib.Path,
    expected_source_gate_transcript_sha256: str,
    temporal_implementation_sources: Mapping[str, pathlib.Path],
    expected_temporal_implementation_source_sha256: Mapping[str, str],
    expected_temporal_implementation_bundle_sha256: str,
    temporal_calibration_path: pathlib.Path,
    expected_temporal_calibration_sha256: str,
) -> dict[str, Any]:
    _, raw_config = _load_locked_json(
        configuration_path, expected_configuration_sha256, "configuration"
    )
    config = _validate_configuration(raw_config)
    _, pilot = _load_locked_json(
        pilot_index_path, expected_pilot_index_sha256, "pilot index"
    )
    pairs, expected_media_ids = _validate_pilot(pilot)
    _, receipt = _load_locked_json(
        media_receipt_path, expected_media_receipt_sha256, "media receipt"
    )
    _validate_media_receipt_scalar_types(receipt)
    try:
        media_records = validate_media_receipt(
            receipt,
            media_root=pathlib.Path(media_root),
            expected_pilot_index_sha256=expected_pilot_index_sha256,
            expected_media_set_sha256=expected_media_set_sha256,
            expected_video_ids=expected_media_ids,
        )
    except NuisanceValidationError as exc:
        raise MediaEditValidationError(str(exc)) from exc
    _, nuisance_config = _load_locked_json(
        nuisance_configuration_path,
        expected_nuisance_configuration_sha256,
        "nuisance configuration",
    )
    nuisance_source_digests = _bind_nuisance_implementation(
        nuisance_implementation_sources,
        expected_nuisance_implementation_bundle_sha256,
    )
    _, nuisance_contract = _load_locked_json(
        nuisance_contract_path,
        expected_nuisance_contract_sha256,
        "nuisance contract",
    )
    try:
        validate_nuisance_contract(
            nuisance_contract,
            expected_pilot_index_sha256=expected_pilot_index_sha256,
            expected_media_receipt_sha256=expected_media_receipt_sha256,
            expected_media_set_sha256=expected_media_set_sha256,
            expected_configuration_sha256=expected_nuisance_configuration_sha256,
            expected_implementation_bundle_sha256=(
                expected_nuisance_implementation_bundle_sha256
            ),
        )
    except NuisanceValidationError as exc:
        raise MediaEditValidationError(str(exc)) from exc
    source_gate = _validate_source_gate_evidence(
        configuration=config,
        source_gate_output_path=source_gate_output_path,
        expected_source_gate_output_sha256=expected_source_gate_output_sha256,
        source_gate_configuration_path=source_gate_configuration_path,
        expected_source_gate_configuration_sha256=(
            expected_source_gate_configuration_sha256
        ),
        pilot_index_path=pilot_index_path,
        expected_pilot_index_sha256=expected_pilot_index_sha256,
        media_root=media_root,
        expected_media_set_sha256=expected_media_set_sha256,
        media_records=media_records,
        source_gate_implementation_sources=source_gate_implementation_sources,
        expected_source_gate_implementation_source_sha256=(
            expected_source_gate_implementation_source_sha256
        ),
        expected_source_gate_implementation_bundle_sha256=(
            expected_source_gate_implementation_bundle_sha256
        ),
        source_gate_transcript_path=source_gate_transcript_path,
        expected_source_gate_transcript_sha256=(expected_source_gate_transcript_sha256),
    )
    temporal_source_digests = _bind_named_sources(
        temporal_implementation_sources,
        expected_temporal_implementation_source_sha256,
        expected_temporal_implementation_bundle_sha256,
        label="temporal implementation",
        required_roles=set(config["temporal_calibration"]["implementation_roles"]),
    )
    calibration_sources = _calibration_sources(pairs, media_records)
    _, temporal_calibration_value = _load_locked_json(
        temporal_calibration_path,
        expected_temporal_calibration_sha256,
        "temporal calibration",
    )
    temporal_calibration = _validate_temporal_calibration(
        temporal_calibration_value,
        configuration=config,
        expected_pilot_index_sha256=expected_pilot_index_sha256,
        expected_media_receipt_sha256=expected_media_receipt_sha256,
        expected_media_set_sha256=expected_media_set_sha256,
        calibration_sources=calibration_sources,
        expected_source_gate_output_sha256=expected_source_gate_output_sha256,
        expected_source_gate_transcript_sha256=(expected_source_gate_transcript_sha256),
        temporal_implementation_source_sha256=temporal_source_digests,
        expected_temporal_implementation_bundle_sha256=(
            expected_temporal_implementation_bundle_sha256
        ),
        temporal_implementation_sources=temporal_implementation_sources,
        media_records=media_records,
        media_root=media_root,
    )
    if (
        _bind_named_sources(
            temporal_implementation_sources,
            expected_temporal_implementation_source_sha256,
            expected_temporal_implementation_bundle_sha256,
            label="temporal implementation",
            required_roles=set(config["temporal_calibration"]["implementation_roles"]),
        )
        != temporal_source_digests
    ):
        raise MediaEditValidationError("temporal implementation changed during replay")
    plan = build_edit_plan(
        pilot,
        media_records,
        config,
        selected_shift_seconds=temporal_calibration["selected_shift_seconds"],
        authorized_orientations=source_gate["allowed_orientations"],
    )
    return {
        "configuration": config,
        "pilot": pilot,
        "media_records": media_records,
        "nuisance_configuration": nuisance_config,
        "nuisance_implementation_source_sha256": nuisance_source_digests,
        "nuisance_contract": nuisance_contract,
        "source_gate": source_gate,
        "temporal_implementation_source_sha256": temporal_source_digests,
        "temporal_calibration": temporal_calibration,
        "plan": plan,
        "input_digests": {
            "configuration_sha256": _sha256_text(
                expected_configuration_sha256, "configuration digest"
            ),
            "pilot_index_sha256": _sha256_text(
                expected_pilot_index_sha256, "pilot-index digest"
            ),
            "media_receipt_sha256": _sha256_text(
                expected_media_receipt_sha256, "media-receipt digest"
            ),
            "media_set_sha256": _sha256_text(
                expected_media_set_sha256, "media-set digest"
            ),
            "nuisance_configuration_sha256": _sha256_text(
                expected_nuisance_configuration_sha256,
                "nuisance-configuration digest",
            ),
            "nuisance_implementation_bundle_sha256": _sha256_text(
                expected_nuisance_implementation_bundle_sha256,
                "nuisance implementation-bundle digest",
            ),
            "nuisance_contract_sha256": _sha256_text(
                expected_nuisance_contract_sha256,
                "nuisance-contract digest",
            ),
            "source_gate_configuration_sha256": _sha256_text(
                expected_source_gate_configuration_sha256,
                "source-gate configuration digest",
            ),
            "source_gate_output_sha256": _sha256_text(
                expected_source_gate_output_sha256,
                "source-gate output digest",
            ),
            "source_gate_implementation_bundle_sha256": _sha256_text(
                expected_source_gate_implementation_bundle_sha256,
                "source-gate implementation-bundle digest",
            ),
            "source_gate_transcript_sha256": _sha256_text(
                expected_source_gate_transcript_sha256,
                "source-gate transcript digest",
            ),
            "temporal_implementation_bundle_sha256": _sha256_text(
                expected_temporal_implementation_bundle_sha256,
                "temporal implementation-bundle digest",
            ),
            "temporal_calibration_sha256": _sha256_text(
                expected_temporal_calibration_sha256,
                "temporal-calibration digest",
            ),
        },
    }


def construct_controlled_media(
    *,
    configuration_path: pathlib.Path,
    expected_configuration_sha256: str,
    pilot_index_path: pathlib.Path,
    expected_pilot_index_sha256: str,
    media_receipt_path: pathlib.Path,
    expected_media_receipt_sha256: str,
    expected_media_set_sha256: str,
    media_root: pathlib.Path,
    nuisance_configuration_path: pathlib.Path,
    expected_nuisance_configuration_sha256: str,
    nuisance_implementation_sources: Mapping[str, pathlib.Path],
    expected_nuisance_implementation_bundle_sha256: str,
    nuisance_contract_path: pathlib.Path,
    expected_nuisance_contract_sha256: str,
    source_gate_configuration_path: pathlib.Path,
    expected_source_gate_configuration_sha256: str,
    source_gate_output_path: pathlib.Path,
    expected_source_gate_output_sha256: str,
    source_gate_implementation_sources: Mapping[str, pathlib.Path],
    expected_source_gate_implementation_source_sha256: Mapping[str, str],
    expected_source_gate_implementation_bundle_sha256: str,
    source_gate_transcript_path: pathlib.Path,
    expected_source_gate_transcript_sha256: str,
    temporal_implementation_sources: Mapping[str, pathlib.Path],
    expected_temporal_implementation_source_sha256: Mapping[str, str],
    expected_temporal_implementation_bundle_sha256: str,
    temporal_calibration_path: pathlib.Path,
    expected_temporal_calibration_sha256: str,
    ffmpeg_path: pathlib.Path,
    ffprobe_path: pathlib.Path,
    output_root: pathlib.Path,
) -> dict[str, Any]:
    """Create every edit, validate final bytes, then publish one complete directory."""

    _validate_output_root_separation(media_root, output_root)
    context = _load_context(
        configuration_path=configuration_path,
        expected_configuration_sha256=expected_configuration_sha256,
        pilot_index_path=pilot_index_path,
        expected_pilot_index_sha256=expected_pilot_index_sha256,
        media_receipt_path=media_receipt_path,
        expected_media_receipt_sha256=expected_media_receipt_sha256,
        expected_media_set_sha256=expected_media_set_sha256,
        media_root=media_root,
        nuisance_configuration_path=nuisance_configuration_path,
        expected_nuisance_configuration_sha256=(expected_nuisance_configuration_sha256),
        nuisance_implementation_sources=nuisance_implementation_sources,
        expected_nuisance_implementation_bundle_sha256=(
            expected_nuisance_implementation_bundle_sha256
        ),
        nuisance_contract_path=nuisance_contract_path,
        expected_nuisance_contract_sha256=expected_nuisance_contract_sha256,
        source_gate_configuration_path=source_gate_configuration_path,
        expected_source_gate_configuration_sha256=(
            expected_source_gate_configuration_sha256
        ),
        source_gate_output_path=source_gate_output_path,
        expected_source_gate_output_sha256=expected_source_gate_output_sha256,
        source_gate_implementation_sources=source_gate_implementation_sources,
        expected_source_gate_implementation_source_sha256=(
            expected_source_gate_implementation_source_sha256
        ),
        expected_source_gate_implementation_bundle_sha256=(
            expected_source_gate_implementation_bundle_sha256
        ),
        source_gate_transcript_path=source_gate_transcript_path,
        expected_source_gate_transcript_sha256=(expected_source_gate_transcript_sha256),
        temporal_implementation_sources=temporal_implementation_sources,
        expected_temporal_implementation_source_sha256=(
            expected_temporal_implementation_source_sha256
        ),
        expected_temporal_implementation_bundle_sha256=(
            expected_temporal_implementation_bundle_sha256
        ),
        temporal_calibration_path=temporal_calibration_path,
        expected_temporal_calibration_sha256=expected_temporal_calibration_sha256,
    )
    ffmpeg, ffmpeg_identity = _tool_identity(ffmpeg_path, "ffmpeg")
    ffprobe, ffprobe_identity = _tool_identity(ffprobe_path, "ffprobe")
    tool_identities = {"ffmpeg": ffmpeg_identity, "ffprobe": ffprobe_identity}
    source_root = pathlib.Path(media_root).resolve(strict=True)
    try:
        source_diagnostics = _probe_sources(
            ffprobe_path=ffprobe,
            media_root=source_root,
            media_records=context["media_records"],
        )
        source_profiles = _probe_source_profiles(
            ffprobe_path=ffprobe,
            media_root=source_root,
            media_records=context["media_records"],
            configuration=context["configuration"],
        )
    except NuisanceValidationError as exc:
        raise MediaEditValidationError(str(exc)) from exc

    output = pathlib.Path(output_root)
    parent = output.parent.resolve(strict=True)
    if parent.is_symlink() or not parent.is_dir():
        raise MediaEditValidationError("output parent must be a non-symlink directory")
    if output.name in {"", ".", ".."} or os.path.lexists(output):
        raise MediaEditValidationError("output root already exists or is unsafe")
    staging = pathlib.Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    try:
        media_output = staging / "media"
        media_output.mkdir()
        edited: list[dict[str, Any]] = []
        output_diagnostics: dict[str, dict[str, Any]] = {}
        for edit in context["plan"]:
            completed_edit, diagnostic, _profile = _run_one_edit(
                ffmpeg_path=ffmpeg,
                ffprobe_path=ffprobe,
                edit=edit,
                media_root=source_root,
                media_output_root=media_output,
                source_diagnostics=source_diagnostics,
                source_profiles=source_profiles,
                configuration=context["configuration"],
            )
            edited.append(completed_edit)
            output_diagnostics[edit["output_id"]] = diagnostic
        post_edit_index = _post_edit_index(edited)
        try:
            nuisance_reports = _evaluate_nuisance_by_orientation(
                post_edit_index,
                source_diagnostics,
                output_diagnostics,
                context["nuisance_configuration"],
            )
        except NuisanceValidationError as exc:
            raise MediaEditValidationError(str(exc)) from exc

        source_hash_integrity = _reauthenticate_source_media(
            media_root=source_root,
            media_records=context["media_records"],
            expected_media_set_sha256=context["input_digests"]["media_set_sha256"],
        )
        output_set_sha256 = _output_set_digest(edited)
        target_count = len({edit["target_id"] for edit in edited})
        clean_original_count = sum(
            edit["condition"] == "clean_original" for edit in edited
        )
        edit_index = _attest(
            {
                "schema": "conflictbench.perception-controlled-media-index.v4",
                "status": "complete",
                "continuation_allowed": all(
                    report["decision"]["nuisance_detection_status"] == "pass"
                    for report in nuisance_reports.values()
                ),
                "nuisance_implementation_source_sha256": context[
                    "nuisance_implementation_source_sha256"
                ],
                "source_gate_implementation_source_sha256": context["source_gate"][
                    "implementation_source_sha256"
                ],
                "temporal_implementation_source_sha256": context[
                    "temporal_implementation_source_sha256"
                ],
                "input_digests": context["input_digests"],
                "tool_identities": tool_identities,
                "configuration": context["configuration"],
                "counts": {
                    "target_count": target_count,
                    "edit_count": len(edited),
                    "clean_original_count": clean_original_count,
                },
                "output_set_sha256": output_set_sha256,
                "edits": edited,
            }
        )
        edit_index_payload = _write_staged_json(staging / "edit_index.json", edit_index)
        post_edit_payload = _write_staged_json(
            staging / "post_edit_index.json", post_edit_index
        )
        validation = _validation_record(
            edit_index_sha256=hashlib.sha256(edit_index_payload).hexdigest(),
            post_edit_index_sha256=hashlib.sha256(post_edit_payload).hexdigest(),
            binding_digests=context["input_digests"],
            nuisance_implementation_source_sha256=context[
                "nuisance_implementation_source_sha256"
            ],
            source_gate_implementation_source_sha256=context["source_gate"][
                "implementation_source_sha256"
            ],
            temporal_implementation_source_sha256=context[
                "temporal_implementation_source_sha256"
            ],
            output_set_sha256=output_set_sha256,
            edit_count=len(edited),
            post_edit_pair_count=len(post_edit_index),
            nuisance_reports=nuisance_reports,
            source_hash_integrity=source_hash_integrity,
        )
        validation_payload = _write_staged_json(staging / "validation.json", validation)
        counts = {
            "edit_count": len(edited),
            "post_edit_pair_count": len(post_edit_index),
            "source_video_count": len(context["media_records"]),
            "target_count": target_count,
            "clean_original_count": clean_original_count,
        }
        run_record = _run_record(
            input_digests=context["input_digests"],
            tool_identities=tool_identities,
            counts=counts,
            output_digests={
                "edit_index_sha256": hashlib.sha256(edit_index_payload).hexdigest(),
                "post_edit_index_sha256": hashlib.sha256(post_edit_payload).hexdigest(),
                "validation_sha256": hashlib.sha256(validation_payload).hexdigest(),
                "output_set_sha256": output_set_sha256,
            },
            continuation_allowed=bool(validation["continuation_allowed"]),
            nuisance_implementation_source_sha256=context[
                "nuisance_implementation_source_sha256"
            ],
            source_gate_implementation_source_sha256=context["source_gate"][
                "implementation_source_sha256"
            ],
            temporal_implementation_source_sha256=context[
                "temporal_implementation_source_sha256"
            ],
        )
        _write_staged_json(staging / "run_record.json", run_record)
        _fsync_directory(media_output)
        _fsync_directory(staging)
        if os.path.lexists(output):
            raise MediaEditValidationError("output root appeared during construction")
        os.rename(staging, output)
        _fsync_directory(parent)
        return run_record
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _validate_attestation(value: Any, schema: str, label: str) -> Mapping[str, Any]:
    record = _mapping(value, label)
    if record.get("schema") != schema:
        raise MediaEditValidationError(f"{label} schema differs")
    attestation = _sha256_text(record.get("attestation_sha256"), f"{label} attestation")
    unsigned = dict(record)
    unsigned.pop("attestation_sha256")
    if _canonical_digest(unsigned) != attestation:
        raise MediaEditValidationError(f"{label} attestation differs")
    return record


def _validate_digest_mapping(
    value: Any,
    label: str,
    *,
    expected_keys: set[str] | None = None,
) -> Mapping[str, Any]:
    digests = _mapping(value, label)
    if expected_keys is not None:
        _exact_keys(digests, expected_keys, label)
    elif not digests:
        raise MediaEditValidationError(f"{label} must be nonempty")
    for key, digest in digests.items():
        _text(key, f"{label} key")
        _sha256_text(digest, f"{label}.{key}")
    return digests


def _validate_anchor_schema(value: Any, label: str) -> None:
    anchor = _mapping(value, label)
    _exact_keys(anchor, {"question", "options", "answer_id", "answer"}, label)
    _text(anchor.get("question"), f"{label} question")
    options = anchor.get("options")
    if not isinstance(options, list) or len(options) != 3:
        raise MediaEditValidationError(f"{label} options must contain three strings")
    for index, option in enumerate(options):
        _text(option, f"{label} option {index}")
    answer_id = _nonnegative_int(anchor.get("answer_id"), f"{label} answer ID")
    if answer_id >= len(options):
        raise MediaEditValidationError(f"{label} answer ID is invalid")
    _text(anchor.get("answer"), f"{label} answer")


def _validate_tool_identities(value: Any, label: str) -> None:
    tools = _mapping(value, label)
    _exact_keys(tools, {"ffmpeg", "ffprobe"}, label)
    for name in ("ffmpeg", "ffprobe"):
        identity = _mapping(tools[name], f"{label}.{name}")
        _exact_keys(identity, _TOOL_IDENTITY_KEYS, f"{label}.{name}")
        _text(identity.get("filename"), f"{label}.{name}.filename")
        _sha256_text(identity.get("sha256"), f"{label}.{name}.sha256")
        _text(identity.get("version"), f"{label}.{name}.version")


def _validate_edit_index_schema(value: Any) -> Mapping[str, Any]:
    record = _validate_attestation(
        value,
        "conflictbench.perception-controlled-media-index.v4",
        "edit index",
    )
    _exact_keys(record, _EDIT_INDEX_KEYS, "edit index")
    _text(record.get("status"), "edit index status")
    _boolean(record.get("continuation_allowed"), "edit index continuation decision")
    _validate_digest_mapping(
        record.get("nuisance_implementation_source_sha256"),
        "edit index nuisance implementation sources",
    )
    _validate_digest_mapping(
        record.get("source_gate_implementation_source_sha256"),
        "edit index source-gate implementation sources",
    )
    _validate_digest_mapping(
        record.get("temporal_implementation_source_sha256"),
        "edit index temporal implementation sources",
    )
    _validate_digest_mapping(
        record.get("input_digests"),
        "edit index input digests",
        expected_keys=_INPUT_DIGEST_KEYS,
    )
    _validate_tool_identities(record.get("tool_identities"), "edit index tools")
    _validate_configuration(record.get("configuration"))
    counts = _mapping(record.get("counts"), "edit index counts")
    _exact_keys(counts, _EDIT_COUNT_KEYS, "edit index counts")
    for key in sorted(_EDIT_COUNT_KEYS):
        _positive_int(counts.get(key), f"edit index {key}")
    _sha256_text(record.get("output_set_sha256"), "edit index output-set digest")
    edits = record.get("edits")
    if not isinstance(edits, list) or not edits:
        raise MediaEditValidationError("edit index entries must be a nonempty list")
    for index, raw_edit in enumerate(edits):
        edit = _mapping(raw_edit, f"edit index entry {index}")
        _exact_keys(edit, _EDIT_KEYS, f"edit index entry {index}")
        for key in (
            "output_id",
            "filename",
            "target_id",
            "component_id",
            "target_video_id",
        ):
            _identifier(edit.get(key), f"edit index entry {index} {key}")
        if edit.get("condition") not in CONDITIONS:
            raise MediaEditValidationError(
                f"edit index entry {index} condition differs"
            )
        orientation = edit.get("source_orientation")
        if orientation is not None and orientation not in SOURCE_ORIENTATIONS:
            raise MediaEditValidationError(
                f"edit index entry {index} source orientation differs"
            )
        for key in ("pair_id", "donor_video_id"):
            if edit.get(key) is not None:
                _identifier(edit.get(key), f"edit index entry {index} {key}")
        if edit.get("partition") not in PARTITIONS:
            raise MediaEditValidationError(
                f"edit index entry {index} partition differs"
            )
        _sha256_text(
            edit.get("key_sha256"), f"edit index entry {index} question-key digest"
        )
        _validate_anchor_schema(edit.get("anchor"), f"edit index entry {index} anchor")
        _validate_source_entry(edit.get("target"), f"edit index entry {index} target")
        donor = edit.get("donor")
        if donor is not None:
            _validate_source_entry(donor, f"edit index entry {index} donor")
        operation = _mapping(
            edit.get("operation"), f"edit index entry {index} operation"
        )
        expected_operation_keys = {"audio_source", "video_source"}
        if edit.get("condition") == "temporal_shift":
            delay_key = (
                "audio_delay_seconds"
                if edit.get("source_orientation") == "audio_over_video"
                else "video_delay_seconds"
            )
            expected_operation_keys.add(delay_key)
        _exact_keys(
            operation,
            expected_operation_keys,
            f"edit index entry {index} operation",
        )
        for key in ("audio_source", "video_source"):
            _text(operation.get(key), f"edit index entry {index} {key}")
        for key in expected_operation_keys - {"audio_source", "video_source"}:
            _positive_number(operation.get(key), f"edit index entry {index} {key}")
        source_files = edit.get("source_files")
        if not isinstance(source_files, list) or not source_files:
            raise MediaEditValidationError(
                f"edit index entry {index} source files must be nonempty"
            )
        for source_index, raw_source in enumerate(source_files):
            source_label = f"edit index entry {index} source file {source_index}"
            source = _mapping(raw_source, source_label)
            _exact_keys(
                source,
                _EDIT_SOURCE_FILE_KEYS,
                source_label,
            )
            _text(source.get("source_role"), f"{source_label} role")
            _identifier(source.get("video_id"), f"{source_label} video ID")
            _text(source.get("filename"), f"{source_label} filename")
            _positive_int(source.get("size_bytes"), f"{source_label} size")
            _sha256_text(source.get("sha256"), f"{source_label} digest")
        output_label = f"edit index entry {index} output"
        output = _mapping(edit.get("output"), output_label)
        _exact_keys(output, _EDIT_OUTPUT_KEYS, output_label)
        _text(output.get("relative_path"), f"{output_label} relative path")
        _positive_int(output.get("size_bytes"), f"{output_label} size")
        for key in ("sha256", "diagnostic_sha256", "profile_sha256"):
            _sha256_text(output.get(key), f"{output_label} {key}")
    return record


def _validate_post_edit_index_schema(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise MediaEditValidationError("post-edit index must be a list")
    records: list[Mapping[str, Any]] = []
    for index, raw_record in enumerate(value):
        record = _mapping(raw_record, f"post-edit record {index}")
        _exact_keys(record, _POST_EDIT_KEYS, f"post-edit record {index}")
        for key in (
            "pair_id",
            "target_id",
            "component_id",
            "target_video_id",
            "output_video_id",
        ):
            _identifier(record.get(key), f"post-edit record {index} {key}")
        if record.get("partition") not in PARTITIONS:
            raise MediaEditValidationError(
                f"post-edit record {index} partition differs"
            )
        _sha256_text(
            record.get("key_sha256"), f"post-edit record {index} question-key digest"
        )
        if record.get("role") not in PAIR_ROLES:
            raise MediaEditValidationError(f"post-edit record {index} role differs")
        if record.get("source_orientation") not in SOURCE_ORIENTATIONS:
            raise MediaEditValidationError(
                f"post-edit record {index} source orientation differs"
            )
        records.append(record)
    return records


def _validate_validation_schema(value: Any) -> Mapping[str, Any]:
    record = _validate_attestation(
        value,
        "conflictbench.perception-controlled-media-validation.v4",
        "validation record",
    )
    _exact_keys(record, _VALIDATION_KEYS, "validation record")
    _text(record.get("status"), "validation status")
    _boolean(record.get("continuation_allowed"), "validation continuation decision")
    _boolean(record.get("human_evaluation_used"), "validation human-evaluation flag")
    _validate_digest_mapping(
        record.get("nuisance_implementation_source_sha256"),
        "validation nuisance implementation sources",
    )
    _validate_digest_mapping(
        record.get("source_gate_implementation_source_sha256"),
        "validation source-gate implementation sources",
    )
    _validate_digest_mapping(
        record.get("temporal_implementation_source_sha256"),
        "validation temporal implementation sources",
    )
    _validate_digest_mapping(
        record.get("input_digests"),
        "validation input digests",
        expected_keys=_VALIDATION_INPUT_DIGEST_KEYS,
    )
    checks = _mapping(record.get("checks"), "validation checks")
    _exact_keys(checks, _VALIDATION_CHECK_KEYS, "validation checks")
    _positive_int(checks.get("edit_count"), "validation edit count")
    _positive_int(checks.get("post_edit_pair_count"), "validation post-edit pair count")
    for key in _VALIDATION_CHECK_KEYS - {"edit_count", "post_edit_pair_count"}:
        if checks.get(key) != "pass":
            raise MediaEditValidationError(f"validation check {key} differs")
    reports = _mapping(record.get("nuisance_reports"), "validation nuisance reports")
    if not reports:
        raise MediaEditValidationError("validation nuisance reports must be nonempty")
    for orientation, report in reports.items():
        if orientation not in SOURCE_ORIENTATIONS:
            raise MediaEditValidationError("validation nuisance orientation differs")
        _mapping(report, f"validation nuisance report {orientation}")
    return record


def _validate_run_schema(value: Any) -> Mapping[str, Any]:
    record = _validate_attestation(
        value,
        "conflictbench.perception-controlled-media-run.v4",
        "run record",
    )
    _exact_keys(record, _RUN_KEYS, "run record")
    _text(record.get("status"), "run status")
    _boolean(record.get("continuation_allowed"), "run continuation decision")
    _validate_digest_mapping(
        record.get("nuisance_implementation_source_sha256"),
        "run nuisance implementation sources",
    )
    _validate_digest_mapping(
        record.get("source_gate_implementation_source_sha256"),
        "run source-gate implementation sources",
    )
    _validate_digest_mapping(
        record.get("temporal_implementation_source_sha256"),
        "run temporal implementation sources",
    )
    _validate_digest_mapping(
        record.get("input_digests"),
        "run input digests",
        expected_keys=_INPUT_DIGEST_KEYS,
    )
    _validate_tool_identities(record.get("tool_identities"), "run tools")
    _sha256_text(
        record.get("implementation_source_sha256"), "run implementation digest"
    )
    counts = _mapping(record.get("counts"), "run counts")
    _exact_keys(counts, _RUN_COUNT_KEYS, "run counts")
    for key in sorted(_RUN_COUNT_KEYS):
        _positive_int(counts.get(key), f"run {key}")
    _validate_digest_mapping(
        record.get("output_digests"),
        "run output digests",
        expected_keys=_RUN_OUTPUT_DIGEST_KEYS,
    )
    return record


def verify_controlled_media(
    *,
    configuration_path: pathlib.Path,
    expected_configuration_sha256: str,
    pilot_index_path: pathlib.Path,
    expected_pilot_index_sha256: str,
    media_receipt_path: pathlib.Path,
    expected_media_receipt_sha256: str,
    expected_media_set_sha256: str,
    media_root: pathlib.Path,
    nuisance_configuration_path: pathlib.Path,
    expected_nuisance_configuration_sha256: str,
    nuisance_implementation_sources: Mapping[str, pathlib.Path],
    expected_nuisance_implementation_bundle_sha256: str,
    nuisance_contract_path: pathlib.Path,
    expected_nuisance_contract_sha256: str,
    source_gate_configuration_path: pathlib.Path,
    expected_source_gate_configuration_sha256: str,
    source_gate_output_path: pathlib.Path,
    expected_source_gate_output_sha256: str,
    source_gate_implementation_sources: Mapping[str, pathlib.Path],
    expected_source_gate_implementation_source_sha256: Mapping[str, str],
    expected_source_gate_implementation_bundle_sha256: str,
    source_gate_transcript_path: pathlib.Path,
    expected_source_gate_transcript_sha256: str,
    temporal_implementation_sources: Mapping[str, pathlib.Path],
    expected_temporal_implementation_source_sha256: Mapping[str, str],
    expected_temporal_implementation_bundle_sha256: str,
    temporal_calibration_path: pathlib.Path,
    expected_temporal_calibration_sha256: str,
    ffmpeg_path: pathlib.Path,
    ffprobe_path: pathlib.Path,
    output_root: pathlib.Path,
) -> dict[str, Any]:
    """Reauthenticate inputs, outputs, anchors, partitions, streams, and diagnostics."""

    _validate_output_root_separation(media_root, output_root)
    context = _load_context(
        configuration_path=configuration_path,
        expected_configuration_sha256=expected_configuration_sha256,
        pilot_index_path=pilot_index_path,
        expected_pilot_index_sha256=expected_pilot_index_sha256,
        media_receipt_path=media_receipt_path,
        expected_media_receipt_sha256=expected_media_receipt_sha256,
        expected_media_set_sha256=expected_media_set_sha256,
        media_root=media_root,
        nuisance_configuration_path=nuisance_configuration_path,
        expected_nuisance_configuration_sha256=(expected_nuisance_configuration_sha256),
        nuisance_implementation_sources=nuisance_implementation_sources,
        expected_nuisance_implementation_bundle_sha256=(
            expected_nuisance_implementation_bundle_sha256
        ),
        nuisance_contract_path=nuisance_contract_path,
        expected_nuisance_contract_sha256=expected_nuisance_contract_sha256,
        source_gate_configuration_path=source_gate_configuration_path,
        expected_source_gate_configuration_sha256=(
            expected_source_gate_configuration_sha256
        ),
        source_gate_output_path=source_gate_output_path,
        expected_source_gate_output_sha256=expected_source_gate_output_sha256,
        source_gate_implementation_sources=source_gate_implementation_sources,
        expected_source_gate_implementation_source_sha256=(
            expected_source_gate_implementation_source_sha256
        ),
        expected_source_gate_implementation_bundle_sha256=(
            expected_source_gate_implementation_bundle_sha256
        ),
        source_gate_transcript_path=source_gate_transcript_path,
        expected_source_gate_transcript_sha256=(expected_source_gate_transcript_sha256),
        temporal_implementation_sources=temporal_implementation_sources,
        expected_temporal_implementation_source_sha256=(
            expected_temporal_implementation_source_sha256
        ),
        expected_temporal_implementation_bundle_sha256=(
            expected_temporal_implementation_bundle_sha256
        ),
        temporal_calibration_path=temporal_calibration_path,
        expected_temporal_calibration_sha256=expected_temporal_calibration_sha256,
    )
    _, ffmpeg_identity = _tool_identity(ffmpeg_path, "ffmpeg")
    ffprobe, ffprobe_identity = _tool_identity(ffprobe_path, "ffprobe")
    tool_identities = {"ffmpeg": ffmpeg_identity, "ffprobe": ffprobe_identity}
    root = pathlib.Path(output_root)
    if root.is_symlink() or not root.is_dir():
        raise MediaEditValidationError(
            "output root must be a regular non-symlink directory"
        )
    root = root.resolve(strict=True)
    expected_root_entries = {
        "media",
        "edit_index.json",
        "post_edit_index.json",
        "validation.json",
        "run_record.json",
    }
    if {entry.name for entry in root.iterdir()} != expected_root_entries:
        raise MediaEditValidationError("output root membership differs")
    media_output = root / "media"
    if media_output.is_symlink() or not media_output.is_dir():
        raise MediaEditValidationError("output media directory differs")

    edit_index_payload, edit_index_value = _read_json(
        root / "edit_index.json", "edit index"
    )
    post_edit_payload, post_edit_value = _read_json(
        root / "post_edit_index.json", "post-edit index"
    )
    validation_payload, validation_value = _read_json(
        root / "validation.json", "validation record"
    )
    _, run_value = _read_json(root / "run_record.json", "run record")
    run_record = _validate_run_schema(run_value)
    if run_record.get("status") != "complete":
        raise MediaEditValidationError("run record is incomplete")
    if not _same_json_types_and_values(
        run_record.get("input_digests"), context["input_digests"]
    ):
        raise MediaEditValidationError("run input binding differs")
    if not _same_json_types_and_values(
        run_record.get("tool_identities"), tool_identities
    ):
        raise MediaEditValidationError("run tool identity differs")
    if run_record.get("implementation_source_sha256") != _implementation_digest():
        raise MediaEditValidationError("run implementation digest differs")
    output_digests = _mapping(run_record.get("output_digests"), "run output digests")
    if (
        output_digests.get("edit_index_sha256")
        != hashlib.sha256(edit_index_payload).hexdigest()
    ):
        raise MediaEditValidationError("edit-index digest differs")
    if (
        output_digests.get("post_edit_index_sha256")
        != hashlib.sha256(post_edit_payload).hexdigest()
    ):
        raise MediaEditValidationError("post-edit-index digest differs")
    if (
        output_digests.get("validation_sha256")
        != hashlib.sha256(validation_payload).hexdigest()
    ):
        raise MediaEditValidationError("validation-record digest differs")

    edit_index = _validate_edit_index_schema(edit_index_value)
    if edit_index.get("status") != "complete":
        raise MediaEditValidationError("edit index is incomplete")
    if not _same_json_types_and_values(
        edit_index.get("input_digests"), context["input_digests"]
    ):
        raise MediaEditValidationError("edit-index input binding differs")
    if not _same_json_types_and_values(
        edit_index.get("tool_identities"), tool_identities
    ):
        raise MediaEditValidationError("edit-index tool identity differs")
    if (
        edit_index.get("nuisance_implementation_source_sha256")
        != context["nuisance_implementation_source_sha256"]
    ):
        raise MediaEditValidationError("edit-index nuisance source binding differs")
    if (
        edit_index.get("source_gate_implementation_source_sha256")
        != context["source_gate"]["implementation_source_sha256"]
    ):
        raise MediaEditValidationError("edit-index source-gate binding differs")
    if (
        edit_index.get("temporal_implementation_source_sha256")
        != context["temporal_implementation_source_sha256"]
    ):
        raise MediaEditValidationError("edit-index temporal source binding differs")
    if not _same_json_types_and_values(
        edit_index.get("configuration"), context["configuration"]
    ):
        raise MediaEditValidationError("edit-index configuration differs")
    edits_value = edit_index.get("edits")
    if not isinstance(edits_value, list) or len(edits_value) != len(context["plan"]):
        raise MediaEditValidationError("edit-index entries differ")
    expected_names = {str(item["filename"]) for item in context["plan"]}
    if {entry.name for entry in media_output.iterdir()} != expected_names:
        raise MediaEditValidationError("edited-media membership differs")

    source_root = pathlib.Path(media_root).resolve(strict=True)
    try:
        source_diagnostics = _probe_sources(
            ffprobe_path=ffprobe,
            media_root=source_root,
            media_records=context["media_records"],
        )
        source_profiles = _probe_source_profiles(
            ffprobe_path=ffprobe,
            media_root=source_root,
            media_records=context["media_records"],
            configuration=context["configuration"],
        )
    except NuisanceValidationError as exc:
        raise MediaEditValidationError(str(exc)) from exc
    verified_edits: list[dict[str, Any]] = []
    output_diagnostics: dict[str, dict[str, Any]] = {}
    for expected_edit, raw_edit in zip(context["plan"], edits_value):
        edit = _mapping(raw_edit, "edit-index entry")
        output_record = _mapping(edit.get("output"), "edit output")
        without_output = dict(edit)
        without_output.pop("output", None)
        if not _same_json_types_and_values(without_output, expected_edit):
            raise MediaEditValidationError(
                "edit metadata, anchor, or partition differs"
            )
        expected_output_fields = {
            "relative_path",
            "size_bytes",
            "sha256",
            "diagnostic_sha256",
            "profile_sha256",
        }
        _exact_keys(output_record, expected_output_fields, "edit output")
        relative_path = output_record.get("relative_path")
        if relative_path != f"media/{expected_edit['filename']}":
            raise MediaEditValidationError("edited-media relative path differs")
        output_path = media_output / str(expected_edit["filename"])
        if output_path.is_symlink() or not output_path.is_file():
            raise MediaEditValidationError(
                "edited media must be a regular non-symlink file"
            )
        file_stat = output_path.stat()
        if stat.S_IMODE(file_stat.st_mode) & 0o222:
            raise MediaEditValidationError("edited media is writable")
        if file_stat.st_size != _positive_int(
            output_record.get("size_bytes"), "edited-media size"
        ):
            raise MediaEditValidationError("edited-media size differs")
        if _sha256_path(output_path) != _sha256_text(
            output_record.get("sha256"), "edited-media digest"
        ):
            raise MediaEditValidationError("edited-media digest differs")
        try:
            diagnostic = probe_media_file(
                ffprobe, output_path, expected_size_bytes=file_stat.st_size
            )
            profile = _probe_media_profile(
                ffprobe, output_path, context["configuration"]
            )
        except NuisanceValidationError as exc:
            raise MediaEditValidationError(str(exc)) from exc
        if _canonical_digest(diagnostic) != _sha256_text(
            output_record.get("diagnostic_sha256"), "output diagnostic digest"
        ):
            raise MediaEditValidationError("output diagnostic digest differs")
        if _canonical_digest(profile) != _sha256_text(
            output_record.get("profile_sha256"), "output profile digest"
        ):
            raise MediaEditValidationError("output profile digest differs")
        expected_duration = _output_duration_for_edit(expected_edit, source_diagnostics)
        target_stream_padded = _target_stream_needs_padding(
            expected_edit,
            source_diagnostics[expected_edit["target_video_id"]],
            expected_duration,
        )
        _validate_output_probe(
            diagnostic,
            target_diagnostic=source_diagnostics[expected_edit["target_video_id"]],
            expected_duration=expected_duration,
            condition=str(expected_edit["condition"]),
            source_orientation=expected_edit["source_orientation"],
            target_stream_padded=target_stream_padded,
            configuration=context["configuration"],
        )
        _validate_output_profile(
            profile,
            target_profile=source_profiles[expected_edit["target_video_id"]],
            condition=str(expected_edit["condition"]),
            source_orientation=expected_edit["source_orientation"],
            target_stream_padded=target_stream_padded,
            configuration=context["configuration"],
        )
        verified_edits.append(dict(edit))
        output_diagnostics[expected_edit["output_id"]] = diagnostic

    output_set_sha256 = _output_set_digest(verified_edits)
    if edit_index.get("output_set_sha256") != output_set_sha256:
        raise MediaEditValidationError("edit-index output-set digest differs")
    if output_digests.get("output_set_sha256") != output_set_sha256:
        raise MediaEditValidationError("run output-set digest differs")
    post_edit_records = _validate_post_edit_index_schema(post_edit_value)
    expected_post_edit = _post_edit_index(verified_edits)
    if not _same_json_types_and_values(post_edit_records, expected_post_edit):
        raise MediaEditValidationError("post-edit index differs")
    try:
        nuisance_reports = _evaluate_nuisance_by_orientation(
            expected_post_edit,
            source_diagnostics,
            output_diagnostics,
            context["nuisance_configuration"],
        )
    except NuisanceValidationError as exc:
        raise MediaEditValidationError(str(exc)) from exc
    source_hash_integrity = _reauthenticate_source_media(
        media_root=source_root,
        media_records=context["media_records"],
        expected_media_set_sha256=context["input_digests"]["media_set_sha256"],
    )
    expected_validation = _validation_record(
        edit_index_sha256=hashlib.sha256(edit_index_payload).hexdigest(),
        post_edit_index_sha256=hashlib.sha256(post_edit_payload).hexdigest(),
        binding_digests=context["input_digests"],
        nuisance_implementation_source_sha256=context[
            "nuisance_implementation_source_sha256"
        ],
        source_gate_implementation_source_sha256=context["source_gate"][
            "implementation_source_sha256"
        ],
        temporal_implementation_source_sha256=context[
            "temporal_implementation_source_sha256"
        ],
        output_set_sha256=output_set_sha256,
        edit_count=len(verified_edits),
        post_edit_pair_count=len(expected_post_edit),
        nuisance_reports=nuisance_reports,
        source_hash_integrity=source_hash_integrity,
    )
    validation_record = _validate_validation_schema(validation_value)
    if not _same_json_types_and_values(validation_record, expected_validation):
        raise MediaEditValidationError("validation record differs from recomputation")
    target_count = len({edit["target_id"] for edit in verified_edits})
    clean_original_count = sum(
        edit["condition"] == "clean_original" for edit in verified_edits
    )
    expected_edit_counts = {
        "target_count": target_count,
        "edit_count": len(verified_edits),
        "clean_original_count": clean_original_count,
    }
    if not _same_json_types_and_values(edit_index.get("counts"), expected_edit_counts):
        raise MediaEditValidationError("edit-index counts differ")
    if edit_index.get("continuation_allowed") is not bool(
        expected_validation["continuation_allowed"]
    ):
        raise MediaEditValidationError("edit-index continuation decision differs")
    counts = {
        "edit_count": len(verified_edits),
        "post_edit_pair_count": len(expected_post_edit),
        "source_video_count": len(context["media_records"]),
        "target_count": target_count,
        "clean_original_count": clean_original_count,
    }
    expected_run = _run_record(
        input_digests=context["input_digests"],
        tool_identities=tool_identities,
        counts=counts,
        output_digests={
            "edit_index_sha256": hashlib.sha256(edit_index_payload).hexdigest(),
            "post_edit_index_sha256": hashlib.sha256(post_edit_payload).hexdigest(),
            "validation_sha256": hashlib.sha256(validation_payload).hexdigest(),
            "output_set_sha256": output_set_sha256,
        },
        continuation_allowed=bool(expected_validation["continuation_allowed"]),
        nuisance_implementation_source_sha256=context[
            "nuisance_implementation_source_sha256"
        ],
        source_gate_implementation_source_sha256=context["source_gate"][
            "implementation_source_sha256"
        ],
        temporal_implementation_source_sha256=context[
            "temporal_implementation_source_sha256"
        ],
    )
    if not _same_json_types_and_values(dict(run_record), expected_run):
        raise MediaEditValidationError("run record differs from recomputation")
    return expected_run
