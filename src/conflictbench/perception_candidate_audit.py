"""Deterministic structural audit for Perception Test MC-QA candidate pools.

This module reads only the pinned train and validation annotation archives. It
does not read media and its structural counts do not establish that either
audio or video is sufficient to answer a question.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import pathlib
import re
import tempfile
import unicodedata
import urllib.parse
import zipfile
from collections.abc import Mapping, Sequence
from typing import Any


class AuditError(ValueError):
    """Raised when a pinned input or annotation record fails validation."""


_CONFIG_KEYS = {
    "schema_version",
    "audit",
    "purpose",
    "source_repository",
    "normalization",
    "eligibility",
    "expected_snapshot_counts",
    "archives",
}
_ARCHIVE_KEYS = {
    "url",
    "generation",
    "size_bytes",
    "sha256",
    "member",
    "metadata_split",
}
_TOP_VIDEO_KEYS = {"metadata", "mc_question"}
_METADATA_KEYS = {
    "split",
    "video_id",
    "frame_rate",
    "num_frames",
    "resolution",
    "audio_samples",
    "audio_sample_rate",
    "is_cup_game",
    "is_camera_moving",
}
_QUESTION_KEYS = {"id", "question", "options", "answer_id", "area", "reasoning", "tag"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TEST_TOKEN_RE = re.compile(r"(?:^|[_.-])test(?:[_.-]|$)", re.IGNORECASE)
_CANONICAL_REPOSITORY_URL = "https://github.com/google-deepmind/perception_test"
_PINNED_REPOSITORY_COMMIT = "3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf"
_PINNED_ARCHIVE_IDENTITIES = {
    "train": {
        "url": "https://storage.googleapis.com/dm-perception-test/zip_data/mc_question_train_annotations.zip?generation=1686575054862298",
        "generation": "1686575054862298",
        "member": "mc_question_train.json",
        "metadata_split": "train",
    },
    "validation": {
        "url": "https://storage.googleapis.com/dm-perception-test/zip_data/mc_question_valid_annotations.zip?generation=1686575054970958",
        "generation": "1686575054970958",
        "member": "mc_question_valid.json",
        "metadata_split": "valid",
    },
}
_EXPECTED_SNAPSHOT_COUNT_KEYS = {
    "split_minimum_two_answer_group_count",
    "cross_split_minimum_two_answer_group_count",
    "cross_split_all_three_option_group_count",
    "duplicate_option_exclusion_count",
}


def normalize_text(value: str) -> str:
    """Apply NFC and collapse whitespace without changing case or punctuation."""

    if not isinstance(value, str):
        raise AuditError("annotation text must be a string")
    return " ".join(unicodedata.normalize("NFC", value).split())


def _nonempty_normalized(value: Any, field: str) -> str:
    normalized = normalize_text(value)
    if not normalized:
        raise AuditError(f"{field} must not be empty after normalization")
    return normalized


def candidate_group_key(question: Mapping[str, Any]) -> tuple[tuple[Any, ...], str]:
    """Return the canonical group key and answer text for one MC-QA entry."""

    options_value = question.get("options")
    if not isinstance(options_value, list) or len(options_value) < 2:
        raise AuditError("options must be a list with at least two entries")
    options = tuple(_nonempty_normalized(value, "option") for value in options_value)
    if len(set(options)) != len(options):
        raise AuditError("duplicate option text after normalization")

    answer_id = question.get("answer_id")
    if isinstance(answer_id, bool) or not isinstance(answer_id, int):
        raise AuditError("answer_id must be an integer")
    if not 0 <= answer_id < len(options):
        raise AuditError("answer_id is outside the option range")
    answer_text = options[answer_id]

    tags_value = question.get("tag")
    if not isinstance(tags_value, list) or not tags_value:
        raise AuditError("tag must be a nonempty list")
    tags = tuple(_nonempty_normalized(value, "tag") for value in tags_value)
    if len(set(tags)) != len(tags):
        raise AuditError("tag entries must be unique after normalization")

    key = (
        _nonempty_normalized(question.get("question"), "question"),
        tuple(sorted(options)),
        _nonempty_normalized(question.get("area"), "area"),
        _nonempty_normalized(question.get("reasoning"), "reasoning"),
        tuple(sorted(tags)),
    )
    return key, answer_text


def _reject_json_constant(value: str) -> None:
    raise AuditError(f"non-finite JSON number is not allowed: {value}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_bytes(data: bytes, source: str) -> Any:
    try:
        text = data.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, AuditError) as exc:
        if isinstance(exc, AuditError):
            raise
        raise AuditError(f"{source} is not strict UTF-8 JSON: {exc}") from exc


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise AuditError(f"{field} keys differ; missing={missing}, unknown={unknown}")


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise AuditError(f"{field} must be an object")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AuditError(f"{field} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AuditError(f"{field} must be a nonnegative integer")
    return value


def _require_positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditError(f"{field} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise AuditError(f"{field} must be finite and positive")
    return converted


def _validate_config(config: Any) -> Mapping[str, Any]:
    config = _require_mapping(config, "configuration")
    _exact_keys(config, _CONFIG_KEYS, "configuration")
    if config["schema_version"] != 2:
        raise AuditError("schema_version must equal 2")
    if config["audit"] != "perception_test_mcqa_candidate_pool":
        raise AuditError("unexpected audit identifier")
    if config["purpose"] != "candidate_pool_feasibility_only":
        raise AuditError("unexpected audit purpose")

    repository = _require_mapping(config["source_repository"], "source_repository")
    _exact_keys(
        repository,
        {"url", "commit", "annotation_license", "license_url"},
        "source_repository",
    )
    if repository["url"] != _CANONICAL_REPOSITORY_URL:
        raise AuditError("source_repository.url is not the canonical official URL")
    if repository["commit"] != _PINNED_REPOSITORY_COMMIT:
        raise AuditError("source_repository.commit is not the pinned official commit")
    if repository["annotation_license"] != "CC-BY-4.0":
        raise AuditError("source_repository.annotation_license must be CC-BY-4.0")
    if (
        repository["license_url"]
        != "https://creativecommons.org/licenses/by/4.0/legalcode"
    ):
        raise AuditError("source_repository.license_url is unexpected")

    normalization = _require_mapping(config["normalization"], "normalization")
    expected_normalization = {
        "unicode": "NFC",
        "whitespace": "strip_and_collapse",
        "case_sensitive": True,
        "punctuation_sensitive": True,
    }
    if normalization != expected_normalization:
        raise AuditError(
            "normalization settings must specify NFC plus whitespace-only normalization"
        )

    eligibility = _require_mapping(config["eligibility"], "eligibility")
    _exact_keys(
        eligibility,
        {
            "min_unique_videos_per_supported_answer_per_split",
            "min_supported_answers_per_group",
            "require_same_supported_answer_vocabulary_across_splits",
            "report_all_three_option_tier",
        },
        "eligibility",
    )
    _require_positive_int(
        eligibility["min_unique_videos_per_supported_answer_per_split"],
        "eligibility.min_unique_videos_per_supported_answer_per_split",
    )
    if eligibility["min_supported_answers_per_group"] != 2:
        raise AuditError("eligibility.min_supported_answers_per_group must equal 2")
    if (
        eligibility["require_same_supported_answer_vocabulary_across_splits"]
        is not True
    ):
        raise AuditError("same supported-answer vocabulary must be required")
    if eligibility["report_all_three_option_tier"] is not True:
        raise AuditError("the all-three-option tier must be reported")

    expected_counts = _require_mapping(
        config["expected_snapshot_counts"], "expected_snapshot_counts"
    )
    _exact_keys(
        expected_counts,
        _EXPECTED_SNAPSHOT_COUNT_KEYS,
        "expected_snapshot_counts",
    )
    for field in (
        "split_minimum_two_answer_group_count",
        "duplicate_option_exclusion_count",
    ):
        split_counts = _require_mapping(expected_counts[field], field)
        _exact_keys(split_counts, {"train", "validation"}, field)
        for split_name in ("train", "validation"):
            _require_nonnegative_int(split_counts[split_name], f"{field}.{split_name}")
    _require_nonnegative_int(
        expected_counts["cross_split_minimum_two_answer_group_count"],
        "expected_snapshot_counts.cross_split_minimum_two_answer_group_count",
    )
    _require_nonnegative_int(
        expected_counts["cross_split_all_three_option_group_count"],
        "expected_snapshot_counts.cross_split_all_three_option_group_count",
    )

    archives = _require_mapping(config["archives"], "archives")
    if set(archives) != {"train", "validation"}:
        raise AuditError("archives must contain exactly train and validation")
    for split_name, identity in _PINNED_ARCHIVE_IDENTITIES.items():
        record = _require_mapping(archives[split_name], f"archives.{split_name}")
        _exact_keys(record, _ARCHIVE_KEYS, f"archives.{split_name}")
        if record["url"] != identity["url"]:
            raise AuditError(
                f"archives.{split_name}.url is not the pinned official URL"
            )
        parsed = urllib.parse.urlsplit(record["url"])
        basename = pathlib.PurePosixPath(parsed.path).name
        if _TEST_TOKEN_RE.search(basename):
            raise AuditError("test archive references are forbidden")
        try:
            query = urllib.parse.parse_qs(parsed.query, strict_parsing=True)
        except ValueError as exc:
            raise AuditError(f"archives.{split_name}.url has an invalid query") from exc
        generation = record["generation"]
        if generation != identity["generation"]:
            raise AuditError(
                f"archives.{split_name}.generation is not the pinned official generation"
            )
        if query.get("generation") != [generation]:
            raise AuditError(f"archives.{split_name}.generation must match the URL")
        _require_positive_int(record["size_bytes"], f"archives.{split_name}.size_bytes")
        if not isinstance(record["sha256"], str) or not _SHA256_RE.fullmatch(
            record["sha256"]
        ):
            raise AuditError(f"archives.{split_name}.sha256 must be lowercase SHA256")
        if (
            not isinstance(record["member"], str)
            or pathlib.PurePosixPath(record["member"]).name != record["member"]
        ):
            raise AuditError(f"archives.{split_name}.member must be a plain filename")
        if _TEST_TOKEN_RE.search(record["member"]):
            raise AuditError("test archive members are forbidden")
        if record["member"] != identity["member"]:
            raise AuditError(f"archives.{split_name}.member is not the pinned member")
        if record["metadata_split"] != identity["metadata_split"]:
            raise AuditError(f"archives.{split_name}.metadata_split is unexpected")
    return config


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_pinned_archive(
    path: pathlib.Path, lock: Mapping[str, Any], split_name: str
) -> Any:
    if not path.is_file():
        raise AuditError(f"{split_name} archive is not a regular file")
    if path.stat().st_size != lock["size_bytes"]:
        raise AuditError(f"{split_name} archive size does not match the pinned value")
    observed_sha256 = _sha256(path)
    if observed_sha256 != lock["sha256"]:
        raise AuditError(f"{split_name} archive SHA256 does not match the pinned value")

    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].filename != lock["member"]:
                raise AuditError(
                    f"{split_name} archive must contain only {lock['member']}"
                )
            member = members[0]
            if member.is_dir() or member.flag_bits & 0x1:
                raise AuditError(
                    f"{split_name} archive member must be an unencrypted file"
                )
            payload = archive.read(member)
    except (OSError, zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        raise AuditError(f"{split_name} archive cannot be read: {exc}") from exc
    return _load_json_bytes(payload, f"{split_name} archive member")


def _validate_metadata(
    metadata: Any, *, expected_video_id: str, expected_split: str
) -> None:
    metadata = _require_mapping(metadata, f"metadata for {expected_video_id}")
    _exact_keys(metadata, _METADATA_KEYS, f"metadata for {expected_video_id}")
    if metadata["video_id"] != expected_video_id:
        raise AuditError(
            f"metadata video_id differs from its top-level key: {expected_video_id}"
        )
    if metadata["split"] != expected_split:
        raise AuditError(
            f"metadata split differs from the pinned split: {expected_video_id}"
        )
    _require_positive_number(
        metadata["frame_rate"], f"frame_rate for {expected_video_id}"
    )
    _require_positive_int(metadata["num_frames"], f"num_frames for {expected_video_id}")
    resolution = metadata["resolution"]
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise AuditError(f"resolution for {expected_video_id} must have two entries")
    for index, value in enumerate(resolution):
        _require_positive_int(value, f"resolution[{index}] for {expected_video_id}")
    _require_nonnegative_int(
        metadata["audio_samples"], f"audio_samples for {expected_video_id}"
    )
    _require_positive_number(
        metadata["audio_sample_rate"], f"audio_sample_rate for {expected_video_id}"
    )
    for field in ("is_cup_game", "is_camera_moving"):
        value = metadata[field]
        if isinstance(value, bool) or value not in (0, 1):
            raise AuditError(f"{field} for {expected_video_id} must be 0 or 1")


def _validate_question_schema(
    question: Any, *, expected_id: int, video_id: str
) -> Mapping[str, Any]:
    question = _require_mapping(question, f"question {expected_id} for {video_id}")
    _exact_keys(question, _QUESTION_KEYS, f"question {expected_id} for {video_id}")
    question_id = question["id"]
    if (
        isinstance(question_id, bool)
        or not isinstance(question_id, int)
        or question_id != expected_id
    ):
        raise AuditError(f"question ids for {video_id} must be consecutive from zero")
    options = question["options"]
    if not isinstance(options, list) or len(options) != 3:
        raise AuditError(
            f"question {expected_id} for {video_id} must have exactly three options"
        )
    answer_id = question["answer_id"]
    if (
        isinstance(answer_id, bool)
        or not isinstance(answer_id, int)
        or not 0 <= answer_id < 3
    ):
        raise AuditError(
            f"answer_id for question {expected_id} in {video_id} is invalid"
        )
    if not isinstance(question["tag"], list) or not question["tag"]:
        raise AuditError(
            f"tag for question {expected_id} in {video_id} must be a nonempty list"
        )
    return question


def _group_key_record(key: tuple[Any, ...]) -> dict[str, Any]:
    question, options, area, reasoning, tags = key
    return {
        "question": question,
        "option_vocabulary": list(options),
        "area": area,
        "reasoning": reasoning,
        "tags": list(tags),
    }


def _group_key_sha256(key: tuple[Any, ...]) -> str:
    encoded = json.dumps(
        _group_key_record(key),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _collect_split(
    payload: Any, *, expected_split: str
) -> tuple[dict[str, Any], dict[Any, Any]]:
    if not isinstance(payload, dict) or not payload:
        raise AuditError(f"{expected_split} annotations must be a nonempty object")

    groups: dict[Any, dict[str, Any]] = {}
    raw_question_count = 0
    structurally_valid_question_count = 0
    duplicate_option_exclusions = []
    structurally_valid_videos: set[str] = set()

    for video_id in sorted(payload):
        if (
            not isinstance(video_id, str)
            or not video_id
            or normalize_text(video_id) != video_id
        ):
            raise AuditError("video ids must be nonempty normalized strings")
        record = _require_mapping(payload[video_id], f"video record {video_id}")
        _exact_keys(record, _TOP_VIDEO_KEYS, f"video record {video_id}")
        _validate_metadata(
            record["metadata"],
            expected_video_id=video_id,
            expected_split=expected_split,
        )
        questions = record["mc_question"]
        if not isinstance(questions, list) or not questions:
            raise AuditError(f"mc_question for {video_id} must be a nonempty list")
        for expected_id, raw_question in enumerate(questions):
            question = _validate_question_schema(
                raw_question, expected_id=expected_id, video_id=video_id
            )
            raw_question_count += 1
            try:
                key, answer_text = candidate_group_key(question)
            except AuditError as exc:
                if str(exc) == "duplicate option text after normalization":
                    duplicate_option_exclusions.append(
                        {
                            "video_id": video_id,
                            "question_id": expected_id,
                            "reason": "duplicate_option_text_after_normalization",
                        }
                    )
                    continue
                raise AuditError(
                    f"question {expected_id} for {video_id}: {exc}"
                ) from exc
            structurally_valid_question_count += 1
            structurally_valid_videos.add(video_id)
            group = groups.setdefault(
                key,
                {
                    "question_count": 0,
                    "videos": set(),
                    "answer_videos": collections.defaultdict(set),
                },
            )
            group["question_count"] += 1
            group["videos"].add(video_id)
            group["answer_videos"][answer_text].add(video_id)

    minimum_two_answer_keys = [
        key for key, group in groups.items() if len(group["answer_videos"]) >= 2
    ]
    minimum_two_answer_videos = set().union(
        *(groups[key]["videos"] for key in minimum_two_answer_keys)
    )
    summary = {
        "at_least_two_supported_answers": {
            "group_key_count": len(minimum_two_answer_keys),
            "question_count": sum(
                groups[key]["question_count"] for key in minimum_two_answer_keys
            ),
            "unique_video_count": len(minimum_two_answer_videos),
        },
        "candidate_group_key_count": len(groups),
        "duplicate_option_exclusion_count": len(duplicate_option_exclusions),
        "duplicate_option_exclusions": duplicate_option_exclusions,
        "raw_question_count": raw_question_count,
        "structurally_valid_question_count": structurally_valid_question_count,
        "video_count": len(payload),
        "videos_with_structurally_valid_questions": len(structurally_valid_videos),
    }
    return summary, groups


def _cross_split_tier_report(
    keys: Sequence[tuple[Any, ...]],
    train_groups: Mapping[Any, Any],
    validation_groups: Mapping[Any, Any],
) -> dict[str, Any]:
    candidate_groups = []
    question_count = {"train": 0, "validation": 0}
    video_ids = {"train": set(), "validation": set()}
    cardinality = collections.Counter()
    for key in sorted(keys):
        supported_answers = sorted(train_groups[key]["answer_videos"])
        cardinality[str(len(supported_answers))] += 1
        record = _group_key_record(key)
        record["key_sha256"] = _group_key_sha256(key)
        record["supported_answer_vocabulary"] = supported_answers
        record["answer_video_counts"] = {}
        record["question_counts"] = {}
        record["unique_video_counts"] = {}
        for split_name, groups in (
            ("train", train_groups),
            ("validation", validation_groups),
        ):
            group = groups[key]
            record["answer_video_counts"][split_name] = {
                answer: len(group["answer_videos"].get(answer, ())) for answer in key[1]
            }
            record["question_counts"][split_name] = group["question_count"]
            record["unique_video_counts"][split_name] = len(group["videos"])
            question_count[split_name] += group["question_count"]
            video_ids[split_name].update(group["videos"])
        candidate_groups.append(record)
    return {
        "group_key_count": len(keys),
        "question_count": question_count,
        "unique_video_count": {
            split_name: len(ids) for split_name, ids in video_ids.items()
        },
        "supported_answer_cardinality": dict(sorted(cardinality.items())),
        "candidate_groups": candidate_groups,
    }


def _observed_snapshot_counts(
    train_summary: Mapping[str, Any],
    validation_summary: Mapping[str, Any],
    minimum_two_tier: Mapping[str, Any],
    all_three_tier: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "split_minimum_two_answer_group_count": {
            "train": train_summary["at_least_two_supported_answers"]["group_key_count"],
            "validation": validation_summary["at_least_two_supported_answers"][
                "group_key_count"
            ],
        },
        "cross_split_minimum_two_answer_group_count": minimum_two_tier[
            "group_key_count"
        ],
        "cross_split_all_three_option_group_count": all_three_tier["group_key_count"],
        "duplicate_option_exclusion_count": {
            "train": train_summary["duplicate_option_exclusion_count"],
            "validation": validation_summary["duplicate_option_exclusion_count"],
        },
    }


def audit_archives(
    config_path: pathlib.Path,
    train_archive: pathlib.Path,
    validation_archive: pathlib.Path,
) -> dict[str, Any]:
    """Audit pinned train/validation annotations and return a deterministic report."""

    config_path = pathlib.Path(config_path)
    train_archive = pathlib.Path(train_archive)
    validation_archive = pathlib.Path(validation_archive)
    try:
        config_bytes = config_path.read_bytes()
    except OSError as exc:
        raise AuditError(f"configuration cannot be read: {exc}") from exc
    config = _validate_config(_load_json_bytes(config_bytes, "configuration"))

    archives = config["archives"]
    train_payload = _read_pinned_archive(train_archive, archives["train"], "train")
    validation_payload = _read_pinned_archive(
        validation_archive, archives["validation"], "validation"
    )
    if isinstance(train_payload, dict) and isinstance(validation_payload, dict):
        overlapping_video_ids = sorted(set(train_payload) & set(validation_payload))
        if overlapping_video_ids:
            preview = ", ".join(overlapping_video_ids[:10])
            raise AuditError(
                "train/validation video IDs overlap; "
                f"count={len(overlapping_video_ids)}, examples={preview}"
            )
    train_summary, train_groups = _collect_split(train_payload, expected_split="train")
    validation_summary, validation_groups = _collect_split(
        validation_payload, expected_split="valid"
    )

    shared_keys = set(train_groups) & set(validation_groups)
    minimum_two_answer_keys: list[tuple[Any, ...]] = []
    all_three_option_keys: list[tuple[Any, ...]] = []
    minimum = config["eligibility"]["min_unique_videos_per_supported_answer_per_split"]
    for key in shared_keys:
        option_vocabulary = set(key[1])
        train_answers = set(train_groups[key]["answer_videos"])
        validation_answers = set(validation_groups[key]["answer_videos"])
        if (
            train_answers != validation_answers
            or len(train_answers)
            < config["eligibility"]["min_supported_answers_per_group"]
        ):
            continue
        if all(
            len(groups[key]["answer_videos"].get(answer, ())) >= minimum
            for groups in (train_groups, validation_groups)
            for answer in train_answers
        ):
            minimum_two_answer_keys.append(key)
            if train_answers == option_vocabulary:
                all_three_option_keys.append(key)

    minimum_two_tier = _cross_split_tier_report(
        minimum_two_answer_keys, train_groups, validation_groups
    )
    all_three_tier = _cross_split_tier_report(
        all_three_option_keys, train_groups, validation_groups
    )
    observed_counts = _observed_snapshot_counts(
        train_summary,
        validation_summary,
        minimum_two_tier,
        all_three_tier,
    )
    expected_counts = dict(config["expected_snapshot_counts"])
    if observed_counts != expected_counts:
        raise AuditError(
            "snapshot counts differ from the pinned expectations; "
            f"expected={expected_counts}, observed={observed_counts}"
        )

    source_archives = {
        split_name: {
            "url": archives[split_name]["url"],
            "generation": archives[split_name]["generation"],
            "size_bytes": archives[split_name]["size_bytes"],
            "sha256": archives[split_name]["sha256"],
            "member": archives[split_name]["member"],
            "metadata_split": archives[split_name]["metadata_split"],
        }
        for split_name in ("train", "validation")
    }
    return {
        "schema_version": 2,
        "status": "pass",
        "scope": "annotation_candidate_pool_feasibility_only",
        "limitations": [
            "Structural annotation counts do not prove audio or video sufficiency.",
            "Validation labels were inspected for structural counts, so validation annotations are not fully unseen.",
        ],
        "annotation_use_disclosure": {
            "train_labels_inspected": True,
            "validation_labels_inspected": True,
            "validation_use": "schema_and_candidate_pool_counts_only",
            "validation_media_or_model_outputs_inspected": False,
            "test_annotations_inspected": False,
        },
        "source_attribution": {
            "dataset": "Perception Test: A Diagnostic Benchmark for Multimodal Video Models",
            "authors": "Pătrăucean et al. (2023)",
            "copyright": "Copyright 2022 DeepMind Technologies Limited",
            "materials_license": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/legalcode",
        },
        "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "source_repository": dict(config["source_repository"]),
        "source_archives": source_archives,
        "normalization": dict(config["normalization"]),
        "eligibility": dict(config["eligibility"]),
        "snapshot_counts": observed_counts,
        "splits": {"train": train_summary, "validation": validation_summary},
        "cross_split": {
            "shared_group_key_count": len(shared_keys),
            "minimum_two_supported_answers": minimum_two_tier,
            "all_three_options_supported": all_three_tier,
        },
    }


def _write_json_atomic(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--train-archive", required=True, type=pathlib.Path)
    parser.add_argument("--validation-archive", required=True, type=pathlib.Path)
    parser.add_argument(
        "--output", type=pathlib.Path, help="write deterministic JSON to this path"
    )
    args = parser.parse_args(argv)
    try:
        report = audit_archives(
            args.config, args.train_archive, args.validation_archive
        )
        rendered = (
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        if args.output:
            _write_json_atomic(args.output, report)
        else:
            print(rendered, end="")
    except (AuditError, OSError) as exc:
        parser.exit(2, f"audit failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
