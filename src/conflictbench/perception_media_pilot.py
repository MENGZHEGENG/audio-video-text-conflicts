"""Build a deterministic Perception Test train-split media-pilot index.

Eligibility is derived only from authenticated training annotations. The
builder separately authenticates a structural audit that inspected training
and validation labels, but does not use its cross-split candidate list for
selection or stratification. It never loads media, model outputs, validation
annotations, or test data itself.
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
import zipfile
from collections.abc import Mapping, Sequence
from typing import Any

from conflictbench import perception_candidate_audit
from conflictbench.perception_candidate_audit import (
    AuditError,
    candidate_group_key,
    normalize_text,
)


class PilotConstructionError(ValueError):
    """Raised when an input or construction rule fails closed."""


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CONFIG_KEYS = {
    "schema_version",
    "builder",
    "purpose",
    "seed",
    "implementation_sha256",
    "counts",
    "construction",
    "input_locks",
    "source",
    "gate_contract",
}
_COUNT_KEYS = {
    "target_count",
    "pair_count",
    "pair_count_bounds",
    "partition_target_counts",
    "expected_eligible_key_count",
    "expected_eligible_key_count_by_supported_answer_count",
}
_CONSTRUCTION_KEYS = {
    "official_split",
    "min_supported_answers_per_key",
    "min_unique_train_videos_per_supported_answer",
    "pair_roles",
    "require_every_eligible_key",
    "prefer_unique_donors",
    "target_and_donor_video_disjoint_when_possible",
    "component_partition_lock",
}
_ARCHIVE_KEYS = {
    "url",
    "generation",
    "size_bytes",
    "sha256",
    "member",
    "metadata_split",
}
_SOURCE_KEYS = {
    "repository_url",
    "repository_commit",
    "annotation_license",
    "license_url",
    "citation",
}
_GATE_KEYS = {
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
}
_AUDIT_KEYS = {
    "schema_version",
    "status",
    "scope",
    "limitations",
    "annotation_use_disclosure",
    "source_attribution",
    "configuration_sha256",
    "source_repository",
    "source_archives",
    "normalization",
    "eligibility",
    "snapshot_counts",
    "splits",
    "cross_split",
}
_TIER_KEYS = {
    "group_key_count",
    "question_count",
    "unique_video_count",
    "supported_answer_cardinality",
    "candidate_groups",
}
_GROUP_KEYS = {
    "question",
    "option_vocabulary",
    "area",
    "reasoning",
    "tags",
    "key_sha256",
    "supported_answer_vocabulary",
    "answer_video_counts",
    "question_counts",
    "unique_video_counts",
}
_SPLIT_SUMMARY_KEYS = {
    "at_least_two_supported_answers",
    "candidate_group_key_count",
    "duplicate_option_exclusion_count",
    "duplicate_option_exclusions",
    "raw_question_count",
    "structurally_valid_question_count",
    "video_count",
    "videos_with_structurally_valid_questions",
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
_QUESTION_KEYS = {
    "id",
    "question",
    "options",
    "answer_id",
    "area",
    "reasoning",
    "tag",
}
_PAIR_ROLES = ["same_answer_nuisance", "opposite_answer_candidate"]
_PARTITIONS = ["scorer_fit", "threshold_calibration", "pilot_gate"]
_NORMALIZATION = {
    "unicode": "NFC",
    "whitespace": "strip_and_collapse",
    "case_sensitive": True,
    "punctuation_sensitive": True,
}
_CANONICAL_REPOSITORY_URL = "https://github.com/google-deepmind/perception_test"
_PINNED_REPOSITORY_COMMIT = "3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf"
_PINNED_ARCHIVE_IDENTITIES = {
    "train": {
        "url": (
            "https://storage.googleapis.com/dm-perception-test/zip_data/"
            "mc_question_train_annotations.zip?generation=1686575054862298"
        ),
        "generation": "1686575054862298",
        "member": "mc_question_train.json",
        "metadata_split": "train",
    },
    "validation": {
        "url": (
            "https://storage.googleapis.com/dm-perception-test/zip_data/"
            "mc_question_valid_annotations.zip?generation=1686575054970958"
        ),
        "generation": "1686575054970958",
        "member": "mc_question_valid.json",
        "metadata_split": "valid",
    },
}
_IMPLEMENTATION_SOURCE_NAMES = (
    "conflictbench.perception_candidate_audit",
    "conflictbench.perception_media_pilot",
)


def _reject_json_constant(value: str) -> None:
    raise PilotConstructionError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PilotConstructionError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json_bytes(data: bytes, source: str) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotConstructionError(f"{source} is not strict UTF-8 JSON: {exc}") from exc


def _read_json(path: pathlib.Path, source: str) -> tuple[bytes, Any]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PilotConstructionError(f"{source} cannot be read: {exc}") from exc
    return data, _load_json_bytes(data, source)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PilotConstructionError(f"{field} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    observed = set(value)
    if observed != expected:
        raise PilotConstructionError(
            f"{field} keys differ; "
            f"missing={sorted(expected - observed)}, unknown={sorted(observed - expected)}"
        )


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PilotConstructionError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PilotConstructionError(f"{field} must be a nonnegative integer")
    return value


def _sha256_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise PilotConstructionError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_provenance_for_paths(
    source_paths: Mapping[str, pathlib.Path],
) -> dict[str, Any]:
    """Hash the builder and candidate-audit sources with stable identities."""

    observed_names = set(source_paths)
    if observed_names != set(_IMPLEMENTATION_SOURCE_NAMES):
        raise PilotConstructionError(
            "implementation source names differ; "
            f"missing={sorted(set(_IMPLEMENTATION_SOURCE_NAMES) - observed_names)}, "
            f"unknown={sorted(observed_names - set(_IMPLEMENTATION_SOURCE_NAMES))}"
        )
    combined = hashlib.sha256()
    source_sha256: dict[str, str] = {}
    for source_name in _IMPLEMENTATION_SOURCE_NAMES:
        source_path = pathlib.Path(source_paths[source_name])
        try:
            source_bytes = source_path.read_bytes()
        except OSError as exc:
            raise PilotConstructionError(
                f"implementation source cannot be read: {source_name}: {exc}"
            ) from exc
        source_sha256[source_name] = hashlib.sha256(source_bytes).hexdigest()
        combined.update(source_name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(source_bytes)
        combined.update(b"\0")
    return {
        "implementation_sha256": combined.hexdigest(),
        "source_sha256": source_sha256,
    }


def implementation_provenance() -> dict[str, Any]:
    """Return provenance for every source that determines candidate selection."""

    return _implementation_provenance_for_paths(
        {
            "conflictbench.perception_candidate_audit": pathlib.Path(
                perception_candidate_audit.__file__
            ).resolve(),
            "conflictbench.perception_media_pilot": pathlib.Path(__file__).resolve(),
        }
    )


def implementation_sha256() -> str:
    """Return the combined builder and candidate-audit implementation digest."""

    return str(implementation_provenance()["implementation_sha256"])


def _canonical_digest(value: Mapping[str, Any]) -> str:
    data = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _rank(seed: int, domain: str, *values: Any) -> str:
    encoded = json.dumps(
        [seed, domain, *values],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_archive_lock(
    value: Any, field: str, split_name: str
) -> Mapping[str, Any]:
    value = _mapping(value, field)
    _exact_keys(value, _ARCHIVE_KEYS, field)
    if not isinstance(value["url"], str) or not value["url"].startswith("https://"):
        raise PilotConstructionError(f"{field}.url must be an HTTPS URL")
    if not isinstance(value["generation"], str) or not value["generation"].isdigit():
        raise PilotConstructionError(f"{field}.generation must contain digits")
    _positive_int(value["size_bytes"], f"{field}.size_bytes")
    _sha256_text(value["sha256"], f"{field}.sha256")
    if (
        not isinstance(value["member"], str)
        or pathlib.PurePosixPath(value["member"]).name != value["member"]
    ):
        raise PilotConstructionError(f"{field}.member must be a plain filename")
    if value["metadata_split"] not in {"train", "valid"}:
        raise PilotConstructionError(f"{field}.metadata_split is invalid")
    identity = _PINNED_ARCHIVE_IDENTITIES[split_name]
    for identity_field, expected in identity.items():
        if value[identity_field] != expected:
            raise PilotConstructionError(
                f"{field} differs from the pinned official identity"
            )
    return value


def _validate_config(value: Any) -> Mapping[str, Any]:
    config = _mapping(value, "configuration")
    _exact_keys(config, _CONFIG_KEYS, "configuration")
    if config["schema_version"] != 1:
        raise PilotConstructionError("configuration.schema_version must equal 1")
    if config["builder"] != "perception_test_training_media_pilot":
        raise PilotConstructionError("configuration.builder is unexpected")
    if config["purpose"] != "training_only_source_sufficiency_construction":
        raise PilotConstructionError("configuration.purpose is unexpected")
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise PilotConstructionError("configuration.seed must be a nonnegative integer")
    configured_implementation = _sha256_text(
        config["implementation_sha256"],
        "configuration.implementation_sha256",
    )
    if configured_implementation != implementation_sha256():
        raise PilotConstructionError("configuration implementation SHA-256 differs")

    counts = _mapping(config["counts"], "counts")
    _exact_keys(counts, _COUNT_KEYS, "counts")
    target_count = _positive_int(counts["target_count"], "counts.target_count")
    pair_count = _positive_int(counts["pair_count"], "counts.pair_count")
    bounds = counts["pair_count_bounds"]
    if bounds != [200, 500]:
        raise PilotConstructionError("counts.pair_count_bounds must equal [200, 500]")
    if not 200 <= pair_count <= 500:
        raise PilotConstructionError("counts.pair_count must be between 200 and 500")
    if pair_count != target_count * len(_PAIR_ROLES):
        raise PilotConstructionError("counts.pair_count must equal two pairs per target")
    expected_keys = _positive_int(
        counts["expected_eligible_key_count"],
        "counts.expected_eligible_key_count",
    )
    expected_key_cardinality = _mapping(
        counts["expected_eligible_key_count_by_supported_answer_count"],
        "counts.expected_eligible_key_count_by_supported_answer_count",
    )
    _exact_keys(
        expected_key_cardinality,
        {"2", "3"},
        "counts.expected_eligible_key_count_by_supported_answer_count",
    )
    for cardinality, count in expected_key_cardinality.items():
        _nonnegative_int(
            count,
            "counts.expected_eligible_key_count_by_supported_answer_count."
            f"{cardinality}",
        )
    if sum(expected_key_cardinality.values()) != expected_keys:
        raise PilotConstructionError(
            "expected eligible-key cardinality counts must sum to the total"
        )
    partition_counts = _mapping(
        counts["partition_target_counts"], "counts.partition_target_counts"
    )
    _exact_keys(partition_counts, set(_PARTITIONS), "counts.partition_target_counts")
    for name in _PARTITIONS:
        _positive_int(partition_counts[name], f"partition_target_counts.{name}")
    if sum(partition_counts.values()) != target_count:
        raise PilotConstructionError("partition target counts must sum to target_count")
    if expected_keys > target_count:
        raise PilotConstructionError(
            "target_count cannot cover every expected eligible key"
        )

    construction = _mapping(config["construction"], "construction")
    _exact_keys(construction, _CONSTRUCTION_KEYS, "construction")
    expected_construction = {
        "official_split": "train",
        "min_supported_answers_per_key": 2,
        "min_unique_train_videos_per_supported_answer": 5,
        "pair_roles": _PAIR_ROLES,
        "require_every_eligible_key": True,
        "prefer_unique_donors": True,
        "target_and_donor_video_disjoint_when_possible": True,
        "component_partition_lock": True,
    }
    if construction != expected_construction:
        raise PilotConstructionError("construction rules differ from the locked protocol")

    locks = _mapping(config["input_locks"], "input_locks")
    _exact_keys(
        locks,
        {"audit_configuration_sha256", "train_archive", "validation_archive"},
        "input_locks",
    )
    _sha256_text(
        locks["audit_configuration_sha256"],
        "input_locks.audit_configuration_sha256",
    )
    train_lock = _validate_archive_lock(
        locks["train_archive"], "train_archive", "train"
    )
    validation_lock = _validate_archive_lock(
        locks["validation_archive"], "validation_archive", "validation"
    )
    if train_lock["metadata_split"] != "train":
        raise PilotConstructionError("train archive metadata split must be train")
    if validation_lock["metadata_split"] != "valid":
        raise PilotConstructionError("validation archive metadata split must be valid")

    source = _mapping(config["source"], "source")
    _exact_keys(source, _SOURCE_KEYS, "source")
    if source["annotation_license"] != "CC-BY-4.0":
        raise PilotConstructionError("source.annotation_license must be CC-BY-4.0")
    if source["license_url"] != "https://creativecommons.org/licenses/by/4.0/legalcode":
        raise PilotConstructionError("source.license_url is unexpected")
    for field in ("repository_url", "repository_commit", "citation"):
        if not isinstance(source[field], str) or not source[field]:
            raise PilotConstructionError(f"source.{field} must be a nonempty string")
    if (
        source["repository_url"] != _CANONICAL_REPOSITORY_URL
        or source["repository_commit"] != _PINNED_REPOSITORY_COMMIT
    ):
        raise PilotConstructionError("source repository is not the pinned official source")

    gate = _mapping(config["gate_contract"], "gate_contract")
    _exact_keys(gate, _GATE_KEYS, "gate_contract")
    expected_gate = {
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
    if gate != expected_gate:
        raise PilotConstructionError("gate contract differs from the locked protocol")
    return config


def _validate_split_summary(value: Any, field: str) -> Mapping[str, Any]:
    value = _mapping(value, field)
    _exact_keys(value, _SPLIT_SUMMARY_KEYS, field)
    at_least_two = _mapping(
        value["at_least_two_supported_answers"],
        f"{field}.at_least_two_supported_answers",
    )
    _exact_keys(
        at_least_two,
        {"group_key_count", "question_count", "unique_video_count"},
        f"{field}.at_least_two_supported_answers",
    )
    for key in ("group_key_count", "question_count", "unique_video_count"):
        _nonnegative_int(at_least_two[key], f"{field}.{key}")
    for key in (
        "candidate_group_key_count",
        "duplicate_option_exclusion_count",
        "raw_question_count",
        "structurally_valid_question_count",
        "video_count",
        "videos_with_structurally_valid_questions",
    ):
        _nonnegative_int(value[key], f"{field}.{key}")
    if not isinstance(value["duplicate_option_exclusions"], list):
        raise PilotConstructionError(f"{field}.duplicate_option_exclusions must be a list")
    if len(value["duplicate_option_exclusions"]) != value[
        "duplicate_option_exclusion_count"
    ]:
        raise PilotConstructionError(f"{field} duplicate exclusion count differs")
    return value


def _group_key_record(group: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "question": group["question"],
        "option_vocabulary": group["option_vocabulary"],
        "area": group["area"],
        "reasoning": group["reasoning"],
        "tags": group["tags"],
    }


def _validate_tier(value: Any, field: str) -> Mapping[str, Any]:
    tier = _mapping(value, field)
    _exact_keys(tier, _TIER_KEYS, field)
    group_count = _nonnegative_int(tier["group_key_count"], f"{field}.group_key_count")
    groups = tier["candidate_groups"]
    if not isinstance(groups, list) or len(groups) != group_count:
        raise PilotConstructionError(f"{field}.candidate_groups count differs")
    for split_count_field in ("question_count", "unique_video_count"):
        split_counts = _mapping(tier[split_count_field], f"{field}.{split_count_field}")
        _exact_keys(split_counts, {"train", "validation"}, f"{field}.{split_count_field}")
        for split_name in ("train", "validation"):
            _nonnegative_int(
                split_counts[split_name],
                f"{field}.{split_count_field}.{split_name}",
            )
    cardinality = _mapping(
        tier["supported_answer_cardinality"],
        f"{field}.supported_answer_cardinality",
    )
    cardinality_total = 0
    for key, count in cardinality.items():
        if key not in {"2", "3"}:
            raise PilotConstructionError(f"{field} has an invalid answer cardinality")
        cardinality_total += _nonnegative_int(count, f"{field}.cardinality.{key}")
    if cardinality_total != group_count:
        raise PilotConstructionError(f"{field} cardinality total differs")
    return tier


def _validate_group(value: Any, field: str) -> Mapping[str, Any]:
    group = _mapping(value, field)
    _exact_keys(group, _GROUP_KEYS, field)
    for key in ("question", "area", "reasoning"):
        if not isinstance(group[key], str) or normalize_text(group[key]) != group[key]:
            raise PilotConstructionError(f"{field}.{key} must be normalized text")
    for key in ("option_vocabulary", "tags", "supported_answer_vocabulary"):
        entries = group[key]
        if (
            not isinstance(entries, list)
            or not entries
            or any(not isinstance(item, str) or normalize_text(item) != item for item in entries)
        ):
            raise PilotConstructionError(f"{field}.{key} must contain normalized text")
        if entries != sorted(set(entries)):
            raise PilotConstructionError(f"{field}.{key} must be sorted and unique")
    if len(group["option_vocabulary"]) != 3:
        raise PilotConstructionError(f"{field} must have three options")
    if len(group["supported_answer_vocabulary"]) < 2:
        raise PilotConstructionError(f"{field} must support at least two answers")
    if not set(group["supported_answer_vocabulary"]) <= set(
        group["option_vocabulary"]
    ):
        raise PilotConstructionError(f"{field} has an answer outside its options")
    _sha256_text(group["key_sha256"], f"{field}.key_sha256")
    if _canonical_digest(_group_key_record(group)) != group["key_sha256"]:
        raise PilotConstructionError(f"{field}.key_sha256 does not match its key")
    for count_field in ("answer_video_counts", "question_counts", "unique_video_counts"):
        counts = _mapping(group[count_field], f"{field}.{count_field}")
        _exact_keys(counts, {"train", "validation"}, f"{field}.{count_field}")
        if count_field == "answer_video_counts":
            for split_name in ("train", "validation"):
                answer_counts = _mapping(
                    counts[split_name], f"{field}.{count_field}.{split_name}"
                )
                _exact_keys(
                    answer_counts,
                    set(group["option_vocabulary"]),
                    f"{field}.{count_field}.{split_name}",
                )
                for answer, count in answer_counts.items():
                    _nonnegative_int(
                        count,
                        f"{field}.{count_field}.{split_name}.{answer}",
                    )
        else:
            for split_name in ("train", "validation"):
                _nonnegative_int(
                    counts[split_name],
                    f"{field}.{count_field}.{split_name}",
                )
    return group


def _validate_audit(value: Any, config: Mapping[str, Any]) -> Mapping[str, Any]:
    audit = _mapping(value, "structural audit")
    _exact_keys(audit, _AUDIT_KEYS, "structural audit")
    if audit["schema_version"] != 2 or audit["status"] != "pass":
        raise PilotConstructionError("structural audit did not pass schema version 2")
    if audit["scope"] != "annotation_candidate_pool_feasibility_only":
        raise PilotConstructionError("structural audit scope is unexpected")
    if not isinstance(audit["limitations"], list) or not audit["limitations"]:
        raise PilotConstructionError("structural audit limitations must be present")
    if audit["configuration_sha256"] != config["input_locks"][
        "audit_configuration_sha256"
    ]:
        raise PilotConstructionError("structural audit configuration SHA-256 differs")
    if audit["normalization"] != _NORMALIZATION:
        raise PilotConstructionError("structural audit normalization differs")

    disclosure = _mapping(
        audit["annotation_use_disclosure"], "annotation_use_disclosure"
    )
    _exact_keys(
        disclosure,
        {
            "train_labels_inspected",
            "validation_labels_inspected",
            "validation_use",
            "validation_media_or_model_outputs_inspected",
            "test_annotations_inspected",
        },
        "annotation_use_disclosure",
    )
    if disclosure != {
        "train_labels_inspected": True,
        "validation_labels_inspected": True,
        "validation_use": "schema_and_candidate_pool_counts_only",
        "validation_media_or_model_outputs_inspected": False,
        "test_annotations_inspected": False,
    }:
        raise PilotConstructionError("structural audit data-use disclosure is unsafe")

    source_repository = _mapping(audit["source_repository"], "source_repository")
    _exact_keys(
        source_repository,
        {"url", "commit", "annotation_license", "license_url"},
        "source_repository",
    )
    source = config["source"]
    if source_repository != {
        "url": source["repository_url"],
        "commit": source["repository_commit"],
        "annotation_license": source["annotation_license"],
        "license_url": source["license_url"],
    }:
        raise PilotConstructionError("structural audit source repository differs")

    source_attribution = _mapping(audit["source_attribution"], "source_attribution")
    _exact_keys(
        source_attribution,
        {"dataset", "authors", "copyright", "materials_license", "license_url"},
        "source_attribution",
    )
    if (
        source_attribution["materials_license"] != source["annotation_license"]
        or source_attribution["license_url"] != source["license_url"]
    ):
        raise PilotConstructionError("structural audit attribution differs")

    archives = _mapping(audit["source_archives"], "source_archives")
    _exact_keys(archives, {"train", "validation"}, "source_archives")
    for split_name in ("train", "validation"):
        archive = _validate_archive_lock(
            archives[split_name], f"source_archives.{split_name}", split_name
        )
        if dict(archive) != dict(config["input_locks"][f"{split_name}_archive"]):
            raise PilotConstructionError(
                f"structural audit {split_name} archive lock differs"
            )

    eligibility = _mapping(audit["eligibility"], "eligibility")
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
    if (
        eligibility["min_supported_answers_per_group"] != 2
        or eligibility["require_same_supported_answer_vocabulary_across_splits"]
        is not True
        or eligibility["report_all_three_option_tier"] is not True
    ):
        raise PilotConstructionError("structural audit eligibility differs")
    _positive_int(
        eligibility["min_unique_videos_per_supported_answer_per_split"],
        "eligibility minimum videos",
    )

    snapshot = _mapping(audit["snapshot_counts"], "snapshot_counts")
    _exact_keys(
        snapshot,
        {
            "split_minimum_two_answer_group_count",
            "cross_split_minimum_two_answer_group_count",
            "cross_split_all_three_option_group_count",
            "duplicate_option_exclusion_count",
        },
        "snapshot_counts",
    )
    _nonnegative_int(
        snapshot["cross_split_minimum_two_answer_group_count"],
        "snapshot_counts.cross_split_minimum_two_answer_group_count",
    )
    _nonnegative_int(
        snapshot["cross_split_all_three_option_group_count"],
        "snapshot_counts.cross_split_all_three_option_group_count",
    )
    split_minimum_counts = _mapping(
        snapshot["split_minimum_two_answer_group_count"],
        "snapshot_counts.split_minimum_two_answer_group_count",
    )
    _exact_keys(
        split_minimum_counts,
        {"train", "validation"},
        "snapshot_counts.split_minimum_two_answer_group_count",
    )
    for split_name, count in split_minimum_counts.items():
        _nonnegative_int(
            count,
            f"snapshot_counts.split_minimum_two_answer_group_count.{split_name}",
        )

    splits = _mapping(audit["splits"], "splits")
    _exact_keys(splits, {"train", "validation"}, "splits")
    train_summary = _validate_split_summary(splits["train"], "splits.train")
    _validate_split_summary(splits["validation"], "splits.validation")
    duplicate_counts = _mapping(
        snapshot["duplicate_option_exclusion_count"],
        "snapshot_counts.duplicate_option_exclusion_count",
    )
    _exact_keys(
        duplicate_counts,
        {"train", "validation"},
        "snapshot_counts.duplicate_option_exclusion_count",
    )
    if duplicate_counts["train"] != train_summary[
        "duplicate_option_exclusion_count"
    ]:
        raise PilotConstructionError("training duplicate exclusion count differs")

    cross = _mapping(audit["cross_split"], "cross_split")
    _exact_keys(
        cross,
        {
            "shared_group_key_count",
            "minimum_two_supported_answers",
            "all_three_options_supported",
        },
        "cross_split",
    )
    _nonnegative_int(cross["shared_group_key_count"], "cross_split.shared_group_key_count")
    minimum_two = _validate_tier(
        cross["minimum_two_supported_answers"],
        "cross_split.minimum_two_supported_answers",
    )
    _validate_tier(
        cross["all_three_options_supported"],
        "cross_split.all_three_options_supported",
    )
    seen_hashes: set[str] = set()
    for index, group in enumerate(minimum_two["candidate_groups"]):
        validated = _validate_group(group, f"candidate_groups[{index}]")
        key_hash = validated["key_sha256"]
        if key_hash in seen_hashes:
            raise PilotConstructionError("duplicate eligible group key SHA-256")
        seen_hashes.add(key_hash)
        supported_count = len(validated["supported_answer_vocabulary"])
        if supported_count not in (2, 3):
            raise PilotConstructionError("eligible group answer count is invalid")
        minimum = eligibility["min_unique_videos_per_supported_answer_per_split"]
        for split_name in ("train", "validation"):
            counts = validated["answer_video_counts"][split_name]
            for answer in validated["supported_answer_vocabulary"]:
                if counts[answer] < minimum:
                    raise PilotConstructionError(
                        "eligible group is below the minimum video count"
                    )
    return audit


def _authenticate_file(
    path: pathlib.Path, lock: Mapping[str, Any], split_name: str
) -> str:
    if not path.is_file():
        raise PilotConstructionError(f"{split_name} archive is not a regular file")
    if path.stat().st_size != lock["size_bytes"]:
        raise PilotConstructionError(f"{split_name} archive size does not match")
    observed = _sha256_path(path)
    if observed != lock["sha256"]:
        raise PilotConstructionError(f"{split_name} archive SHA-256 does not match")
    return observed


def _read_train_payload(
    path: pathlib.Path, lock: Mapping[str, Any]
) -> Mapping[str, Any]:
    _authenticate_file(path, lock, "train")
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].filename != lock["member"]:
                raise PilotConstructionError(
                    f"train archive must contain only {lock['member']}"
                )
            member = members[0]
            if member.is_dir() or member.flag_bits & 0x1:
                raise PilotConstructionError(
                    "train archive member must be an unencrypted file"
                )
            if member.file_size > 64 * 1024 * 1024:
                raise PilotConstructionError("train annotation member is unexpectedly large")
            payload = _load_json_bytes(archive.read(member), "train annotation member")
    except (OSError, zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        raise PilotConstructionError(f"train archive cannot be read: {exc}") from exc
    return _mapping(payload, "train annotations")


def _positive_number(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PilotConstructionError(f"{field} must be numeric")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise PilotConstructionError(f"{field} must be finite and positive")


def _validate_metadata(value: Any, video_id: str) -> None:
    metadata = _mapping(value, f"metadata for {video_id}")
    _exact_keys(metadata, _METADATA_KEYS, f"metadata for {video_id}")
    if metadata["video_id"] != video_id or metadata["split"] != "train":
        raise PilotConstructionError(f"metadata identity differs for {video_id}")
    _positive_number(metadata["frame_rate"], f"frame_rate for {video_id}")
    _positive_int(metadata["num_frames"], f"num_frames for {video_id}")
    resolution = metadata["resolution"]
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise PilotConstructionError(f"resolution for {video_id} is invalid")
    for item in resolution:
        _positive_int(item, f"resolution for {video_id}")
    _nonnegative_int(metadata["audio_samples"], f"audio_samples for {video_id}")
    _positive_number(metadata["audio_sample_rate"], f"audio_sample_rate for {video_id}")
    for field in ("is_cup_game", "is_camera_moving"):
        if isinstance(metadata[field], bool) or metadata[field] not in (0, 1):
            raise PilotConstructionError(f"{field} for {video_id} must be 0 or 1")


def _entry_from_question(
    video_id: str, expected_question_id: int, value: Any
) -> tuple[str, dict[str, Any]]:
    question = _mapping(value, f"question {expected_question_id} for {video_id}")
    _exact_keys(
        question,
        _QUESTION_KEYS,
        f"question {expected_question_id} for {video_id}",
    )
    if (
        isinstance(question["id"], bool)
        or not isinstance(question["id"], int)
        or question["id"] != expected_question_id
    ):
        raise PilotConstructionError(f"question IDs for {video_id} are not consecutive")
    options = question["options"]
    if not isinstance(options, list) or len(options) != 3:
        raise PilotConstructionError(f"question options for {video_id} must have length 3")
    try:
        key, answer = candidate_group_key(question)
    except AuditError as exc:
        raise PilotConstructionError(str(exc)) from exc
    key_record = {
        "question": key[0],
        "option_vocabulary": list(key[1]),
        "area": key[2],
        "reasoning": key[3],
        "tags": list(key[4]),
    }
    key_sha256 = _canonical_digest(key_record)
    normalized_options = [normalize_text(item) for item in options]
    answer_id = question["answer_id"]
    return key_sha256, {
        "video_id": video_id,
        "question_id": expected_question_id,
        "question": normalize_text(question["question"]),
        "options": normalized_options,
        "answer_id": answer_id,
        "answer": answer,
        "area": normalize_text(question["area"]),
        "reasoning": normalize_text(question["reasoning"]),
        "tags": sorted(normalize_text(item) for item in question["tag"]),
    }


def _collect_eligible_entries(
    payload: Mapping[str, Any],
    *,
    min_supported_answers: int,
    min_unique_videos_per_supported_answer: int,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], list[dict[str, Any]]]:
    entries: dict[str, dict[str, dict[str, dict[str, Any]]]] = collections.defaultdict(
        lambda: collections.defaultdict(dict)
    )
    exclusions: list[dict[str, Any]] = []

    if not payload:
        raise PilotConstructionError("train annotations must be nonempty")
    for video_id in sorted(payload):
        if (
            not isinstance(video_id, str)
            or not video_id
            or normalize_text(video_id) != video_id
        ):
            raise PilotConstructionError("video IDs must be nonempty normalized strings")
        video = _mapping(payload[video_id], f"video record {video_id}")
        _exact_keys(video, _TOP_VIDEO_KEYS, f"video record {video_id}")
        _validate_metadata(video["metadata"], video_id)
        questions = video["mc_question"]
        if not isinstance(questions, list) or not questions:
            raise PilotConstructionError(f"mc_question for {video_id} must be nonempty")
        for question_id, question in enumerate(questions):
            try:
                key_hash, entry = _entry_from_question(video_id, question_id, question)
            except PilotConstructionError as exc:
                if str(exc) == "duplicate option text after normalization":
                    exclusions.append(
                        {
                            "video_id": video_id,
                            "question_id": question_id,
                            "reason": "duplicate_option_text_after_normalization",
                        }
                    )
                    continue
                raise
            answer = entry["answer"]
            prior_answers = [
                prior_answer
                for prior_answer, videos in entries[key_hash].items()
                if video_id in videos
            ]
            if prior_answers and answer not in prior_answers:
                raise PilotConstructionError(
                    f"video {video_id} has conflicting answers for one eligible key"
                )
            prior = entries[key_hash][answer].get(video_id)
            if prior is None:
                entries[key_hash][answer][video_id] = entry
            else:
                exclusions.append(
                    {
                        "video_id": video_id,
                        "question_id": question_id,
                        "reason": "duplicate_video_key_answer_entry",
                        "retained_question_id": prior["question_id"],
                    }
                )

    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for key_hash, answer_entries in entries.items():
        supported_answers = {
            answer: videos
            for answer, videos in answer_entries.items()
            if len(videos) >= min_unique_videos_per_supported_answer
        }
        if len(supported_answers) < min_supported_answers:
            continue
        result[key_hash] = {}
        for answer in sorted(supported_answers):
            selected_entries = sorted(
                answer_entries.get(answer, {}).values(),
                key=lambda entry: (entry["video_id"], entry["question_id"]),
            )
            result[key_hash][answer] = selected_entries
    return result, exclusions


def _target_quotas(
    key_hashes: Sequence[str], target_count: int, seed: int
) -> dict[str, int]:
    key_count = len(key_hashes)
    base, remainder = divmod(target_count, key_count)
    if base < 1:
        raise PilotConstructionError("target count does not cover every eligible key")
    ranked = sorted(key_hashes, key=lambda key: (_rank(seed, "quota", key), key))
    return {
        key: base + (1 if index < remainder else 0)
        for index, key in enumerate(ranked)
    }


def _select_targets(
    entries: Mapping[str, Mapping[str, Sequence[dict[str, Any]]]],
    quotas: Mapping[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    used_videos: set[str] = set()
    selected: list[dict[str, Any]] = []
    key_order = sorted(
        entries,
        key=lambda key: (
            sum(len(items) for items in entries[key].values()) / quotas[key],
            _rank(seed, "target-key-order", key),
            key,
        ),
    )
    for key_hash in key_order:
        answers = sorted(
            entries[key_hash],
            key=lambda answer: (_rank(seed, "target-answer-order", key_hash, answer), answer),
        )
        chosen_by_answer = collections.Counter()
        for slot in range(quotas[key_hash]):
            answer_order = sorted(
                answers,
                key=lambda answer: (
                    chosen_by_answer[answer],
                    _rank(seed, "target-answer-slot", key_hash, slot, answer),
                    answer,
                ),
            )
            chosen: dict[str, Any] | None = None
            chosen_answer = ""
            for answer in answer_order:
                candidates = [
                    entry
                    for entry in entries[key_hash][answer]
                    if entry["video_id"] not in used_videos
                ]
                if not candidates:
                    continue
                candidates.sort(
                    key=lambda entry: (
                        _rank(
                            seed,
                            "target-entry",
                            key_hash,
                            slot,
                            answer,
                            entry["video_id"],
                            entry["question_id"],
                        ),
                        entry["video_id"],
                        entry["question_id"],
                    )
                )
                chosen = dict(candidates[0])
                chosen_answer = answer
                break
            if chosen is None:
                raise PilotConstructionError(
                    f"eligible key {key_hash} cannot supply its exact target quota"
                )
            chosen["key_sha256"] = key_hash
            used_videos.add(chosen["video_id"])
            chosen_by_answer[chosen_answer] += 1
            selected.append(chosen)
    return selected


def _partition_targets(
    targets: Sequence[dict[str, Any]],
    desired_counts: Mapping[str, int],
    seed: int,
) -> dict[tuple[str, int, str], str]:
    ordered = sorted(
        targets,
        key=lambda target: (
            _rank(
                seed,
                "partition-target",
                target["key_sha256"],
                target["video_id"],
                target["question_id"],
            ),
            target["video_id"],
            target["question_id"],
        ),
    )
    assigned = {name: 0 for name in _PARTITIONS}
    total = len(ordered)
    result: dict[tuple[str, int, str], str] = {}
    for index, target in enumerate(ordered):
        available = [
            name for name in _PARTITIONS if assigned[name] < desired_counts[name]
        ]
        partition = max(
            available,
            key=lambda name: (
                desired_counts[name] * (index + 1) / total - assigned[name],
                -_PARTITIONS.index(name),
            ),
        )
        identity = (
            target["video_id"],
            target["question_id"],
            target["key_sha256"],
        )
        result[identity] = partition
        assigned[partition] += 1
    if assigned != dict(desired_counts):
        raise PilotConstructionError("exact partition target counts were not met")
    return result


def _public_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "video_id": entry["video_id"],
        "question_id": entry["question_id"],
        "question": entry["question"],
        "options": list(entry["options"]),
        "answer_id": entry["answer_id"],
        "answer": entry["answer"],
    }


def _maximum_unique_assignment(
    requests: Sequence[tuple[str, Sequence[str]]],
) -> dict[str, str]:
    """Return a maximum request-to-donor assignment with unique donor IDs."""

    candidate_map = {request_id: list(candidates) for request_id, candidates in requests}
    if len(candidate_map) != len(requests):
        raise PilotConstructionError("donor request IDs must be unique")
    donor_to_request: dict[str, str] = {}
    request_to_donor: dict[str, str] = {}

    def augment(request_id: str, visited: set[str]) -> bool:
        for donor_id in candidate_map[request_id]:
            if donor_id in visited:
                continue
            visited.add(donor_id)
            prior_request = donor_to_request.get(donor_id)
            if prior_request is None or augment(prior_request, visited):
                donor_to_request[donor_id] = request_id
                request_to_donor[request_id] = donor_id
                return True
        return False

    for request_id, _ in requests:
        augment(request_id, set())
    return dict(sorted(request_to_donor.items()))


def _choose_donor(
    *,
    candidates: Sequence[dict[str, Any]],
    target: Mapping[str, Any],
    partition: str,
    role: str,
    seed: int,
    target_ids: set[str],
    used_donor_ids: set[str],
    video_partitions: dict[str, str],
    excluded_ids: set[str],
) -> tuple[dict[str, Any], str | None]:
    compatible = [
        entry
        for entry in candidates
        if entry["video_id"] != target["video_id"]
        and entry["video_id"] not in excluded_ids
        and video_partitions.get(entry["video_id"], partition) == partition
    ]
    if not compatible:
        raise PilotConstructionError(
            f"no partition-safe {role} donor exists for target {target['video_id']}"
        )

    def preference(entry: Mapping[str, Any]) -> tuple[int, int, str, str, int]:
        video_id = entry["video_id"]
        return (
            1 if video_id in used_donor_ids else 0,
            1 if video_id in target_ids else 0,
            _rank(
                seed,
                "donor",
                role,
                partition,
                target["key_sha256"],
                target["video_id"],
                video_id,
                entry["question_id"],
            ),
            video_id,
            entry["question_id"],
        )

    donor = dict(min(compatible, key=preference))
    donor_id = donor["video_id"]
    reuse_reason = None
    if donor_id in used_donor_ids:
        reuse_reason = "donor_reused_within_partition_after_unique_pool_exhausted"
    elif donor_id in target_ids:
        reuse_reason = "target_video_used_as_donor_within_partition_after_pool_exhausted"
    video_partitions.setdefault(donor_id, partition)
    used_donor_ids.add(donor_id)
    return donor, reuse_reason


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _construct_pairs(
    *,
    targets: Sequence[dict[str, Any]],
    entries: Mapping[str, Mapping[str, Sequence[dict[str, Any]]]],
    target_partitions: Mapping[tuple[str, int, str], str],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    target_ids = {target["video_id"] for target in targets}
    video_partitions: dict[str, str] = {}
    for target in targets:
        identity = (
            target["video_id"],
            target["question_id"],
            target["key_sha256"],
        )
        partition = target_partitions[identity]
        prior = video_partitions.setdefault(target["video_id"], partition)
        if prior != partition:
            raise PilotConstructionError("one target video was assigned across partitions")

    used_donor_ids: set[str] = set()
    donor_reuse_events: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    graph = _DisjointSet()
    ordered_targets = sorted(
        targets,
        key=lambda target: (
            _rank(
                seed,
                "donor-target-order",
                target["key_sha256"],
                target["video_id"],
                target["question_id"],
            ),
            target["video_id"],
        ),
    )

    request_specs: dict[str, dict[str, Any]] = {}
    for target in ordered_targets:
        identity = (
            target["video_id"],
            target["question_id"],
            target["key_sha256"],
        )
        partition = target_partitions[identity]
        answer_entries = entries[target["key_sha256"]]
        role_candidates = {
            "same_answer_nuisance": list(answer_entries[target["answer"]]),
            "opposite_answer_candidate": [
                entry
                for candidate_answer in sorted(answer_entries)
                if candidate_answer != target["answer"]
                for entry in answer_entries[candidate_answer]
            ],
        }
        for role in _PAIR_ROLES:
            request_id = _canonical_digest(
                {
                    "key_sha256": target["key_sha256"],
                    "partition": partition,
                    "role": role,
                    "target_video_id": target["video_id"],
                    "target_question_id": target["question_id"],
                }
            )
            candidate_by_video: dict[str, dict[str, Any]] = {}
            for candidate in sorted(
                role_candidates[role],
                key=lambda entry: (
                    _rank(
                        seed,
                        "unique-donor",
                        request_id,
                        entry["video_id"],
                        entry["question_id"],
                    ),
                    entry["video_id"],
                    entry["question_id"],
                ),
            ):
                if candidate["video_id"] in target_ids:
                    continue
                candidate_by_video.setdefault(candidate["video_id"], candidate)
            request_specs[request_id] = {
                "target": target,
                "partition": partition,
                "role": role,
                "candidate_by_video": candidate_by_video,
            }
    matching_requests = sorted(
        (
            (request_id, list(spec["candidate_by_video"]))
            for request_id, spec in request_specs.items()
        ),
        key=lambda item: (len(item[1]), item[0]),
    )
    unique_assignment = _maximum_unique_assignment(matching_requests)
    has_global_unique_assignment = len(unique_assignment) == len(request_specs)

    for target in ordered_targets:
        identity = (
            target["video_id"],
            target["question_id"],
            target["key_sha256"],
        )
        partition = target_partitions[identity]
        answer = target["answer"]
        answer_entries = entries[target["key_sha256"]]
        if has_global_unique_assignment:
            selected: dict[str, tuple[dict[str, Any], str | None]] = {}
            for role in _PAIR_ROLES:
                request_id = _canonical_digest(
                    {
                        "key_sha256": target["key_sha256"],
                        "partition": partition,
                        "role": role,
                        "target_video_id": target["video_id"],
                        "target_question_id": target["question_id"],
                    }
                )
                donor_id = unique_assignment[request_id]
                donor = dict(request_specs[request_id]["candidate_by_video"][donor_id])
                prior_partition = video_partitions.setdefault(donor_id, partition)
                if prior_partition != partition:
                    raise PilotConstructionError(
                        "a uniquely assigned donor crosses partitions"
                    )
                used_donor_ids.add(donor_id)
                selected[role] = (donor, None)
            same_donor, same_reuse = selected["same_answer_nuisance"]
            opposite_donor, opposite_reuse = selected[
                "opposite_answer_candidate"
            ]
        else:
            same_donor, same_reuse = _choose_donor(
                candidates=answer_entries[answer],
                target=target,
                partition=partition,
                role="same_answer_nuisance",
                seed=seed,
                target_ids=target_ids,
                used_donor_ids=used_donor_ids,
                video_partitions=video_partitions,
                excluded_ids=set(),
            )
            opposite_answers = sorted(
                (candidate for candidate in answer_entries if candidate != answer),
                key=lambda candidate: (
                    -sum(
                        1
                        for entry in answer_entries[candidate]
                        if entry["video_id"] not in used_donor_ids
                        and entry["video_id"] not in target_ids
                        and video_partitions.get(entry["video_id"], partition)
                        == partition
                    ),
                    _rank(
                        seed,
                        "opposite-answer",
                        target["key_sha256"],
                        target["video_id"],
                        candidate,
                    ),
                    candidate,
                ),
            )
            opposite_candidates = [
                entry
                for opposite_answer in opposite_answers
                for entry in answer_entries[opposite_answer]
            ]
            opposite_donor, opposite_reuse = _choose_donor(
                candidates=opposite_candidates,
                target=target,
                partition=partition,
                role="opposite_answer_candidate",
                seed=seed,
                target_ids=target_ids,
                used_donor_ids=used_donor_ids,
                video_partitions=video_partitions,
                excluded_ids={same_donor["video_id"]},
            )

        for role, donor, reuse_reason in (
            ("same_answer_nuisance", same_donor, same_reuse),
            ("opposite_answer_candidate", opposite_donor, opposite_reuse),
        ):
            pair_identity = {
                "key_sha256": target["key_sha256"],
                "partition": partition,
                "role": role,
                "target_video_id": target["video_id"],
                "target_question_id": target["question_id"],
                "donor_video_id": donor["video_id"],
                "donor_question_id": donor["question_id"],
            }
            pair_id = _canonical_digest(pair_identity)[:24]
            pair = {
                "pair_id": pair_id,
                "component_id": "pending",
                "official_split": "train",
                "partition": partition,
                "key_sha256": target["key_sha256"],
                "role": role,
                "anchor": {
                    "question": target["question"],
                    "options": list(target["options"]),
                    "answer_id": target["answer_id"],
                    "answer": target["answer"],
                },
                "target": _public_entry(target),
                "donor": _public_entry(donor),
            }
            pairs.append(pair)
            graph.union(target["video_id"], donor["video_id"])
            if reuse_reason is not None:
                donor_reuse_events.append(
                    {
                        "pair_id": pair_id,
                        "partition": partition,
                        "video_id": donor["video_id"],
                        "reason": reuse_reason,
                    }
                )

    component_members: dict[str, set[str]] = collections.defaultdict(set)
    for video_id in graph.parent:
        component_members[graph.find(video_id)].add(video_id)
    component_ids: dict[str, str] = {}
    components: list[dict[str, Any]] = []
    for root, members in component_members.items():
        partitions = {video_partitions[video_id] for video_id in members}
        if len(partitions) != 1:
            raise PilotConstructionError("a donor-connected component crosses partitions")
        partition = next(iter(partitions))
        component_id = hashlib.sha256(
            json.dumps(sorted(members), separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24]
        component_ids[root] = component_id
        target_count = sum(1 for target in targets if target["video_id"] in members)
        components.append(
            {
                "component_id": component_id,
                "partition": partition,
                "target_count": target_count,
                "video_ids": sorted(members),
            }
        )
    for pair in pairs:
        root = graph.find(pair["target"]["video_id"])
        pair["component_id"] = component_ids[root]
    pairs.sort(key=lambda pair: (pair["partition"], pair["component_id"], pair["pair_id"]))
    components.sort(key=lambda item: (item["partition"], item["component_id"]))
    donor_reuse_events.sort(key=lambda item: item["pair_id"])
    return pairs, components, donor_reuse_events


def _expanded_gate_contract(config_gate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(config_gate),
        "required_later_records": {
            "source_answer_scoring": [
                "component_id",
                "pair_id",
                "partition",
                "source_role",
                "modality",
                "predicted_answer",
                "answer_scores",
                "answer_margin",
                "scorer_id",
                "scorer_config_sha256",
            ],
            "nuisance_detection": [
                "component_id",
                "pair_id",
                "partition",
                "predicted_role",
                "role_scores",
                "detector_id",
                "detector_config_sha256",
            ],
        },
        "source_answer_rules": [
            "Fit source-answer scorers only on scorer_fit components.",
            "Freeze scorer identities and all pass thresholds using scorer_fit and threshold_calibration before opening pilot_gate outcomes.",
            "For every retained pair, score target and donor audio-only and video-only inputs against the exact target option vocabulary.",
            "Retain an opposite-answer pair only when the target source supports the target answer and the donor source supports the donor answer under the frozen margin rule.",
            "Retain a same-answer pair only when target and donor sources support their shared answer under the same frozen rule.",
        ],
        "nuisance_detection_rules": [
            "Fit the edit-role detector only on scorer_fit components and tune its gate only on threshold_calibration components.",
            "Use non-semantic media and encoding diagnostics; do not provide question answers or source-answer scorer outputs to the detector.",
            "Apply the frozen detector once to pilot_gate components and reject a construction when edit role is recoverable above the predeclared ceiling.",
        ],
        "decision_rule": (
            "This builder emits no scientific pass/fail result. A separately hashed "
            "threshold record, frozen before pilot_gate evaluation, is required."
        ),
    }


def _strata_summary(
    entries: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summaries = []
    for key_hash in sorted(entries):
        group_pairs = [pair for pair in pairs if pair["key_sha256"] == key_hash]
        representative = min(
            (
                entry
                for answer_entries in entries[key_hash].values()
                for entry in answer_entries
            ),
            key=lambda entry: (entry["video_id"], entry["question_id"]),
        )
        targets = {
            (pair["target"]["video_id"], pair["target"]["question_id"]): pair[
                "target"
            ]
            for pair in group_pairs
        }
        summaries.append(
            {
                "key_sha256": key_hash,
                "question": representative["question"],
                "supported_answers": sorted(entries[key_hash]),
                "target_count": len(targets),
                "target_answer_counts": dict(
                    sorted(collections.Counter(item["answer"] for item in targets.values()).items())
                ),
                "pair_count": len(group_pairs),
                "partition_target_counts": dict(
                    sorted(
                        collections.Counter(
                            pair["partition"]
                            for pair in group_pairs
                            if pair["role"] == "same_answer_nuisance"
                        ).items()
                    )
                ),
            }
        )
    return summaries


def build_media_pilot_index(
    *,
    config_path: pathlib.Path,
    structural_audit_path: pathlib.Path,
    expected_structural_audit_sha256: str,
    train_archive: pathlib.Path,
    validation_archive: pathlib.Path,
) -> dict[str, Any]:
    """Build the locked candidate index without reading media or outcomes."""

    config_path = pathlib.Path(config_path)
    structural_audit_path = pathlib.Path(structural_audit_path)
    train_archive = pathlib.Path(train_archive)
    validation_archive = pathlib.Path(validation_archive)

    config_bytes, raw_config = _read_json(config_path, "configuration")
    config = _validate_config(raw_config)
    expected_audit_sha256 = _sha256_text(
        expected_structural_audit_sha256,
        "expected structural audit SHA-256",
    )
    audit_bytes, raw_audit = _read_json(structural_audit_path, "structural audit")
    observed_audit_sha256 = hashlib.sha256(audit_bytes).hexdigest()
    if observed_audit_sha256 != expected_audit_sha256:
        raise PilotConstructionError("structural audit SHA-256 does not match")
    audit = _validate_audit(raw_audit, config)

    locks = config["input_locks"]
    _authenticate_file(validation_archive, locks["validation_archive"], "validation")
    train_payload = _read_train_payload(train_archive, locks["train_archive"])
    construction = config["construction"]
    eligible_entries, exclusions = _collect_eligible_entries(
        train_payload,
        min_supported_answers=construction["min_supported_answers_per_key"],
        min_unique_videos_per_supported_answer=construction[
            "min_unique_train_videos_per_supported_answer"
        ],
    )

    key_hashes = sorted(eligible_entries)
    if len(key_hashes) != config["counts"]["expected_eligible_key_count"]:
        raise PilotConstructionError("eligible training key count differs")
    eligible_key_cardinality = dict(
        sorted(
            collections.Counter(
                str(len(eligible_entries[key_hash])) for key_hash in key_hashes
            ).items()
        )
    )
    for cardinality in ("2", "3"):
        eligible_key_cardinality.setdefault(cardinality, 0)
    if eligible_key_cardinality != config["counts"][
        "expected_eligible_key_count_by_supported_answer_count"
    ]:
        raise PilotConstructionError(
            "eligible training key supported-answer counts differ"
        )
    quotas = _target_quotas(
        key_hashes,
        config["counts"]["target_count"],
        config["seed"],
    )
    targets = _select_targets(eligible_entries, quotas, config["seed"])
    target_partitions = _partition_targets(
        targets,
        config["counts"]["partition_target_counts"],
        config["seed"],
    )
    pairs, components, donor_reuse_events = _construct_pairs(
        targets=targets,
        entries=eligible_entries,
        target_partitions=target_partitions,
        seed=config["seed"],
    )
    if len(pairs) != config["counts"]["pair_count"]:
        raise PilotConstructionError("constructed pair count differs from the exact count")

    role_counts = collections.Counter(pair["role"] for pair in pairs)
    unique_targets = {pair["target"]["video_id"] for pair in pairs}
    unique_donors = {pair["donor"]["video_id"] for pair in pairs}
    partition_counts = {}
    for partition in _PARTITIONS:
        partition_pairs = [pair for pair in pairs if pair["partition"] == partition]
        partition_targets = {
            (pair["target"]["video_id"], pair["target"]["question_id"])
            for pair in partition_pairs
        }
        partition_components = {
            pair["component_id"] for pair in partition_pairs
        }
        partition_counts[partition] = {
            "component_count": len(partition_components),
            "pair_count": len(partition_pairs),
            "target_count": len(partition_targets),
        }
        if len(partition_targets) != config["counts"]["partition_target_counts"][
            partition
        ]:
            raise PilotConstructionError(f"{partition} target count differs")

    source_attribution = audit["source_attribution"]
    implementation = implementation_provenance()
    result = {
        "schema_version": 1,
        "status": "construction_complete_scores_not_run",
        "scope": "train_candidate_index_from_authenticated_train_labels",
        "seed": config["seed"],
        "implementation_sha256": implementation["implementation_sha256"],
        "implementation_source_sha256": implementation["source_sha256"],
        "input_digests": {
            "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "structural_audit_sha256": observed_audit_sha256,
            "structural_audit_configuration_sha256": audit[
                "configuration_sha256"
            ],
            "train_archive_sha256": locks["train_archive"]["sha256"],
            "validation_archive_sha256": locks["validation_archive"]["sha256"],
        },
        "source_use": {
            "official_splits_in_candidate_index": ["train"],
            "validation_archive_sha256_verified": True,
            "builder_validation_annotations_loaded": False,
            "builder_test_annotations_loaded": False,
            "media_loaded": False,
            "model_outputs_loaded": False,
        },
        "annotation_use_disclosure": {
            "eligible_keys_derived_from_authenticated_train_archive": True,
            "train_eligibility_min_supported_answers": construction[
                "min_supported_answers_per_key"
            ],
            "train_eligibility_min_unique_videos_per_supported_answer": construction[
                "min_unique_train_videos_per_supported_answer"
            ],
            "structural_audit_authenticated": True,
            "structural_audit_used_for_candidate_selection": False,
            "structural_audit_train_labels_inspected": True,
            "structural_audit_validation_labels_inspected": True,
            "structural_audit_validation_use": (
                "schema_and_candidate_pool_counts_only"
            ),
            "builder_train_labels_loaded": True,
            "builder_validation_labels_loaded": False,
            "builder_test_annotations_loaded": False,
        },
        "source_attribution": dict(source_attribution),
        "change_note": (
            "Derived annotation-only candidate index created by NFC/whitespace "
            "normalization, deterministic sampling, and donor-role assignment. "
            "No media was copied or modified; later counterfactual media must be "
            "identified as changed material under CC-BY-4.0."
        ),
        "construction_rules": dict(config["construction"]),
        "counts": {
            "eligible_key_count": len(key_hashes),
            "eligible_key_count_by_supported_answer_count": (
                eligible_key_cardinality
            ),
            "target_count": len(unique_targets),
            "pair_count": len(pairs),
            "same_answer_pair_count": role_counts["same_answer_nuisance"],
            "opposite_answer_pair_count": role_counts["opposite_answer_candidate"],
            "unique_target_video_count": len(unique_targets),
            "unique_donor_count": len(unique_donors),
            "component_count": len(components),
            "exclusion_count": len(exclusions),
        },
        "partition_counts": partition_counts,
        "strata": _strata_summary(eligible_entries, pairs),
        "exclusions": exclusions,
        "donor_reuse_events": donor_reuse_events,
        "components": components,
        "gate_contract": _expanded_gate_contract(config["gate_contract"]),
        "candidate_index": pairs,
    }
    result["attestation_sha256"] = _canonical_digest(result)
    validate_media_pilot_attestation(result)
    return result


def validate_media_pilot_attestation(value: Any) -> None:
    """Reject output changed after construction or built by different code."""

    record = _mapping(value, "media pilot output")
    attestation_sha256 = _sha256_text(
        record.get("attestation_sha256"),
        "media pilot output.attestation_sha256",
    )
    unsigned = dict(record)
    unsigned.pop("attestation_sha256")
    if _canonical_digest(unsigned) != attestation_sha256:
        raise PilotConstructionError("media pilot output attestation SHA-256 differs")

    observed_implementation = _sha256_text(
        record.get("implementation_sha256"),
        "media pilot output.implementation_sha256",
    )
    current = implementation_provenance()
    if observed_implementation != current["implementation_sha256"]:
        raise PilotConstructionError("media pilot output implementation SHA-256 differs")

    source_sha256 = _mapping(
        record.get("implementation_source_sha256"),
        "media pilot output.implementation_source_sha256",
    )
    _exact_keys(
        source_sha256,
        set(_IMPLEMENTATION_SOURCE_NAMES),
        "media pilot output.implementation_source_sha256",
    )
    for source_name, digest in source_sha256.items():
        _sha256_text(
            digest,
            f"media pilot output.implementation_source_sha256.{source_name}",
        )
    if dict(source_sha256) != current["source_sha256"]:
        raise PilotConstructionError(
            "media pilot output implementation source SHA-256 differs"
        )


def _write_json_atomic(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PilotConstructionError(f"output already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--structural-audit", required=True, type=pathlib.Path)
    parser.add_argument("--structural-audit-sha256", required=True)
    parser.add_argument("--train-archive", required=True, type=pathlib.Path)
    parser.add_argument("--validation-archive", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)

    result = build_media_pilot_index(
        config_path=args.config,
        structural_audit_path=args.structural_audit,
        expected_structural_audit_sha256=args.structural_audit_sha256,
        train_archive=args.train_archive,
        validation_archive=args.validation_archive,
    )
    _write_json_atomic(args.output, result)
    return 0
