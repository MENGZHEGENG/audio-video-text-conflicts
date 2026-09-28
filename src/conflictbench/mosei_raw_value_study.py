"""Per-seed leakage-safe value study over pooled CMU-MOSEI descriptors."""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import mosei
from . import mosei_value_study as core
from .mosei_raw import MODALITIES, RAW_CACHE_SCHEMA, RawMoseiSplit, load_raw_mosei_split

RAW_STUDY_SCHEMA = "conflictbench.mosei-raw-value-study.v1"
RAW_RESULT_SCHEMA = "conflictbench.mosei-raw-value-result.v1"


@dataclass(frozen=True)
class _ProjectionState:
    mean: np.ndarray
    scale: np.ndarray
    direction: np.ndarray


@dataclass(frozen=True)
class PreparedProjectedData:
    """Projected splits and an audit record for one seeded partition."""

    projected_train: core.StudySplit
    projected_valid: core.StudySplit
    partition_indices: dict[str, np.ndarray]
    projection_record: dict[str, Any]
    raw_metadata: dict[str, Any]
    raw_cache_sha256: str


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def configuration_sha256(config: Mapping[str, Any]) -> str:
    """Hash a raw-study configuration with stable JSON serialization."""

    return core.configuration_sha256(config)


def implementation_sha256() -> str:
    """Hash the raw preparation code and reused value-study implementation."""

    paths = (
        pathlib.Path(__file__).resolve(),
        pathlib.Path(__file__).with_name("mosei_raw.py").resolve(),
        pathlib.Path(mosei.__file__).resolve(),
        pathlib.Path(core.__file__).resolve(),
        pathlib.Path(core.__file__).with_name("value_of_information.py").resolve(),
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _core_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema") != RAW_STUDY_SCHEMA:
        raise ValueError(f"study configuration schema must be {RAW_STUDY_SCHEMA}")
    projection = config.get("projection")
    expected_projection = {
        "fit_scope": "official_train_task_fit_groups",
        "standardization": "per_modality_task_fit_zscore",
        "supervised_projection": "task_fit_standardized_mean_difference",
        "minimum_scale": 1e-8,
    }
    if projection != expected_projection:
        raise ValueError("projection configuration does not match the leakage-safe protocol")
    if config.get("evidence_status") != "exploratory_taskfit_reprojection":
        raise ValueError("raw MOSEI evidence status must remain exploratory")
    if config.get("conflict_definition") != "none_unmodified_official_mosei_sentiment":
        raise ValueError("raw MOSEI lane must declare that it has no controlled conflict")
    counts = config.get("expected_split_counts")
    if not isinstance(counts, Mapping) or set(counts) != {"train", "valid"}:
        raise ValueError("expected_split_counts must contain train and valid only")
    for name in ("train", "valid"):
        value = counts[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("expected split counts must be positive integers")
    seeds = config.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("seeds must be a nonempty list of unique integers")
    expected_digest = config.get("expected_cache_sha256")
    if (
        not isinstance(expected_digest, str)
        or len(expected_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_digest)
    ):
        raise ValueError("expected_cache_sha256 must lock the generated raw cache")

    converted = copy.deepcopy(dict(config))
    converted["schema"] = core.STUDY_SCHEMA
    converted["expected_split_counts"] = {
        "train": int(counts["train"]),
        "valid": int(counts["valid"]),
        "test": 0,
    }
    converted.pop("projection", None)
    converted.pop("conflict_definition", None)
    converted.pop("expected_cache_sha256", None)
    converted["input_representation"] = (
        "utterance-aligned descriptors projected from task-fit groups per seed"
    )
    converted["upstream_projection_scope"] = "official_train_task_fit_groups_per_seed"
    converted["evidence_status"] = str(config.get("evidence_status", "confirmatory_candidate"))
    core._validate_config(converted)
    return converted


def _fit_projection(
    descriptors: np.ndarray,
    labels: np.ndarray,
    minimum_scale: float,
) -> _ProjectionState:
    values = np.asarray(descriptors, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or len(values) != len(y) or len(values) == 0:
        raise ValueError("projection data must be a nonempty aligned matrix")
    if not np.isfinite(values).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("projection data must contain finite descriptors and binary labels")
    if len(np.unique(y)) != 2:
        raise ValueError("task-fit projection requires both labels")
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale[scale < minimum_scale] = 1.0
    standardized = (values - mean) / scale
    direction = np.mean(standardized[y == 1], axis=0) - np.mean(standardized[y == 0], axis=0)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm < minimum_scale:
        raise ValueError("task-fit modality projection has zero class separation")
    return _ProjectionState(mean=mean, scale=scale, direction=direction / norm)


def _apply_projection(descriptors: np.ndarray, state: _ProjectionState) -> np.ndarray:
    values = np.asarray(descriptors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(state.mean) or not np.isfinite(values).all():
        raise ValueError("descriptor matrix is incompatible with the fitted projection")
    return ((values - state.mean) / state.scale) @ state.direction


def _as_projected_split(raw: RawMoseiSplit, states: Mapping[str, _ProjectionState]) -> core.StudySplit:
    derived_groups = np.asarray([core.video_group_id(value) for value in raw.sample_ids], dtype="U")
    if not np.array_equal(derived_groups, np.asarray(raw.groups, dtype="U")):
        raise ValueError("stored video groups disagree with utterance identifiers")
    x = np.column_stack(
        [_apply_projection(raw.descriptors[modality], states[modality]) for modality in MODALITIES]
    )
    return core.StudySplit(
        x=x,
        y=np.asarray(raw.y, dtype=np.int64),
        sample_ids=tuple(raw.sample_ids),
        groups=derived_groups,
    )


def _check_counts(metadata: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    observed = metadata.get("loaded_split_counts")
    expected = config["expected_split_counts"]
    if not isinstance(observed, Mapping) or {
        name: int(observed.get(name, -1)) for name in ("train", "valid")
    } != {name: int(expected[name]) for name in ("train", "valid")}:
        raise ValueError("raw cache split counts do not match the locked configuration")


def prepare_projected_data(
    cache_path: str | pathlib.Path,
    config: Mapping[str, Any],
    *,
    seed: int,
) -> PreparedProjectedData:
    """Partition groups first, then fit and freeze per-modality projections."""

    _core_config(config)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in config["seeds"]:
        raise ValueError("seed is not in the locked raw-study seed set")
    source = pathlib.Path(cache_path).expanduser().resolve()
    cache_digest = _sha256(source)
    expected_digest = config.get("expected_cache_sha256")
    if expected_digest is not None and cache_digest != expected_digest:
        raise ValueError("raw cache SHA256 does not match the locked configuration")
    train, metadata = load_raw_mosei_split(source, "train")
    _check_counts(metadata, config)
    task_indices, router_indices, calibration_indices = core._partition_indices(
        train.groups,
        config["train_group_fractions"],
        int(seed),
    )
    partitions = {
        "task_fit": task_indices,
        "router_fit": router_indices,
        "calibration": calibration_indices,
    }
    for name, indices in partitions.items():
        if len(np.unique(train.y[indices])) != 2:
            raise ValueError(f"{name} partition must contain both labels")
    minimum_scale = float(config["projection"]["minimum_scale"])
    if not np.isfinite(minimum_scale) or minimum_scale <= 0:
        raise ValueError("projection minimum_scale must be finite and positive")
    states = {
        modality: _fit_projection(
            train.descriptors[modality][task_indices],
            train.y[task_indices],
            minimum_scale,
        )
        for modality in MODALITIES
    }
    projected_train = _as_projected_split(train, states)

    # Validation is opened only after every supervised projection is frozen.
    valid, validation_metadata = load_raw_mosei_split(source, "valid")
    if validation_metadata != metadata:
        raise ValueError("raw cache metadata changed between split reads")
    if set(train.sample_ids).intersection(valid.sample_ids):
        raise ValueError("training and validation utterances overlap")
    if set(train.groups).intersection(valid.groups):
        raise ValueError("training and validation video groups overlap")
    projected_valid = _as_projected_split(valid, states)

    group_hashes, group_digest = core._validation_group_record(train.groups[task_indices])
    projection_record: dict[str, Any] = {
        "protocol": "partition_groups_then_fit_task_only_projection",
        "fit_partition": "task_fit",
        "fit_utterances": len(task_indices),
        "fit_video_groups": len(np.unique(train.groups[task_indices])),
        "fit_group_hashes": group_hashes,
        "fit_group_digest": group_digest,
        "label_use": "task_fit_only",
        "application_scope": ["task_fit", "router_fit", "calibration", "official_validation"],
        "modalities": {
            modality: {
                "dimension": len(states[modality].mean),
                "mean_sha256": _array_sha256(states[modality].mean),
                "scale_sha256": _array_sha256(states[modality].scale),
                "direction_sha256": _array_sha256(states[modality].direction),
            }
            for modality in MODALITIES
        },
    }
    projection_record["attestation_sha256"] = _json_sha256(projection_record)
    return PreparedProjectedData(
        projected_train=projected_train,
        projected_valid=projected_valid,
        partition_indices=partitions,
        projection_record=projection_record,
        raw_metadata=metadata,
        raw_cache_sha256=cache_digest,
    )


def _write_projected_cache(
    prepared: PreparedProjectedData,
    destination: pathlib.Path,
) -> pathlib.Path:
    metadata = {
        "dataset": "CMU-MOSEI",
        "alignment": "positive_interval_overlap_mean",
        "loaded_split_counts": {
            "train": len(prepared.projected_train.y),
            "valid": len(prepared.projected_valid.y),
            "test": 0,
        },
        "projection": "task_fit_standardized_mean_difference",
    }
    empty_x = np.empty((0, len(MODALITIES)), dtype=np.float32)
    empty_y = np.empty(0, dtype=np.int64)
    empty_ids = np.empty(0, dtype="U1")
    with destination.open("wb") as handle:
        np.savez_compressed(
            handle,
            cache_schema=np.asarray(core.CACHE_SCHEMA),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            train_x=np.asarray(prepared.projected_train.x, dtype=np.float32),
            train_y=np.asarray(prepared.projected_train.y, dtype=np.int64),
            train_sample_ids=np.asarray(prepared.projected_train.sample_ids, dtype="U"),
            valid_x=np.asarray(prepared.projected_valid.x, dtype=np.float32),
            valid_y=np.asarray(prepared.projected_valid.y, dtype=np.int64),
            valid_sample_ids=np.asarray(prepared.projected_valid.sample_ids, dtype="U"),
            test_x=empty_x,
            test_y=empty_y,
            test_sample_ids=empty_ids,
        )
    return destination


def run_raw_value_study(
    cache_path: str | pathlib.Path,
    config: Mapping[str, Any],
    *,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    """Run the current three-family router after per-seed safe projection."""

    converted = _core_config(config)
    prepared = prepare_projected_data(cache_path, config, seed=int(seed))
    with tempfile.TemporaryDirectory(prefix="conflictbench-mosei-raw-") as directory:
        projected_path = _write_projected_cache(
            prepared, pathlib.Path(directory) / "projected-scores.npz"
        )
        core_result = core.run_value_study(
            projected_path,
            converted,
            mode=mode,
            seed=int(seed),
        )

    if (
        core_result["group_partitions"]["task_fit"]["group_digest"]
        != prepared.projection_record["fit_group_digest"]
    ):
        raise RuntimeError("projection and router task partitions diverged")
    core_protocol = {
        "result_schema": core_result["schema"],
        "study_schema": core_result["study_schema"],
        "configuration_sha256": core_result["configuration_sha256"],
        "implementation_sha256": core_result["implementation_sha256"],
        "projected_cache_sha256": core_result["data_access"]["cache_sha256"],
        "router_families": list(core.ROUTER_FAMILIES),
    }
    result = copy.deepcopy(core_result)
    result["schema"] = RAW_RESULT_SCHEMA
    result["study_schema"] = RAW_STUDY_SCHEMA
    result["configuration_sha256"] = configuration_sha256(config)
    result["implementation_sha256"] = implementation_sha256()
    result["core_protocol"] = core_protocol
    result["projection_fit"] = prepared.projection_record
    result["data_access"].update(
        {
            "cache_sha256": prepared.raw_cache_sha256,
            "cache_schema": RAW_CACHE_SCHEMA,
            "loaded_splits": ["train", "valid"],
            "test_opened": False,
            "official_test_policy": "excluded_from_cache_and_value_study",
            "raw_feature_dimensions": dict(prepared.raw_metadata["feature_dimensions"]),
            "sentiment_mode": prepared.raw_metadata["sentiment_mode"],
            "alignment": prepared.raw_metadata["alignment"],
            "pooling": prepared.raw_metadata["pooling"],
            "supervised_transform": prepared.raw_metadata["supervised_transform"],
            "loaded_split_counts": copy.deepcopy(
                prepared.raw_metadata["loaded_split_counts"]
            ),
            "official_fold_counts": copy.deepcopy(
                prepared.raw_metadata["official_fold_counts"]
            ),
            "missing_permitted_fold_ids": copy.deepcopy(
                prepared.raw_metadata["missing_permitted_fold_ids"]
            ),
            "conflict_definition": config["conflict_definition"],
            "source_provenance": copy.deepcopy(prepared.raw_metadata["source_provenance"]),
        }
    )
    errors = validate_raw_value_result(result, expected_config=config)
    result["validation"] = {"ok": not errors, "errors": errors}
    return result


def validate_raw_value_result(
    record: Mapping[str, Any],
    *,
    expected_config: Mapping[str, Any] | None = None,
    expected_mode: str | None = None,
    expected_seed: int | None = None,
    expected_cache_sha256: str | None = None,
) -> list[str]:
    """Validate raw-lane provenance and the reused core result structure."""

    errors: list[str] = []
    if record.get("schema") != RAW_RESULT_SCHEMA:
        errors.append("unexpected raw result schema")
    if record.get("study_schema") != RAW_STUDY_SCHEMA:
        errors.append("unexpected raw study schema")
    if record.get("implementation_sha256") != implementation_sha256():
        errors.append("raw implementation SHA256 does not match the active code")
    if expected_config is not None:
        try:
            converted = _core_config(expected_config)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"raw configuration is invalid: {exc}")
            converted = None
        if record.get("configuration_sha256") != configuration_sha256(expected_config):
            errors.append("raw configuration SHA256 does not match the supplied configuration")
    else:
        converted = None
    if expected_mode is not None and record.get("mode") != expected_mode:
        errors.append("mode does not match the expected value")
    if expected_seed is not None and record.get("seed") != expected_seed:
        errors.append("seed does not match the expected value")

    access = record.get("data_access")
    if not isinstance(access, Mapping):
        errors.append("raw data access record is missing")
    else:
        if access.get("loaded_splits") != ["train", "valid"] or access.get("test_opened") is not False:
            errors.append("raw study access is not restricted to training and validation")
        if access.get("cache_schema") != RAW_CACHE_SCHEMA:
            errors.append("raw cache schema record is invalid")
        if access.get("official_test_policy") != "excluded_from_cache_and_value_study":
            errors.append("raw official-test policy is invalid")
        if access.get("conflict_definition") != "none_unmodified_official_mosei_sentiment":
            errors.append("raw conflict definition is invalid")
        if access.get("sentiment_mode") not in {"nonnegative", "positive"}:
            errors.append("raw sentiment mode is invalid")
        if access.get("alignment") not in {
            "positive_interval_overlap_mean",
            "presegmented_entries",
        }:
            errors.append("raw alignment protocol is invalid")
        if access.get("pooling") != "finite_frame_mean":
            errors.append("raw pooling protocol is invalid")
        if access.get("supervised_transform") != "none":
            errors.append("raw cache must precede every supervised transformation")
        cache_digest = access.get("cache_sha256")
        if not _is_sha256(cache_digest):
            errors.append("raw cache SHA256 is invalid")
        if expected_cache_sha256 is not None and cache_digest != expected_cache_sha256:
            errors.append("raw cache SHA256 does not match the expected value")
        if (
            expected_config is not None
            and expected_config.get("expected_cache_sha256") is not None
            and cache_digest != expected_config["expected_cache_sha256"]
        ):
            errors.append("raw cache SHA256 does not match the supplied configuration")
        raw_dimensions = access.get("raw_feature_dimensions")
        if (
            not isinstance(raw_dimensions, Mapping)
            or set(raw_dimensions) != set(MODALITIES)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in raw_dimensions.values()
            )
        ):
            errors.append("raw feature dimensions are invalid")
        loaded_counts = access.get("loaded_split_counts")
        if (
            not isinstance(loaded_counts, Mapping)
            or set(loaded_counts) != {"train", "valid"}
            or loaded_counts.get("train") != access.get("train_utterances")
            or loaded_counts.get("valid") != access.get("validation_utterances")
        ):
            errors.append("raw loaded-split counts are invalid")
        official_counts = access.get("official_fold_counts")
        if (
            not isinstance(official_counts, Mapping)
            or set(official_counts) != {"train", "valid", "test"}
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in official_counts.values()
            )
        ):
            errors.append("raw official-fold counts are invalid")
        missing_ids = access.get("missing_permitted_fold_ids")
        if (
            not isinstance(missing_ids, Mapping)
            or set(missing_ids) != {"train", "valid"}
            or any(
                not isinstance(values, list)
                or values != sorted(values)
                or len(values) != len(set(values))
                or any(not isinstance(value, str) or not value for value in values)
                for values in missing_ids.values()
            )
        ):
            errors.append("raw missing-fold record is invalid")
        provenance = access.get("source_provenance")
        if not isinstance(provenance, Mapping) or set(provenance) != {*MODALITIES, "labels", "splits"}:
            errors.append("raw source provenance is incomplete")
        elif any(
            not isinstance(entry, Mapping)
            or set(entry)
            != {
                "name",
                "source_file_sha256",
                "permitted_content_sha256",
                "content_scope",
                "bytes",
            }
            or pathlib.Path(str(entry.get("name", ""))).name != entry.get("name")
            or not _is_sha256(entry.get("source_file_sha256"))
            or not _is_sha256(entry.get("permitted_content_sha256"))
            or not isinstance(entry.get("content_scope"), str)
            or not entry["content_scope"]
            or isinstance(entry.get("bytes"), bool)
            or not isinstance(entry.get("bytes"), int)
            or entry["bytes"] < 0
            for entry in provenance.values()
        ):
            errors.append("raw source provenance contains an invalid entry")

    projection = record.get("projection_fit")
    if not isinstance(projection, Mapping):
        errors.append("projection fit record is missing")
    else:
        payload = dict(projection)
        digest = payload.pop("attestation_sha256", None)
        if digest != _json_sha256(payload):
            errors.append("projection fit attestation is invalid")
        partition_record = record.get("group_partitions")
        task_partition = (
            partition_record.get("task_fit", {})
            if isinstance(partition_record, Mapping)
            else {}
        )
        if (
            projection.get("protocol") != "partition_groups_then_fit_task_only_projection"
            or projection.get("fit_partition") != "task_fit"
            or projection.get("label_use") != "task_fit_only"
            or projection.get("application_scope")
            != ["task_fit", "router_fit", "calibration", "official_validation"]
            or projection.get("fit_group_digest") != task_partition.get("group_digest")
            or projection.get("fit_group_hashes") != task_partition.get("group_hashes")
            or projection.get("fit_utterances") != task_partition.get("utterances")
            or projection.get("fit_video_groups") != task_partition.get("video_groups")
        ):
            errors.append("projection fit scope disagrees with the task partition")
        modalities = projection.get("modalities")
        if not isinstance(modalities, Mapping) or set(modalities) != set(MODALITIES):
            errors.append("projection modality record is incomplete")
        elif any(
            not isinstance(entry, Mapping)
            or set(entry)
            != {"dimension", "mean_sha256", "scale_sha256", "direction_sha256"}
            or isinstance(entry["dimension"], bool)
            or not isinstance(entry["dimension"], int)
            or entry["dimension"] <= 0
            or not all(
                _is_sha256(entry[name])
                for name in ("mean_sha256", "scale_sha256", "direction_sha256")
            )
            for entry in modalities.values()
        ):
            errors.append("projection modality record is invalid")
        elif isinstance(access, Mapping) and isinstance(
            access.get("raw_feature_dimensions"), Mapping
        ) and any(
            modalities[name]["dimension"] != access["raw_feature_dimensions"].get(name)
            for name in MODALITIES
        ):
            errors.append("projection dimensions disagree with raw feature dimensions")

    protocol = record.get("core_protocol")
    if not isinstance(protocol, Mapping):
        errors.append("core protocol record is missing")
    elif (
        set(protocol)
        != {
            "result_schema",
            "study_schema",
            "configuration_sha256",
            "implementation_sha256",
            "projected_cache_sha256",
            "router_families",
        }
        or protocol.get("result_schema") != core.RESULT_SCHEMA
        or protocol.get("study_schema") != core.STUDY_SCHEMA
        or protocol.get("implementation_sha256") != core.implementation_sha256()
        or protocol.get("router_families") != list(core.ROUTER_FAMILIES)
        or not _is_sha256(protocol.get("projected_cache_sha256"))
    ):
        errors.append("core protocol record is invalid")
    elif isinstance(access, Mapping) and converted is not None:
        reconstructed = copy.deepcopy(dict(record))
        reconstructed.pop("core_protocol", None)
        reconstructed.pop("projection_fit", None)
        reconstructed["schema"] = protocol.get("result_schema")
        reconstructed["study_schema"] = protocol.get("study_schema")
        reconstructed["configuration_sha256"] = protocol.get("configuration_sha256")
        reconstructed["implementation_sha256"] = protocol.get("implementation_sha256")
        reconstructed["data_access"]["cache_sha256"] = protocol.get("projected_cache_sha256")
        reconstructed["data_access"]["input_representation"] = converted["input_representation"]
        reconstructed["data_access"]["upstream_projection_scope"] = converted[
            "upstream_projection_scope"
        ]
        reconstructed["data_access"]["evidence_status"] = converted["evidence_status"]
        core_errors = core.validate_value_result(
            reconstructed,
            expected_mode=expected_mode,
            expected_seed=expected_seed,
            expected_configuration_sha256=(
                core.configuration_sha256(converted)
            ),
            expected_config=converted,
        )
        errors.extend(f"core: {error}" for error in core_errors)
    else:
        errors.append("core result could not be reconstructed safely")
    return errors
