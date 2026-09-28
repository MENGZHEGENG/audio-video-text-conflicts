"""Unprojected utterance-level descriptor cache for CMU-MOSEI.

This module pools aligned computational sequences but performs no
label-dependent transformation.  The cache contains official training and
validation data only; the held-out official test fold is intentionally absent.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .mosei import (
    _assign_split,
    _decode,
    _feature_group,
    _h5py,
    _read_split_file,
    _video_id,
    iter_csd,
)

MODALITIES = ("audio", "video", "text")
RAW_CACHE_SCHEMA = "conflictbench.mosei-raw-descriptor-cache.v1"
_ALLOWED_SPLITS = ("train", "valid")


@dataclass(frozen=True)
class RawMoseiSplit:
    """Pooled descriptor matrices aligned to labels and video groups."""

    descriptors: dict[str, np.ndarray]
    y: np.ndarray
    sample_ids: tuple[str, ...]
    groups: np.ndarray


@dataclass(frozen=True)
class RawMoseiBundle:
    """Training/validation descriptors plus public-safe source provenance."""

    splits: dict[str, RawMoseiSplit]
    metadata: dict[str, Any]


def _scalar(value: Any) -> Any:
    return np.asarray(value).reshape(()).item()


def _content_sha256(records: Mapping[str, Any], *, dtype: Any) -> str:
    digest = hashlib.sha256()
    for key in sorted(records):
        values = np.ascontiguousarray(np.asarray(records[key], dtype=dtype))
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(values.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(values.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_record(
    path: str | pathlib.Path,
    permitted_content_sha256: str,
    content_scope: str,
) -> dict[str, Any]:
    source = pathlib.Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "name": source.name,
        "source_file_sha256": _file_sha256(source),
        "permitted_content_sha256": permitted_content_sha256,
        "content_scope": content_scope,
        "bytes": int(source.stat().st_size),
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_allowed_labels(
    labels_path: str | pathlib.Path,
    split_ids: Mapping[str, set[str]],
    sentiment_mode: str,
) -> tuple[
    dict[str, int],
    dict[str, str],
    dict[str, list[tuple[str, float, float]]],
    str,
]:
    """Read labels for training/validation groups without opening test values."""

    labels: dict[str, int] = {}
    assignment: dict[str, str] = {}
    windows: dict[str, list[tuple[str, float, float]]] = {}
    layouts: set[str] = set()
    with _h5py().File(labels_path, "r") as handle:
        for raw_key, entry in _feature_group(handle).items():
            key = _decode(raw_key)
            split_name = _assign_split(key, split_ids)
            if split_name not in _ALLOWED_SPLITS:
                continue
            if not hasattr(entry, "keys") or "features" not in entry:
                raise ValueError(f"missing labels for permitted source {key}")
            values = np.asarray(entry["features"][()], dtype=np.float64)
            if "intervals" in entry:
                layouts.add("positive_interval_overlap_mean")
                intervals = np.asarray(entry["intervals"][()], dtype=np.float64)
                if values.ndim != 2 or intervals.shape != (len(values), 2):
                    raise ValueError(f"invalid label sequence shape for {key}")
                if not np.isfinite(intervals).all() or np.any(intervals[:, 1] <= intervals[:, 0]):
                    raise ValueError(f"invalid label intervals for {key}")
                for index, (label_values, interval) in enumerate(zip(values, intervals)):
                    sentiment = float(label_values[0])
                    if not np.isfinite(sentiment) or (sentiment_mode == "positive" and sentiment == 0):
                        continue
                    sample_id = f"{key}[{index}]"
                    labels[sample_id] = int(sentiment >= 0)
                    assignment[sample_id] = split_name
                    windows.setdefault(key, []).append(
                        (sample_id, float(interval[0]), float(interval[1]))
                    )
            else:
                layouts.add("presegmented_entries")
                if values.ndim > 1 and len(values) > 1:
                    raise ValueError(f"multi-entry labels require intervals for {key}")
                sentiment = float(values.reshape(-1)[0])
                if not np.isfinite(sentiment) or (sentiment_mode == "positive" and sentiment == 0):
                    continue
                labels[key] = int(sentiment >= 0)
                assignment[key] = split_name
    if not labels:
        raise ValueError("no finite CMU-MOSEI training or validation labels were found")
    if len(layouts) != 1:
        raise ValueError("CMU-MOSEI labels mix interval-aligned and presegmented layouts")
    return labels, assignment, windows, next(iter(layouts))


def _validate_split(split_name: str, split: RawMoseiSplit) -> None:
    if split_name not in _ALLOWED_SPLITS:
        raise ValueError(f"unsupported raw MOSEI split: {split_name}")
    if set(split.descriptors) != set(MODALITIES):
        raise ValueError(f"{split_name} descriptors must contain exactly {MODALITIES}")
    y = np.asarray(split.y)
    groups = np.asarray(split.groups)
    if y.ndim != 1 or groups.ndim != 1 or len(split.sample_ids) != len(y) or len(groups) != len(y):
        raise ValueError(f"{split_name} labels, identifiers, and groups must align")
    if len(y) == 0 or len(set(split.sample_ids)) != len(split.sample_ids):
        raise ValueError(f"{split_name} must have unique nonempty identifiers")
    if not np.isin(y, (0, 1)).all():
        raise ValueError(f"{split_name} labels must be binary")
    if any(not str(value) for value in groups.tolist()):
        raise ValueError(f"{split_name} group identifiers must be nonempty")
    for modality in MODALITIES:
        values = np.asarray(split.descriptors[modality])
        if values.ndim != 2 or values.shape[0] != len(y) or values.shape[1] == 0:
            raise ValueError(f"{split_name} {modality} descriptors must have shape [n, d]")
        if not np.isfinite(values).all():
            raise ValueError(f"{split_name} {modality} descriptors must be finite")


def _validate_bundle(bundle: RawMoseiBundle) -> None:
    if set(bundle.splits) != set(_ALLOWED_SPLITS):
        raise ValueError("raw MOSEI cache requires training and validation splits only")
    for name in _ALLOWED_SPLITS:
        _validate_split(name, bundle.splits[name])
    train = bundle.splits["train"]
    valid = bundle.splits["valid"]
    if set(train.sample_ids).intersection(valid.sample_ids):
        raise ValueError("training and validation segment identifiers overlap")
    if {str(value) for value in train.groups}.intersection(str(value) for value in valid.groups):
        raise ValueError("training and validation video groups overlap")
    _validate_metadata(bundle.metadata)
    dimensions = bundle.metadata.get("feature_dimensions")
    expected_dimensions = {
        name: int(train.descriptors[name].shape[1]) for name in MODALITIES
    }
    if dimensions != expected_dimensions:
        raise ValueError("raw MOSEI feature dimensions do not match metadata")
    for name in MODALITIES:
        if valid.descriptors[name].shape[1] != expected_dimensions[name]:
            raise ValueError(f"{name} feature dimension changes across splits")
    counts = bundle.metadata.get("loaded_split_counts")
    if counts != {name: len(bundle.splits[name].y) for name in _ALLOWED_SPLITS}:
        raise ValueError("raw MOSEI split counts do not match metadata")


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    if metadata.get("dataset") != "CMU-MOSEI":
        raise ValueError("raw MOSEI metadata has an invalid dataset")
    if metadata.get("supervised_transform") != "none":
        raise ValueError("raw MOSEI metadata must declare no supervised transformation")
    if metadata.get("official_test_policy") != "excluded_from_cache_and_value_study":
        raise ValueError("raw MOSEI metadata has an invalid official test policy")
    if metadata.get("sentiment_mode") not in {"nonnegative", "positive"}:
        raise ValueError("raw MOSEI sentiment mode is invalid")
    if metadata.get("alignment") not in {
        "positive_interval_overlap_mean",
        "presegmented_entries",
    } or metadata.get("pooling") != "finite_frame_mean":
        raise ValueError("raw MOSEI pooling metadata is invalid")
    dimensions = metadata.get("feature_dimensions")
    if (
        not isinstance(dimensions, Mapping)
        or set(dimensions) != set(MODALITIES)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in dimensions.values()
        )
    ):
        raise ValueError("raw MOSEI feature-dimension metadata is invalid")
    loaded_counts = metadata.get("loaded_split_counts")
    if (
        not isinstance(loaded_counts, Mapping)
        or set(loaded_counts) != set(_ALLOWED_SPLITS)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in loaded_counts.values()
        )
    ):
        raise ValueError("raw MOSEI split count metadata is invalid")
    official_counts = metadata.get("official_fold_counts")
    if (
        not isinstance(official_counts, Mapping)
        or set(official_counts) != {"train", "valid", "test"}
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in official_counts.values()
        )
    ):
        raise ValueError("raw MOSEI official fold count metadata is invalid")
    missing_ids = metadata.get("missing_permitted_fold_ids")
    if (
        not isinstance(missing_ids, Mapping)
        or set(missing_ids) != set(_ALLOWED_SPLITS)
        or any(
            not isinstance(values, list)
            or any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))
            or values != sorted(values)
            for values in missing_ids.values()
        )
    ):
        raise ValueError("raw MOSEI missing-fold metadata is invalid")
    provenance = metadata.get("source_provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != {*MODALITIES, "labels", "splits"}:
        raise ValueError("raw MOSEI source provenance is incomplete")
    for record in provenance.values():
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {
                "name",
                "source_file_sha256",
                "permitted_content_sha256",
                "content_scope",
                "bytes",
            }
            or not isinstance(record["name"], str)
            or pathlib.Path(record["name"]).name != record["name"]
            or not _is_sha256(record["source_file_sha256"])
            or not _is_sha256(record["permitted_content_sha256"])
            or not isinstance(record["content_scope"], str)
            or not record["content_scope"]
            or isinstance(record["bytes"], bool)
            or not isinstance(record["bytes"], int)
            or record["bytes"] < 0
        ):
            raise ValueError("raw MOSEI source provenance entry is invalid")


def build_raw_mosei_bundle(
    modality_paths: Mapping[str, str | pathlib.Path],
    labels_path: str | pathlib.Path,
    splits_path: str | pathlib.Path,
    *,
    sentiment_mode: str = "nonnegative",
    max_samples_per_split: int | None = None,
) -> RawMoseiBundle:
    """Pool CSD frames for official training/validation utterances only."""

    if set(modality_paths) != set(MODALITIES):
        raise ValueError(f"modality_paths must contain exactly {MODALITIES}")
    if sentiment_mode not in {"nonnegative", "positive"}:
        raise ValueError("sentiment_mode must be 'nonnegative' or 'positive'")
    if max_samples_per_split is not None and max_samples_per_split <= 0:
        raise ValueError("max_samples_per_split must be positive")

    split_ids = _read_split_file(splits_path)
    split_names = ("train", "valid", "test")
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = split_ids[left].intersection(split_ids[right])
            if overlap:
                preview = ", ".join(sorted(overlap)[:5])
                raise ValueError(
                    f"official MOSEI split group identifiers overlap between "
                    f"{left} and {right}: {preview}"
                )
    labels, assignment, windows, alignment = _read_allowed_labels(
        labels_path, split_ids, sentiment_mode
    )
    descriptors_by_modality: dict[str, dict[str, np.ndarray]] = {}
    for modality in MODALITIES:
        descriptors_by_modality[modality] = dict(
            iter_csd(
                modality_paths[modality],
                wanted=set(labels),
                windows=windows,
            )
        )
    shared = set(labels)
    for descriptors in descriptors_by_modality.values():
        shared.intersection_update(descriptors)
    if not shared:
        raise ValueError("CMU-MOSEI sources have no shared permitted utterances")

    by_split = {
        name: [sample_id for sample_id in sorted(shared) if assignment[sample_id] == name]
        for name in _ALLOWED_SPLITS
    }
    if any(not values for values in by_split.values()):
        raise ValueError("CMU-MOSEI sources do not populate training and validation")
    splits: dict[str, RawMoseiSplit] = {}
    for split_name, sample_ids in by_split.items():
        if max_samples_per_split is not None:
            sample_ids = sample_ids[:max_samples_per_split]
        matrices: dict[str, np.ndarray] = {}
        for modality in MODALITIES:
            vectors = [descriptors_by_modality[modality][sample_id] for sample_id in sample_ids]
            dimensions = {int(vector.shape[0]) for vector in vectors}
            if len(dimensions) != 1:
                raise ValueError(f"inconsistent {modality} feature dimensions")
            matrices[modality] = np.asarray(vectors, dtype=np.float32)
        splits[split_name] = RawMoseiSplit(
            descriptors=matrices,
            y=np.asarray([labels[sample_id] for sample_id in sample_ids], dtype=np.int64),
            sample_ids=tuple(sample_ids),
            groups=np.asarray([_video_id(sample_id) for sample_id in sample_ids], dtype="U"),
        )

    train_dimensions = {
        modality: int(splits["train"].descriptors[modality].shape[1])
        for modality in MODALITIES
    }
    represented = {
        split_name: {str(value) for value in splits[split_name].groups}
        for split_name in _ALLOWED_SPLITS
    }
    metadata = {
        "dataset": "CMU-MOSEI",
        "sentiment_mode": sentiment_mode,
        "alignment": alignment,
        "pooling": "finite_frame_mean",
        "supervised_transform": "none",
        "loaded_split_counts": {name: len(splits[name].y) for name in _ALLOWED_SPLITS},
        "official_fold_counts": {
            name: len(split_ids[name]) for name in ("train", "valid", "test")
        },
        "missing_permitted_fold_ids": {
            name: sorted(set(split_ids[name]) - represented[name]) for name in _ALLOWED_SPLITS
        },
        "feature_dimensions": train_dimensions,
        "source_provenance": {
            **{
                name: _source_record(
                    modality_paths[name],
                    _content_sha256(descriptors_by_modality[name], dtype=np.float64),
                    "official_train_and_validation_pooled_values",
                )
                for name in MODALITIES
            },
            "labels": _source_record(
                labels_path,
                _content_sha256(labels, dtype=np.int64),
                "official_train_and_validation_labels",
            ),
            "splits": _source_record(
                splits_path,
                _content_sha256(
                    {
                        f"{split_name}:{value}": np.asarray([], dtype=np.int8)
                        for split_name in _ALLOWED_SPLITS
                        for value in split_ids[split_name]
                    },
                    dtype=np.int8,
                ),
                "official_train_and_validation_group_ids",
            ),
        },
        "official_test_policy": "excluded_from_cache_and_value_study",
    }
    bundle = RawMoseiBundle(splits=splits, metadata=metadata)
    _validate_bundle(bundle)
    return bundle


def save_raw_mosei_cache(bundle: RawMoseiBundle, path: str | pathlib.Path) -> pathlib.Path:
    """Write a compressed unprojected descriptor cache atomically."""

    _validate_bundle(bundle)
    destination = pathlib.Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(bundle.metadata)
    metadata["cache_schema"] = RAW_CACHE_SCHEMA
    arrays: dict[str, Any] = {
        "cache_schema": np.asarray(RAW_CACHE_SCHEMA),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for split_name in _ALLOWED_SPLITS:
        split = bundle.splits[split_name]
        for modality in MODALITIES:
            arrays[f"{split_name}_{modality}"] = np.asarray(
                split.descriptors[modality], dtype=np.float32
            )
        arrays[f"{split_name}_y"] = np.asarray(split.y, dtype=np.int64)
        arrays[f"{split_name}_sample_ids"] = np.asarray(split.sample_ids, dtype="U")
        arrays[f"{split_name}_groups"] = np.asarray(split.groups, dtype="U")

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent
    )
    os.close(fd)
    temporary = pathlib.Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_raw_mosei_split(
    path: str | pathlib.Path,
    split_name: str,
) -> tuple[RawMoseiSplit, dict[str, Any]]:
    """Load one permitted split without touching arrays under other names."""

    if split_name not in _ALLOWED_SPLITS:
        raise ValueError("only train and valid raw MOSEI splits may be opened")
    source = pathlib.Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    required = {
        "cache_schema",
        "metadata_json",
        *(
            field
            for allowed_split in _ALLOWED_SPLITS
            for field in (
                f"{allowed_split}_y",
                f"{allowed_split}_sample_ids",
                f"{allowed_split}_groups",
                *(f"{allowed_split}_{name}" for name in MODALITIES),
            )
        ),
    }
    with np.load(source, allow_pickle=False) as archive:
        fields = set(archive.files)
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"raw MOSEI cache is missing fields: {missing}")
        unexpected = sorted(fields - required)
        if unexpected:
            raise ValueError(f"raw MOSEI cache contains unexpected fields: {unexpected}")
        schema = str(_scalar(archive["cache_schema"]))
        if schema != RAW_CACHE_SCHEMA:
            raise ValueError(f"unsupported raw MOSEI cache schema: {schema}")
        metadata = json.loads(str(_scalar(archive["metadata_json"])))
        if not isinstance(metadata, dict):
            raise TypeError("raw MOSEI cache metadata must be an object")
        _validate_metadata(metadata)
        split = RawMoseiSplit(
            descriptors={
                name: np.asarray(archive[f"{split_name}_{name}"], dtype=np.float64)
                for name in MODALITIES
            },
            y=np.asarray(archive[f"{split_name}_y"], dtype=np.int64),
            sample_ids=tuple(str(value) for value in archive[f"{split_name}_sample_ids"].tolist()),
            groups=np.asarray(archive[f"{split_name}_groups"], dtype="U"),
        )
    _validate_split(split_name, split)
    if int(metadata["loaded_split_counts"][split_name]) != len(split.y):
        raise ValueError(f"raw MOSEI {split_name} split count disagrees with metadata")
    for modality in MODALITIES:
        if int(metadata["feature_dimensions"][modality]) != split.descriptors[modality].shape[1]:
            raise ValueError(f"raw MOSEI {modality} dimension disagrees with metadata")
    metadata = dict(metadata)
    metadata["cache_schema"] = RAW_CACHE_SCHEMA
    return split, metadata
