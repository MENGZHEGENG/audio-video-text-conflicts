"""Adapters for public pre-extracted multimodal benchmark descriptors.

The MMSA release used by MOSI, MOSEI, and CH-SIMS stores one dictionary per
official split.  This module converts those dictionaries into the compact
three-scalar representation consumed by ConflictBench.  The conversion is
deliberately identical to the corrected MOSEI path: sequence rows are
mean-pooled per sample, modality projections are fit on train only, and
validation/test labels never influence the projections.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import pathlib
import pickle
import tempfile
from typing import Any, Mapping

import numpy as np

from .core import Dataset


_MODALITY_ORDER = ("audio", "video", "text")
_CACHE_SCHEMA = "conflictbench.descriptor-cache.v1"


@dataclass(frozen=True)
class DescriptorBundle:
    splits: dict[str, Dataset]
    sample_ids: dict[str, tuple[str, ...]]
    metadata: dict[str, Any]


def _items(value: Any, n: int, *, name: str) -> list[Any]:
    """Normalize a split field while rejecting silent length mismatches."""

    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    try:
        result = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain one value per sample") from exc
    if len(result) != n:
        raise ValueError(f"{name} has {len(result)} values for {n} samples")
    return result


def _pool(value: Any, length: Any | None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim > 1:
        if length is not None:
            try:
                limit = int(length)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid sequence length {length!r}") from exc
            if limit <= 0:
                raise ValueError("sequence lengths must be positive")
            array = array[:limit]
        finite = np.isfinite(array)
        if not finite.any():
            raise ValueError("descriptor sequence contains no finite values")
        array = np.nanmean(np.where(finite, array, np.nan), axis=0)
        array = np.asarray(array, dtype=np.float64).reshape(-1)
    else:
        array = array.reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError("descriptor contains non-finite values")
    return array


def _split_vectors(split: Mapping[str, Any], *, sentiment_mode: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, int]]:
    required = {"audio", "vision", "text", "regression_labels"}
    missing = sorted(required - set(split))
    if missing:
        raise ValueError(f"MMSA split is missing fields: {missing}")
    n = len(split["regression_labels"])
    if n == 0:
        raise ValueError("MMSA splits must contain at least one sample")
    labels_raw = _items(split["regression_labels"], n, name="regression_labels")
    ids_raw = _items(split.get("id", [str(i) for i in range(n)]), n, name="id")
    ids = tuple(str(value) for value in ids_raw)
    if len(set(ids)) != len(ids):
        raise ValueError("sample IDs must be unique within each split")

    vectors: list[np.ndarray] = []
    lengths = {
        "audio": split.get("audio_lengths"),
        "video": split.get("vision_lengths"),
        "text": split.get("text_lengths"),
    }
    pooled: dict[str, list[np.ndarray]] = {name: [] for name in _MODALITY_ORDER}
    for name, source in (("audio", "audio"), ("video", "vision"), ("text", "text")):
        values = _items(split[source], n, name=source)
        length_values = None if lengths[name] is None else _items(lengths[name], n, name=f"{name}_lengths")
        for index, value in enumerate(values):
            pooled[name].append(_pool(value, None if length_values is None else length_values[index]))

    dimensions = {name: int(pooled[name][0].shape[0]) for name in _MODALITY_ORDER}
    for name in _MODALITY_ORDER:
        if any(vector.shape != (dimensions[name],) for vector in pooled[name]):
            raise ValueError(f"inconsistent {name} descriptor dimensions within split")
    def scalar_label(value: Any) -> float:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size != 1:
            raise ValueError("regression_labels must contain one scalar per sample")
        return float(array[0])

    raw_labels = np.asarray([scalar_label(value) for value in labels_raw], dtype=np.float64)
    if not np.isfinite(raw_labels).all():
        raise ValueError("regression_labels contains non-finite values")
    if sentiment_mode == "positive":
        keep = raw_labels != 0
    else:
        keep = np.ones(n, dtype=bool)
    labels = (raw_labels >= 0).astype(np.int64)
    rows = np.flatnonzero(keep)
    matrix = np.asarray([[pooled[name][i] for name in _MODALITY_ORDER] for i in rows], dtype=object)
    # Keep modality arrays separate because public releases use different widths.
    return matrix, labels[rows], tuple(ids[i] for i in rows), dimensions


def _project(train: dict[str, list[np.ndarray]], rows: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    projected: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {}
    for name in _MODALITY_ORDER:
        fit = np.asarray(train[name], dtype=np.float64)
        y = rows["train_y"]
        if len(fit) == 0 or len(np.unique(y)) < 2:
            raise ValueError("training split must contain both binary sentiment classes")
        mean = fit.mean(axis=0)
        scale = fit.std(axis=0)
        scale[scale < 1e-8] = 1.0
        direction = (fit[y == 1].mean(axis=0) - fit[y == 0].mean(axis=0)) / scale
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm < 1e-8:
            raise ValueError(f"{name} projection has zero class separation")
        for split_name, values in rows[f"{name}_raw"].items():
            matrix = (np.asarray(values, dtype=np.float64) - mean) / scale
            projected.setdefault(split_name, np.zeros((len(matrix), 3), dtype=np.float32))[:, _MODALITY_ORDER.index(name)] = (matrix @ direction / norm).astype(np.float32)
        meta[name] = {"dimension": int(fit.shape[1]), "train_entries": int(len(fit))}
    return projected, meta


def load_mmsa_bundle(
    feature_path: str | pathlib.Path,
    *,
    dataset_name: str,
    sentiment_mode: str = "nonnegative",
) -> DescriptorBundle:
    """Load an MMSA processed feature pickle for MOSI, MOSEI, or CH-SIMS."""

    if sentiment_mode not in {"nonnegative", "positive"}:
        raise ValueError("sentiment_mode must be 'nonnegative' or 'positive'")
    path = pathlib.Path(feature_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("MMSA feature file must contain a split mapping")
    split_data: dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...], dict[str, int]]] = {}
    for split_name in ("train", "valid", "test"):
        if split_name not in raw:
            raise ValueError(f"MMSA feature file is missing {split_name} split")
        split_data[split_name] = _split_vectors(raw[split_name], sentiment_mode=sentiment_mode)

    # Fit each modality independently on train and apply the same projection to
    # all folds.  This avoids leaking validation/test statistics into scores.
    train_rows = split_data["train"]
    train_dimensions = train_rows[3]
    for split_name, (_matrix, _labels, _ids, dimensions) in split_data.items():
        if dimensions != train_dimensions:
            raise ValueError(
                f"MMSA feature dimensions in {split_name} do not match train: "
                f"{dimensions} != {train_dimensions}"
            )
    raw_by_modality: dict[str, dict[str, np.ndarray]] = {name: {} for name in _MODALITY_ORDER}
    for split_name, (matrix, labels, _ids, _dims) in split_data.items():
        for index, name in enumerate(_MODALITY_ORDER):
            raw_by_modality[name][split_name] = np.asarray([row[index] for row in matrix], dtype=np.float64)
    train = {name: [row[index] for row in train_rows[0]] for index, name in enumerate(_MODALITY_ORDER)}
    rows: dict[str, Any] = {"train_y": train_rows[1], "audio_raw": raw_by_modality["audio"], "video_raw": raw_by_modality["video"], "text_raw": raw_by_modality["text"]}
    projected, projection_meta = _project(train, rows)

    datasets: dict[str, Dataset] = {}
    sample_ids: dict[str, tuple[str, ...]] = {}
    for split_name, (_matrix, labels, ids, _dims) in split_data.items():
        datasets[split_name] = Dataset(
            x=projected[split_name], y=labels,
            ambiguous=np.zeros(len(labels), dtype=bool),
            mechanism=np.full(len(labels), dataset_name.lower(), dtype="U64"),
        )
        sample_ids[split_name] = ids
    dimensions = split_data["train"][3]
    metadata = {
        "dataset": dataset_name,
        "source_format": "MMSA_processed_pickle",
        "source_path": str(path),
        "sentiment_mode": sentiment_mode,
        "split_counts": {name: int(len(datasets[name].y)) for name in ("train", "valid", "test")},
        "feature_dimensions": dimensions,
        "projection": "train_standardized_mean_difference",
        "projection_fit": projection_meta,
    }
    return DescriptorBundle(splits=datasets, sample_ids=sample_ids, metadata=metadata)


def save_descriptor_cache(bundle: DescriptorBundle, path: str | pathlib.Path) -> pathlib.Path:
    destination = pathlib.Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {"cache_schema": np.asarray(_CACHE_SCHEMA), "metadata_json": np.asarray(json.dumps(bundle.metadata, sort_keys=True))}
    for split_name in ("train", "valid", "test"):
        if split_name not in bundle.splits or split_name not in bundle.sample_ids:
            raise ValueError(f"descriptor cache requires {split_name} split")
        split = bundle.splits[split_name]
        ids = tuple(str(value) for value in bundle.sample_ids[split_name])
        if len(ids) != len(split.y) or len(set(ids)) != len(ids):
            raise ValueError(f"descriptor cache {split_name} IDs and rows are inconsistent")
        arrays[f"{split_name}_x"] = split.x.astype(np.float32)
        arrays[f"{split_name}_y"] = split.y.astype(np.int64)
        arrays[f"{split_name}_sample_ids"] = np.asarray(ids, dtype="U")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent)
    os.close(fd)
    temporary = pathlib.Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_descriptor_cache(path: str | pathlib.Path) -> DescriptorBundle:
    source = pathlib.Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        schema = str(np.asarray(archive["cache_schema"]).reshape(()).item())
        if schema != _CACHE_SCHEMA:
            raise ValueError(f"unsupported descriptor cache schema: {schema}")
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).reshape(()).item()))
        datasets: dict[str, Dataset] = {}
        sample_ids: dict[str, tuple[str, ...]] = {}
        for split_name in ("train", "valid", "test"):
            x = np.asarray(archive[f"{split_name}_x"], dtype=np.float32)
            y = np.asarray(archive[f"{split_name}_y"], dtype=np.int64)
            ids = tuple(str(value) for value in np.asarray(archive[f"{split_name}_sample_ids"]).tolist())
            if x.ndim != 2 or x.shape[1] != 3 or len(y) != len(ids) or len(set(ids)) != len(ids):
                raise ValueError(f"descriptor cache {split_name} has invalid shapes or IDs")
            datasets[split_name] = Dataset(x=x, y=y, ambiguous=np.zeros(len(y), dtype=bool), mechanism=np.full(len(y), str(metadata.get("dataset", "descriptor")), dtype="U64"))
            sample_ids[split_name] = ids
    metadata = dict(metadata)
    metadata["cache_schema"] = _CACHE_SCHEMA
    return DescriptorBundle(splits=datasets, sample_ids=sample_ids, metadata=metadata)
