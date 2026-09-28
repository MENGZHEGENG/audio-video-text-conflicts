"""Loader for aligned CMU-MOSEI computational sequences (CSD files).

The benchmark policies consume one scalar per modality.  This module turns
the official aligned CSD features into those scores with a train-only
standardized mean-difference projection.  No labels from validation or test
segments are used when fitting the projections.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .core import Dataset

_MODALITY_ORDER = ("audio", "video", "text")
_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "valid": "valid",
    "validation": "valid",
    "dev": "valid",
    "test": "test",
    "testing": "test",
}
_CACHE_SCHEMA = "conflictbench.mosei-cache.v2"


@dataclass(frozen=True)
class MoseiBundle:
    """Real-data splits plus provenance needed in a run record."""

    splits: dict[str, Dataset]
    sample_ids: dict[str, tuple[str, ...]]
    metadata: dict[str, Any]


def _cache_scalar(value: Any) -> Any:
    """Extract a scalar stored as a zero-dimensional NumPy array."""

    return np.asarray(value).reshape(()).item()


def _h5py() -> Any:
    try:
        import h5py
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("CMU-MOSEI loading requires h5py; install conflictbench[mosei]") from exc
    return h5py


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def _feature_group(handle: Any) -> Any:
    """Return the CSD entry group for both official and small test fixtures."""

    if "data" in handle and hasattr(handle["data"], "keys"):
        return handle["data"]
    # The official labels release wraps ``data`` below a modality-named group
    # (for example ``/All Labels/data``), while feature releases use
    # ``/data``.  Keep the fallback bounded to top-level groups so metadata
    # datasets are never mistaken for segment entries.
    for key in handle:
        candidate = handle[key]
        if hasattr(candidate, "keys") and "data" in candidate and hasattr(candidate["data"], "keys"):
            return candidate["data"]
    return handle


def _finite_vector(values: Any, *, reduce_sequence: bool) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim > 1 and reduce_sequence:
        array = np.nanmean(np.where(np.isfinite(array), array, np.nan), axis=0)
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    if not np.isfinite(array).all():
        return np.array([], dtype=np.float64)
    return array


def iter_csd(
    path: str | pathlib.Path,
    *,
    reduce_sequence: bool = True,
    wanted: set[str] | None = None,
    windows: Mapping[str, Sequence[tuple[str, float, float]]] | None = None,
):
    """Yield finite ``(segment_id, feature)`` pairs from a CSD file.

    ``wanted`` avoids reading feature datasets for unrelated segments.  The
    generator opens one HDF5 file at a time, which is important for the large
    MOSEI files and keeps the loader's memory use independent of file size.
    """

    h5py = _h5py()
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        group = _feature_group(handle)
        for raw_key in group:
            key = _decode(raw_key)
            targets = windows.get(key, ()) if windows is not None else ()
            if wanted is not None and key not in wanted and not any(item[0] in wanted for item in targets):
                continue
            entry = group[raw_key]
            if not hasattr(entry, "keys") or "features" not in entry:
                continue
            if targets:
                if "intervals" not in entry:
                    raise ValueError(f"missing feature intervals for {key} in {path}")
                features = np.asarray(entry["features"][()], dtype=np.float64)
                intervals = np.asarray(entry["intervals"][()], dtype=np.float64)
                if features.ndim != 2 or intervals.shape != (len(features), 2):
                    raise ValueError(f"invalid sequence shape for {key} in {path}")
                if not np.isfinite(intervals).all() or np.any(intervals[:, 1] < intervals[:, 0]):
                    raise ValueError(f"invalid feature intervals for {key} in {path}")
                for sample_id, start, end in targets:
                    if wanted is not None and sample_id not in wanted:
                        continue
                    overlap = (intervals[:, 0] < end) & (intervals[:, 1] > start)
                    if not overlap.any():
                        continue
                    vector = _finite_vector(features[overlap], reduce_sequence=True)
                    if len(vector):
                        yield sample_id, vector
                continue
            vector = _finite_vector(entry["features"][()], reduce_sequence=reduce_sequence)
            if len(vector):
                yield key, vector


def read_csd(path: str | pathlib.Path, *, reduce_sequence: bool = True) -> dict[str, np.ndarray]:
    """Read ``entry/features`` vectors from a CMU-MOSEI CSD file.

    CSD is an HDF5 format.  Entries can live below ``/data`` (the feature
    release layout), below a top-level modality group (the labels release), or
    at the file root, which keeps tiny fixtures easy to construct.
    Sequence features are averaged over time by default.  Large-data callers
    should use :func:`iter_csd` so they do not retain every feature vector.
    """

    records = dict(iter_csd(path, reduce_sequence=reduce_sequence))
    if not records:
        raise ValueError(f"no finite feature entries found in {path}")
    return records


def _video_id(sample_id: str) -> str:
    """Extract the video-level identifier used by the official MOSEI folds."""

    match = re.match(r"^(.*?)(?:\[\d+\]|_\d+)$", sample_id)
    return match.group(1) if match else sample_id


def _read_split_file(path: str | pathlib.Path) -> dict[str, set[str]]:
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("MOSEI split file must be a JSON object")
    splits: dict[str, set[str]] = {}
    split_sources: dict[str, str] = {}
    for name, values in raw.items():
        canonical = _SPLIT_ALIASES.get(str(name).lower())
        if canonical is None:
            continue
        if canonical in splits:
            raise ValueError(
                "multiple MOSEI split entries map to "
                f"{canonical!r}: {split_sources[canonical]!r} and {str(name)!r}"
            )
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError(f"split {name!r} must contain a list of IDs")
        decoded = [_decode(value) for value in values]
        if len(decoded) != len(set(decoded)):
            raise ValueError(f"split {name!r} contains duplicate group identifiers")
        splits[canonical] = set(decoded)
        split_sources[canonical] = str(name)
    missing = sorted({"train", "valid", "test"} - set(splits))
    if missing:
        raise ValueError(f"MOSEI split file is missing: {missing}")
    ordered = ("train", "valid", "test")
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            overlap = splits[left].intersection(splits[right])
            if overlap:
                preview = ", ".join(sorted(overlap)[:5])
                raise ValueError(
                    f"MOSEI split group identifiers overlap between {left} and {right}: {preview}"
                )
    return splits


def _assign_split(sample_id: str, split_ids: Mapping[str, set[str]]) -> str | None:
    for split_name in ("train", "valid", "test"):
        candidates = split_ids[split_name]
        if sample_id in candidates or _video_id(sample_id) in candidates:
            return split_name
    return None


def _fit_projection_csd(
    path: str | pathlib.Path,
    train_ids: Sequence[str],
    labels: Mapping[str, int],
    windows: Mapping[str, Sequence[tuple[str, float, float]]] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Fit a projection from one CSD with bounded memory use."""

    train_wanted = set(train_ids)
    count = 0
    positive_count = 0
    negative_count = 0
    total: np.ndarray | None = None
    total_sq: np.ndarray | None = None
    positive_sum: np.ndarray | None = None
    negative_sum: np.ndarray | None = None
    for key, vector in iter_csd(path, wanted=train_wanted, windows=windows):
        label = labels.get(key)
        if label is None:
            continue
        if total is None:
            total = np.zeros_like(vector, dtype=np.float64)
            total_sq = np.zeros_like(vector, dtype=np.float64)
            positive_sum = np.zeros_like(vector, dtype=np.float64)
            negative_sum = np.zeros_like(vector, dtype=np.float64)
        if vector.shape != total.shape:
            raise ValueError(f"inconsistent feature dimensions in {path}")
        vector = vector.astype(np.float64, copy=False)
        count += 1
        total += vector
        total_sq += vector * vector
        if label == 1:
            positive_count += 1
            positive_sum += vector
        else:
            negative_count += 1
            negative_sum += vector

    if count == 0 or total is None or total_sq is None or positive_sum is None or negative_sum is None:
        raise ValueError("no shared MOSEI training entries for a modality")
    if positive_count == 0 or negative_count == 0:
        raise ValueError("MOSEI training split must contain both binary sentiment classes")

    mean = total / count
    variance = np.maximum(total_sq / count - mean * mean, 0.0)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    direction = (positive_sum / positive_count - negative_sum / negative_count) / scale
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("modality projection has zero class separation")

    scores: dict[str, float] = {}
    wanted = set(labels)
    for key, vector in iter_csd(path, wanted=wanted, windows=windows):
        if vector.shape != mean.shape:
            raise ValueError(f"inconsistent feature dimensions in {path}")
        normalized = (vector.astype(np.float64, copy=False) - mean) / scale
        scores[key] = float(np.dot(normalized, direction) / norm)
    return scores, {"dimension": int(mean.shape[0]), "train_entries": count}


def load_mosei_bundle(
    modality_paths: Mapping[str, str | pathlib.Path],
    labels_path: str | pathlib.Path,
    splits_path: str | pathlib.Path,
    *,
    sentiment_mode: str = "nonnegative",
    max_samples_per_split: int | None = None,
) -> MoseiBundle:
    """Load aligned MOSEI scores for ``train``, ``valid``, and ``test``.

    ``modality_paths`` must provide ``audio``, ``video``, and ``text`` CSD
    files.  ``sentiment_mode`` is either ``nonnegative`` (zero is positive)
    or ``positive`` (zero-valued sentiment entries are omitted).
    """

    if set(modality_paths) != set(_MODALITY_ORDER):
        raise ValueError(f"modality_paths must contain exactly {_MODALITY_ORDER}")
    if sentiment_mode not in {"nonnegative", "positive"}:
        raise ValueError("sentiment_mode must be 'nonnegative' or 'positive'")

    labels: dict[str, int] = {}
    windows: dict[str, list[tuple[str, float, float]]] = {}
    with _h5py().File(labels_path, "r") as handle:
        for raw_key, entry in _feature_group(handle).items():
            if not hasattr(entry, "keys") or "features" not in entry:
                continue
            key = _decode(raw_key)
            values = np.asarray(entry["features"][()], dtype=np.float64)
            if "intervals" in entry:
                intervals = np.asarray(entry["intervals"][()], dtype=np.float64)
                if values.ndim != 2 or intervals.shape != (len(values), 2):
                    raise ValueError(f"invalid label sequence shape for {key}")
                if not np.isfinite(intervals).all() or np.any(intervals[:, 1] <= intervals[:, 0]):
                    raise ValueError(f"invalid label intervals for {key}")
                for index, (row, interval) in enumerate(zip(values, intervals)):
                    sentiment = float(row[0])
                    if not np.isfinite(sentiment) or (sentiment_mode == "positive" and sentiment == 0):
                        continue
                    sample_id = f"{key}[{index}]"
                    labels[sample_id] = int(sentiment >= 0)
                    windows.setdefault(key, []).append((sample_id, float(interval[0]), float(interval[1])))
            else:
                if values.ndim > 1 and len(values) > 1:
                    raise ValueError(f"multi-row labels require intervals for {key}")
                sentiment = float(values.reshape(-1)[0])
                if np.isfinite(sentiment) and not (sentiment_mode == "positive" and sentiment == 0):
                    labels[key] = int(sentiment >= 0)

    split_ids = _read_split_file(splits_path)
    candidates = {key: _assign_split(key, split_ids) for key in sorted(labels)}
    assigned = {key: split for key, split in candidates.items() if split is not None}
    if not assigned:
        raise ValueError("no shared MOSEI entries match the supplied official folds")
    scores_by_modality: dict[str, dict[str, float]] = {}
    projection_meta: dict[str, Any] = {}
    for name in _MODALITY_ORDER:
        scores_by_modality[name], projection_meta[name] = _fit_projection_csd(
            modality_paths[name],
            [key for key, split in assigned.items() if split == "train"],
            labels,
            windows,
        )

    shared = set(assigned)
    for scores in scores_by_modality.values():
        shared &= set(scores)
    if not shared:
        raise ValueError("MOSEI CSD files have no shared segment IDs")
    assigned = {key: assigned[key] for key in sorted(shared)}
    by_split = {name: [key for key in assigned if assigned[key] == name] for name in ("train", "valid", "test")}
    if any(not values for values in by_split.values()):
        raise ValueError("supplied MOSEI folds do not match all three CSD splits")

    # Fold files list video IDs, while CSD entries are usually video-segment
    # IDs.  Keep both the official counts and unmatched IDs so a mirror that
    # silently omits labeled videos cannot be mistaken for a complete fold.
    official_fold_counts = {name: len(split_ids[name]) for name in ("train", "valid", "test")}
    missing_official_fold_ids: dict[str, list[str]] = {}
    for split_name, ids in by_split.items():
        represented = set(ids)
        represented.update(_video_id(key) for key in ids)
        missing_official_fold_ids[split_name] = sorted(set(split_ids[split_name]) - represented)

    datasets: dict[str, Dataset] = {}
    sample_ids: dict[str, tuple[str, ...]] = {}
    for split_name, ids in by_split.items():
        if max_samples_per_split is not None:
            if max_samples_per_split <= 0:
                raise ValueError("max_samples_per_split must be positive")
            ids = ids[:max_samples_per_split]
        x = np.asarray([[scores_by_modality[name][key] for name in _MODALITY_ORDER] for key in ids], dtype=np.float32)
        y = np.asarray([labels[key] for key in ids], dtype=np.int64)
        datasets[split_name] = Dataset(
            x=x,
            y=y,
            ambiguous=np.zeros(len(ids), dtype=bool),
            mechanism=np.full(len(ids), "mosei", dtype="U16"),
        )
        sample_ids[split_name] = tuple(ids)

    metadata = {
        "dataset": "CMU-MOSEI",
        "sentiment_mode": sentiment_mode,
        "label_fold_candidates": len(candidates),
        "shared_entries": len(shared),
        "assigned_entries": len(assigned),
        "official_fold_counts": official_fold_counts,
        "observed_split_counts": {name: len(ids) for name, ids in by_split.items()},
        "loaded_split_counts": {name: len(split.y) for name, split in datasets.items()},
        "missing_official_fold_ids": missing_official_fold_ids,
        "feature_dimensions": {name: projection_meta[name]["dimension"] for name in _MODALITY_ORDER},
        "projection": "train_standardized_mean_difference",
        "alignment": "positive_interval_overlap_mean" if windows else "presegmented_entries",
        "projection_fit": projection_meta,
    }
    return MoseiBundle(splits=datasets, sample_ids=sample_ids, metadata=metadata)


def save_mosei_cache(bundle: MoseiBundle, path: str | pathlib.Path) -> pathlib.Path:
    """Write compact projected scores atomically for multi-seed evaluation.

    The cache contains three scalar observations per segment, labels, sample
    IDs, and the loader's provenance metadata.  It contains no raw or frame-
    level features, so repeated seeds do not reread the large CSD files.
    """

    destination = pathlib.Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "cache_schema": np.asarray(_CACHE_SCHEMA),
        "metadata_json": np.asarray(json.dumps(bundle.metadata, sort_keys=True)),
    }
    for split_name in ("train", "valid", "test"):
        if split_name not in bundle.splits or split_name not in bundle.sample_ids:
            raise ValueError(f"MOSEI cache requires {split_name} split")
        split = bundle.splits[split_name]
        arrays[f"{split_name}_x"] = np.asarray(split.x, dtype=np.float32)
        arrays[f"{split_name}_y"] = np.asarray(split.y, dtype=np.int64)
        arrays[f"{split_name}_sample_ids"] = np.asarray(bundle.sample_ids[split_name], dtype="U")

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


def load_mosei_cache(path: str | pathlib.Path) -> MoseiBundle:
    """Read a projected MOSEI cache produced by :func:`save_mosei_cache`."""

    source = pathlib.Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        archive = np.load(source, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"could not read MOSEI cache {source}: {exc}") from exc
    with archive:
        try:
            schema = str(_cache_scalar(archive["cache_schema"]))
            if schema != _CACHE_SCHEMA:
                raise ValueError(f"unsupported MOSEI cache schema: {schema}")
            metadata = json.loads(str(_cache_scalar(archive["metadata_json"])))
            if not isinstance(metadata, dict):
                raise TypeError("MOSEI cache metadata must be an object")
            datasets: dict[str, Dataset] = {}
            sample_ids: dict[str, tuple[str, ...]] = {}
            for split_name in ("train", "valid", "test"):
                x = np.asarray(archive[f"{split_name}_x"], dtype=np.float32)
                y = np.asarray(archive[f"{split_name}_y"], dtype=np.int64)
                ids = tuple(str(value) for value in np.asarray(archive[f"{split_name}_sample_ids"]).tolist())
                if len(ids) != len(y):
                    raise ValueError(f"MOSEI cache {split_name} IDs and labels have different lengths")
                datasets[split_name] = Dataset(
                    x=x,
                    y=y,
                    ambiguous=np.zeros(len(y), dtype=bool),
                    mechanism=np.full(len(y), "mosei", dtype="U16"),
                )
                sample_ids[split_name] = ids
        except KeyError as exc:
            raise ValueError(f"MOSEI cache is missing field: {exc.args[0]}") from exc
    metadata = dict(metadata)
    metadata["cache_schema"] = _CACHE_SCHEMA
    return MoseiBundle(splits=datasets, sample_ids=sample_ids, metadata=metadata)
