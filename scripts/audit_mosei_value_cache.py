#!/usr/bin/env python3
"""Validate a CMU-MOSEI v2 cache before value-of-information experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np


CACHE_SCHEMA = "conflictbench.mosei-cache.v2"
AUDIT_SCHEMA = "conflictbench.mosei-value-cache-audit.v1"
SPLITS = ("train", "valid", "test")
EXPECTED_COUNTS = {"train": 16_327, "valid": 1_871, "test": 4_662}
REQUIRED_ARRAYS = frozenset(
    {"cache_schema", "metadata_json"}
    | {
        f"{split}_{suffix}"
        for split in SPLITS
        for suffix in ("x", "y", "sample_ids")
    }
)
_UTTERANCE_ID = re.compile(r"^([^\[\]\s]+)\[(0|[1-9][0-9]*)\]$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _scalar_text(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind != "U":
        raise ValueError(f"{name} must be a scalar Unicode string")
    return str(array.item())


def _video_group(sample_id: str) -> str:
    match = _UTTERANCE_ID.fullmatch(sample_id)
    if match is None:
        raise ValueError(f"invalid utterance sample ID: {sample_id!r}")
    return match.group(1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_metadata(metadata: Any) -> Mapping[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must encode a JSON object")
    if metadata.get("dataset") != "CMU-MOSEI":
        raise ValueError("metadata_json dataset must be CMU-MOSEI")
    if metadata.get("alignment") != "positive_interval_overlap_mean":
        raise ValueError("metadata_json must describe utterance alignment")
    if metadata.get("loaded_split_counts") != EXPECTED_COUNTS:
        raise ValueError("metadata_json loaded_split_counts do not match the required counts")
    return metadata


def audit_cache(cache_path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    """Return a structural audit record without reporting test labels or scores."""

    source = Path(cache_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    digest = _sha256(source)
    if expected_sha256 is not None:
        expected = expected_sha256.strip().lower()
        if _SHA256.fullmatch(expected) is None:
            raise ValueError("expected SHA256 must contain exactly 64 hexadecimal characters")
        if digest != expected:
            raise ValueError(f"cache SHA256 mismatch: expected {expected}, observed {digest}")

    try:
        archive = np.load(source, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"could not read MOSEI cache {source}: {exc}") from exc

    split_records: dict[str, dict[str, int]] = {}
    seen_sample_ids: set[str] = set()
    groups_by_split: dict[str, set[str]] = {}
    with archive:
        available = set(archive.files)
        if available != REQUIRED_ARRAYS:
            missing = sorted(REQUIRED_ARRAYS - available)
            unexpected = sorted(available - REQUIRED_ARRAYS)
            raise ValueError(
                f"MOSEI v2 cache arrays differ from the required set; "
                f"missing={missing}, unexpected={unexpected}"
            )

        schema = _scalar_text(archive["cache_schema"], "cache_schema")
        if schema != CACHE_SCHEMA:
            raise ValueError(f"unsupported MOSEI cache schema: {schema}")
        metadata_text = _scalar_text(archive["metadata_json"], "metadata_json")
        try:
            metadata = json.loads(metadata_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"metadata_json is not valid JSON: {exc}") from exc
        _validate_metadata(metadata)

        for split in SPLITS:
            expected_count = EXPECTED_COUNTS[split]
            x = np.asarray(archive[f"{split}_x"])
            y = np.asarray(archive[f"{split}_y"])
            raw_ids = np.asarray(archive[f"{split}_sample_ids"])

            if x.dtype != np.dtype(np.float32) or x.shape != (expected_count, 3):
                raise ValueError(
                    f"{split}_x must have dtype float32 and shape ({expected_count}, 3)"
                )
            if y.dtype != np.dtype(np.int64) or y.shape != (expected_count,):
                raise ValueError(
                    f"{split}_y must have dtype int64 and shape ({expected_count},)"
                )
            if raw_ids.dtype.kind != "U" or raw_ids.shape != (expected_count,):
                raise ValueError(
                    f"{split}_sample_ids must be a Unicode vector of length {expected_count}"
                )
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f"{split} contains non-finite numeric values")
            if not np.isin(y, (0, 1)).all():
                raise ValueError(f"{split}_y must contain only binary labels")

            sample_ids = [str(value) for value in raw_ids.tolist()]
            if len(set(sample_ids)) != len(sample_ids):
                raise ValueError(f"{split} contains duplicate sample IDs")
            overlap = seen_sample_ids.intersection(sample_ids)
            if overlap:
                raise ValueError(f"sample IDs overlap across splits: {sorted(overlap)[:3]}")
            seen_sample_ids.update(sample_ids)

            groups = {_video_group(sample_id) for sample_id in sample_ids}
            groups_by_split[split] = groups
            split_records[split] = {
                "samples": expected_count,
                "video_groups": len(groups),
            }

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = groups_by_split[left].intersection(groups_by_split[right])
            if overlap:
                raise ValueError(
                    f"video groups overlap between {left} and {right}: {sorted(overlap)[:3]}"
                )

    return {
        "schema": AUDIT_SCHEMA,
        "cache": {
            "filename": source.name,
            "schema": CACHE_SCHEMA,
            "sha256": digest,
        },
        "checks": {
            "arrays_exact": True,
            "finite_numeric_values": True,
            "sample_ids_unique": True,
            "utterance_ids_valid": True,
            "video_groups_disjoint": True,
        },
        "splits": split_records,
        "test_split_use": "structural_audit_only",
    }


def write_audit(record: Mapping[str, Any], output_path: str | Path) -> Path:
    """Create a JSON audit record without replacing an existing file."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="CMU-MOSEI v2 .npz cache")
    parser.add_argument("--output", required=True, help="new JSON audit record")
    parser.add_argument(
        "--expected-sha256",
        help="optional expected lowercase or uppercase SHA256 digest",
    )
    args = parser.parse_args()

    record = audit_cache(args.cache, expected_sha256=args.expected_sha256)
    write_audit(record, args.output)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
