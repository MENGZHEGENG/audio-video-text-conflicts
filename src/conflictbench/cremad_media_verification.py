"""Byte-level verification for CREMA-D media admitted from official LFS pointers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


SCHEMA = "conflictbench.cremad-media-verification.v1"
POINTER_SCHEMA = "conflictbench.cremad-pointer-index.v1"
CLIP_ID = re.compile(r"[0-9]{4}_[A-Z]{3}_[A-Z]{3}_[A-Z]{2}")
MEDIA = {"audio": ("AudioWAV", ".wav"), "video": ("VideoFlash", ".flv")}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, error: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(error)
    return value


def verify_media(
    repository_root: Path, pointer_index_path: Path
) -> dict[str, Any]:
    """Recompute every selected media digest and reject any unexpected byte."""

    raw_index = pointer_index_path.read_bytes()
    pointer_index = json.loads(raw_index)
    if not isinstance(pointer_index, Mapping) or pointer_index.get("schema") != POINTER_SCHEMA:
        raise ValueError("pointer index")
    records = pointer_index.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("pointer records")

    seen: set[str] = set()
    media_count = 0
    total_bytes = 0
    for record_value in records:
        record = _mapping(record_value, "pointer records")
        clip_id = record.get("clip_id")
        if not isinstance(clip_id, str) or CLIP_ID.fullmatch(clip_id) is None or clip_id in seen:
            raise ValueError("pointer records")
        seen.add(clip_id)
        for modality, (directory, suffix) in MEDIA.items():
            expected = _mapping(record.get(modality), "pointer records")
            expected_digest = expected.get("sha256")
            expected_size = expected.get("size_bytes")
            if (
                not isinstance(expected_digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
                or not isinstance(expected_size, int)
                or isinstance(expected_size, bool)
                or expected_size <= 0
            ):
                raise ValueError("pointer records")
            path = repository_root / directory / f"{clip_id}{suffix}"
            if not path.is_file() or path.stat().st_size != expected_size:
                raise ValueError(f"media bytes: {clip_id} {modality}")
            if _sha256(path) != expected_digest:
                raise ValueError(f"media bytes: {clip_id} {modality}")
            media_count += 1
            total_bytes += expected_size

    return {
        "schema": SCHEMA,
        "pointer_index_sha256": hashlib.sha256(raw_index).hexdigest(),
        "pointer_record_index_sha256": pointer_index.get("record_index_sha256"),
        "repository": pointer_index.get("repository"),
        "commit": pointer_index.get("commit"),
        "clip_count": len(seen),
        "media_count": media_count,
        "total_bytes": total_bytes,
    }
