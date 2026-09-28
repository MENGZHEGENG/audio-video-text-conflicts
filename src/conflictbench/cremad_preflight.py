"""Pointer-only provenance checks for the official CREMA-D repository."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


SCHEMA = "conflictbench.cremad-pointer-index.v1"
SUMMARY_COLUMNS = (
    "FileName",
    "VoiceVote",
    "VoiceLevel",
    "FaceVote",
    "FaceLevel",
    "MultiModalVote",
    "MultiModalLevel",
)
VOTES = frozenset({"A", "D", "F", "H", "N", "S"})
LFS_POINTER = re.compile(
    r"\Aversion https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([1-9][0-9]*)\n?\Z"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError("repository revision")
    value = completed.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("repository revision")
    return value


def _pointer(path: Path) -> dict[str, Any]:
    match = LFS_POINTER.fullmatch(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"lfs pointer: {path.name}")
    return {"sha256": match.group(1), "size_bytes": int(match.group(2))}


def parse_vote_set(value: str) -> tuple[str, ...]:
    """Parse CREMA-D's colon-delimited tied perceptual votes without loss."""

    tokens = tuple(token.strip() for token in value.strip().split(":"))
    if not tokens or any(token not in VOTES for token in tokens):
        raise ValueError("label values")
    if len(set(tokens)) != len(tokens):
        raise ValueError("label values")
    return tokens


def build_pointer_index(repo_root: Path, expected_commit: str) -> dict[str, Any]:
    """Read only the official Git and LFS pointer metadata; never media bytes."""

    if _head(repo_root) != expected_commit:
        raise ValueError("repository revision")
    if not (repo_root / ".git").is_dir():
        raise ValueError("repository root")

    license_path = repo_root / "LICENSE.txt"
    license_text = license_path.read_text(encoding="utf-8")
    normalized_license = " ".join(license_text.split())
    if (
        "Open Database License" not in normalized_license
        or "Database Contents License" not in normalized_license
    ):
        raise ValueError("terms")

    demographics_path = repo_root / "VideoDemographics.csv"
    with demographics_path.open(encoding="utf-8", newline="") as handle:
        demographics = list(csv.DictReader(handle))
    actor_ids = {row.get("ActorID", "").strip() for row in demographics}
    if len(actor_ids) != 91 or "" in actor_ids:
        raise ValueError("actor groups")

    summary_path = repo_root / "processedResults" / "summaryTable.csv"
    with summary_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(SUMMARY_COLUMNS).issubset(reader.fieldnames):
            raise ValueError("label fields")
        rows = list(reader)
    if len(rows) != 7442:
        raise ValueError("summary rows")

    records: list[dict[str, Any]] = []
    integrated_singleton_count = 0
    seen: set[str] = set()
    for row in rows:
        name = row["FileName"].strip()
        if not re.fullmatch(r"[0-9]{4}_[A-Z]{3}_[A-Z]{3}_[A-Z]{2}", name):
            raise ValueError("filename format")
        if name in seen:
            raise ValueError("duplicate filename")
        seen.add(name)
        actor_id = name[:4]
        if actor_id not in actor_ids:
            raise ValueError("actor groups")
        vote_sets = {
            field: parse_vote_set(row[field])
            for field in ("VoiceVote", "FaceVote", "MultiModalVote")
        }
        integrated_singleton_count += len(vote_sets["MultiModalVote"]) == 1
        audio = _pointer(repo_root / "AudioWAV" / f"{name}.wav")
        video = _pointer(repo_root / "VideoFlash" / f"{name}.flv")
        records.append(
            {
                "clip_id": name,
                "actor_id": actor_id,
                "audio": audio,
                "video": video,
            }
        )

    metadata = {
        path: _sha256(repo_root / path)
        for path in (
            "LICENSE.txt",
            "README.md",
            "VideoDemographics.csv",
            "SentenceFilenames.csv",
            "processedResults/summaryTable.csv",
        )
    }
    return {
        "schema": SCHEMA,
        "repository": "https://github.com/CheyneyComputerScience/CREMA-D",
        "commit": expected_commit,
        "terms": "ODbL-1.0 database; DbCL-1.0 contents",
        "actor_count": len(actor_ids),
        "clip_count": len(records),
        "integrated_singleton_count": integrated_singleton_count,
        "integrated_tied_count": len(records) - integrated_singleton_count,
        "vote_encoding": "colon-delimited sets; singleton integrated labels form the fixed primary subset",
        "label_fields": {
            "audio": "VoiceVote",
            "video": "FaceVote",
            "integrated": "MultiModalVote",
        },
        "metadata_sha256": metadata,
        "records": records,
        "record_index_sha256": hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
