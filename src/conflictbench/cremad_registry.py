"""Build a deterministic, actor-disjoint CREMA-D source registry."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from conflictbench.source_registry import SCHEMA


EXPECTED_COMMIT = "1658cd342dff90010aa843eaeebd53610a08b1dc"
EXPECTED_ACTORS = 91
EXPECTED_CLIPS = 7442
ROLE_COUNTS = {"task_fit": 45, "router_fit": 15, "calibration": 15, "evaluation": 16}
ROLE_SALT = "conflictbench.cremad.actor-role.v1:"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def allocate_actor_roles(actor_ids: set[str]) -> dict[str, list[str]]:
    """Allocate all official actors before labels or models are consulted."""

    if len(actor_ids) != EXPECTED_ACTORS or "" in actor_ids:
        raise ValueError("actor groups")
    ordered = sorted(actor_ids, key=lambda actor_id: hashlib.sha256((ROLE_SALT + actor_id).encode()).hexdigest())
    roles: dict[str, list[str]] = {}
    cursor = 0
    for role, count in ROLE_COUNTS.items():
        roles[role] = ordered[cursor : cursor + count]
        cursor += count
    if cursor != len(ordered):
        raise ValueError("actor groups")
    return roles


def _file(project_root: Path, path: Path, source_url: str, roles: list[str]) -> dict[str, Any]:
    relative = path.resolve().relative_to(project_root.resolve()).as_posix()
    digest = _sha256(path)
    return {
        "filename": relative,
        "size_bytes": path.stat().st_size,
        "source_url": source_url,
        "expected_sha256": digest,
        "observed_sha256": digest,
        "roles": roles,
    }


def build_registry(
    project_root: Path,
    repository_root: Path,
    pointer_index_path: Path,
    media_verification_path: Path,
    role_allocation_path: Path,
) -> dict[str, Any]:
    """Generate registry metadata after a successful full-media verification."""

    pointer = json.loads(pointer_index_path.read_text(encoding="utf-8"))
    media = json.loads(media_verification_path.read_text(encoding="utf-8"))
    if (
        not isinstance(pointer, Mapping)
        or pointer.get("commit") != EXPECTED_COMMIT
        or pointer.get("clip_count") != EXPECTED_CLIPS
        or not isinstance(pointer.get("records"), list)
        or not isinstance(media, Mapping)
        or media.get("schema") != "conflictbench.cremad-media-verification.v1"
        or media.get("pointer_index_sha256") != _sha256(pointer_index_path)
        or media.get("clip_count") != EXPECTED_CLIPS
        or media.get("media_count") != EXPECTED_CLIPS * 2
    ):
        raise ValueError("verified media")

    demographics_path = repository_root / "VideoDemographics.csv"
    with demographics_path.open(encoding="utf-8", newline="") as handle:
        actor_ids = {row.get("ActorID", "").strip() for row in csv.DictReader(handle)}
    roles = allocate_actor_roles(actor_ids)
    role_allocation = {
        "schema": "conflictbench.cremad-actor-role-allocation.v1",
        "repository": "https://github.com/CheyneyComputerScience/CREMA-D",
        "commit": EXPECTED_COMMIT,
        "group_field": "ActorID",
        "method": "SHA-256 rank over ActorID with fixed role counts; no labels or media read",
        "role_counts": ROLE_COUNTS,
        "role_groups": roles,
    }
    if role_allocation_path.exists():
        raise ValueError("role allocation exists")
    role_allocation_path.parent.mkdir(parents=True, exist_ok=True)
    role_allocation_path.write_text(json.dumps(role_allocation, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    commit_url = "https://github.com/CheyneyComputerScience/CREMA-D/blob/" + EXPECTED_COMMIT
    files = [
        _file(project_root, pointer_index_path, commit_url, ["release_index"]),
        _file(project_root, media_verification_path, commit_url, ["release_index"]),
        _file(project_root, repository_root / "LICENSE.txt", commit_url + "/LICENSE.txt", ["release_index"]),
        _file(project_root, repository_root / "processedResults" / "summaryTable.csv", commit_url + "/processedResults/summaryTable.csv", ["labels"]),
        _file(project_root, demographics_path, commit_url + "/VideoDemographics.csv", ["group_mapping"]),
        _file(project_root, role_allocation_path, commit_url + "/VideoDemographics.csv", ["role_allocation"]),
    ]
    for record in pointer["records"]:
        if not isinstance(record, Mapping):
            raise ValueError("pointer records")
        clip_id = record.get("clip_id")
        if not isinstance(clip_id, str):
            raise ValueError("pointer records")
        for modality, directory, suffix in (("audio", "AudioWAV", ".wav"), ("video", "VideoFlash", ".flv")):
            expected = record.get(modality)
            if not isinstance(expected, Mapping):
                raise ValueError("pointer records")
            path = repository_root / directory / f"{clip_id}{suffix}"
            files.append(
                {
                    "filename": path.resolve().relative_to(project_root.resolve()).as_posix(),
                    "size_bytes": expected["size_bytes"],
                    "source_url": "https://github.com/CheyneyComputerScience/CREMA-D",
                    "expected_sha256": expected["sha256"],
                    "observed_sha256": expected["sha256"],
                    "roles": ["data"],
                }
            )
    return {
        "schema": SCHEMA,
        "dataset": {"name": "CREMA-D", "release": EXPECTED_COMMIT},
        "terms": {"url": commit_url + "/LICENSE.txt", "reviewed_on": "2026-09-18", "research_use_permitted": True},
        "files": files,
        "labels": {
            "integrated": "MultiModalVote",
            "audio": "VoiceVote",
            "video": "FaceVote",
            "text": "fixed sentence context; excluded from emotion decision features",
            "file_role": "labels",
        },
        "group_field": "ActorID",
        "grouping": {"unit": "speaker", "provenance_url": commit_url + "/VideoDemographics.csv", "file_role": "group_mapping"},
        "role_allocation": {"method": role_allocation["method"], "file_role": "role_allocation"},
        "role_groups": roles,
    }
