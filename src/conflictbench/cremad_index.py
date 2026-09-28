"""Leakage-safe CREMA-D paired-media index construction."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from conflictbench.cremad_preflight import parse_vote_set


SCHEMA = "conflictbench.cremad-paired-index.v1"


def build_index(repository_root: Path, role_allocation_path: Path) -> dict[str, Any]:
    """Pair verified media by clip id without exposing filename tokens as inputs."""

    allocation = json.loads(role_allocation_path.read_text(encoding="utf-8"))
    if not isinstance(allocation, Mapping) or not isinstance(allocation.get("role_groups"), Mapping):
        raise ValueError("role allocation")
    roles = {
        actor: role
        for role, actors in allocation["role_groups"].items()
        if isinstance(actors, list)
        for actor in actors
        if isinstance(actor, str)
    }
    with (repository_root / "processedResults" / "summaryTable.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        clip_id = row.get("FileName", "").strip()
        actor_id = clip_id[:4]
        if not clip_id or clip_id in seen or actor_id not in roles:
            raise ValueError("paired labels")
        seen.add(clip_id)
        audio = repository_root / "AudioWAV" / f"{clip_id}.wav"
        video = repository_root / "VideoFlash" / f"{clip_id}.flv"
        if not audio.is_file() or not video.is_file():
            raise ValueError("paired media")
        integrated = parse_vote_set(row["MultiModalVote"])
        records.append({
            "pair_id": clip_id,
            "actor_id": actor_id,
            "role": roles[actor_id],
            "audio_path": audio.as_posix(),
            "video_path": video.as_posix(),
            "voice_vote": list(parse_vote_set(row["VoiceVote"])),
            "face_vote": list(parse_vote_set(row["FaceVote"])),
            "integrated_vote": list(integrated),
            "integrated_singleton": len(integrated) == 1,
        })
    return {
        "schema": SCHEMA,
        "input_policy": "pair_id is for pairing only; no filename token is an encoder, split, or decision feature",
        "text_policy": "fixed script context only; excluded from primary emotion decision features",
        "records": records,
        "summary": {"record_count": len(records), "singleton_integrated_count": sum(record["integrated_singleton"] for record in records)},
    }
