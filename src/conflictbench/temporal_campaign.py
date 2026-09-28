"""Campaign-specification expansion for temporal factorial experiments.

The campaign specification is intentionally small and explicit.  A job is the
Cartesian product of a named cell and a seed set; the resolved configuration
is embedded in the run output so aggregates can never silently mix cells.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import pathlib
import re
from typing import Any

from .runner import _validate_config


TEMPORAL_CAMPAIGN_SCHEMA = "conflictbench.temporal_campaign.v2"
_CELL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class TemporalCampaignJob:
    """One deterministic cell/seed job in a campaign."""

    index: int
    cell_index: int
    cell_id: str
    seed: int
    seed_group: str
    config: dict[str, Any]


def load_campaign_spec(path: str | pathlib.Path) -> dict[str, Any]:
    """Load and validate a temporal campaign specification."""

    spec_path = pathlib.Path(path)
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    validate_campaign_spec(data)
    return data


def _seeds(spec: dict[str, Any], seed_group: str) -> tuple[int, ...]:
    if seed_group not in {"primary", "replication"}:
        raise ValueError("seed_group must be 'primary' or 'replication'")
    field = f"{seed_group}_seeds"
    values = spec[field]
    return tuple(int(value) for value in values)


def validate_campaign_spec(spec: dict[str, Any]) -> None:
    """Fail closed on malformed cells, duplicate seeds, or unsafe overrides."""

    if not isinstance(spec, dict) or spec.get("schema") != TEMPORAL_CAMPAIGN_SCHEMA:
        raise ValueError(f"campaign specification schema must be {TEMPORAL_CAMPAIGN_SCHEMA}")
    if not isinstance(spec.get("name"), str) or not spec["name"].strip():
        raise ValueError("campaign specification name must be a non-empty string")
    base = spec.get("base_config")
    if not isinstance(base, dict):
        raise ValueError("base_config must be an object")
    base = dict(base)
    base.setdefault("dataset", "synthetic_temporal")
    if str(base.get("dataset", "")).lower() != "synthetic_temporal":
        raise ValueError("temporal campaign base_config must use synthetic_temporal")
    if "seed" in base:
        raise ValueError("base_config must not set seed; seeds come from the campaign specification")
    _validate_config({**base, "seed": 0})

    replication_overrides = spec.get("replication_overrides", {})
    if not isinstance(replication_overrides, dict):
        raise ValueError("replication_overrides must be an object")

    cells = spec.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("cells must be a non-empty list")
    seen_ids: set[str] = set()
    forbidden = {"seed", "dataset", "name", "campaign_cell", "campaign_index", "campaign_name"}
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("each cell must be an object")
        cell_id = cell.get("id")
        if not isinstance(cell_id, str) or not _CELL_ID.fullmatch(cell_id):
            raise ValueError("cell id must contain only letters, digits, '.', '_' or '-'")
        if cell_id in seen_ids:
            raise ValueError(f"duplicate cell id: {cell_id}")
        seen_ids.add(cell_id)
        if not isinstance(cell.get("hypothesis"), str) or not cell["hypothesis"].strip():
            raise ValueError(f"cell {cell_id} needs a hypothesis")
        overrides = cell.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"cell {cell_id} overrides must be an object")
        if forbidden.intersection(overrides):
            raise ValueError(f"cell {cell_id} attempts to override protected fields")
        for seed_group in ("primary", "replication"):
            group_overrides = (
                replication_overrides if seed_group == "replication" else {}
            )
            if forbidden.intersection(group_overrides):
                raise ValueError(
                    "replication_overrides attempts to override protected fields"
                )
            candidate = {**base, **overrides, **group_overrides, "seed": 0}
            _validate_config(candidate)
            if candidate.get("dataset") != "synthetic_temporal":
                raise ValueError(f"cell {cell_id} changed dataset")

    primary = _seeds(spec, "primary")
    replication = _seeds(spec, "replication")
    if not primary or not replication:
        raise ValueError("primary_seeds and replication_seeds must both be non-empty")
    if len(set(primary)) != len(primary) or len(set(replication)) != len(replication):
        raise ValueError("seeds must be unique within each seed group")
    if set(primary).intersection(replication):
        raise ValueError("primary and replication seed groups must be disjoint")
    if any(seed < 0 for seed in (*primary, *replication)):
        raise ValueError("seeds must be non-negative")


def expand_campaign_spec(
    spec: dict[str, Any], seed_group: str
) -> tuple[TemporalCampaignJob, ...]:
    """Expand cells and the requested seed set in stable row-major order."""

    validate_campaign_spec(spec)
    seeds = _seeds(spec, seed_group)
    base = dict(spec["base_config"])
    base.setdefault("dataset", "synthetic_temporal")
    replication_overrides = dict(spec.get("replication_overrides", {}))
    jobs: list[TemporalCampaignJob] = []
    for cell_index, cell in enumerate(spec["cells"]):
        cell_id = str(cell["id"])
        for seed in seeds:
            index = len(jobs)
            config = {
                **base,
                **dict(cell.get("overrides", {})),
                **(replication_overrides if seed_group == "replication" else {}),
                "dataset": "synthetic_temporal",
                "seed": int(seed),
                "campaign_name": str(spec["name"]),
                "campaign_cell": cell_id,
                "campaign_cell_index": int(cell_index),
                "campaign_run_index": int(index),
                "campaign_hypothesis": str(cell["hypothesis"]),
                "campaign_seed_group": seed_group,
            }
            _validate_config(config)
            jobs.append(
                TemporalCampaignJob(
                    index=index,
                    cell_index=cell_index,
                    cell_id=cell_id,
                    seed=int(seed),
                    seed_group=seed_group,
                    config=config,
                )
            )
    return tuple(jobs)


def resolve_campaign_job(
    spec: dict[str, Any], seed_group: str, index: int
) -> TemporalCampaignJob:
    """Resolve one stable array index and give a useful bounds error."""

    jobs = expand_campaign_spec(spec, seed_group)
    if index < 0 or index >= len(jobs):
        raise IndexError(
            f"{seed_group} run index {index} is outside 0..{len(jobs) - 1}"
        )
    return jobs[index]
