#!/usr/bin/env python3
"""Paired actor-level contrast of CREMA-D fusion gains across two encoders."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "conflictbench.cremad-evaluation-predictions.v1":
        raise ValueError(f"unexpected prediction schema: {path}")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"empty prediction records: {path}")
    keyed = {row["pair_id"]: row for row in records}
    if len(keyed) != len(records):
        raise ValueError(f"duplicate pair_id: {path}")
    return keyed


def compare(first: dict[str, dict], second: dict[str, dict], *, draws: int, seed: int) -> dict:
    if set(first) != set(second) or draws < 1:
        raise ValueError("prediction cohorts or bootstrap draws differ")
    pair_ids = sorted(first)
    actors = []
    first_gain = []
    second_gain = []
    for pair_id in pair_ids:
        left, right = first[pair_id], second[pair_id]
        if left["actor_id"] != right["actor_id"] or left["label"] != right["label"]:
            raise ValueError(f"unpaired actor or label: {pair_id}")
        actors.append(left["actor_id"])
        first_gain.append(
            float(left["fused_prediction"] == left["label"])
            - float(left["start_prediction"] == left["label"])
        )
        second_gain.append(
            float(right["fused_prediction"] == right["label"])
            - float(right["start_prediction"] == right["label"])
        )
    actors = np.asarray(actors)
    first_gain = np.asarray(first_gain)
    second_gain = np.asarray(second_gain)
    unique = np.unique(actors)
    if len(unique) < 2:
        raise ValueError("need at least two evaluation actors")
    positions = {actor: np.flatnonzero(actors == actor) for actor in unique}
    difference = first_gain - second_gain
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([positions[actor] for actor in chosen])
        samples[draw] = difference[indices].mean()
    return {
        "schema": "conflictbench.cremad-encoder-pair-interaction.v1",
        "records": len(pair_ids),
        "evaluation_actors": len(unique),
        "first_fusion_gain": float(first_gain.mean()),
        "second_fusion_gain": float(second_gain.mean()),
        "paired_gain_difference": float(difference.mean()),
        "paired_gain_difference_bootstrap_95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "independent_unit": "actor",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, type=Path)
    parser.add_argument("--second", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20270923)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("immutable output exists")
    result = compare(_load(args.first), _load(args.second), draws=args.draws, seed=args.seed)
    result["inputs"] = [
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in (args.first, args.second)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
