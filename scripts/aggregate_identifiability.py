#!/usr/bin/env python3
"""Aggregate a complete, write-once controlled-identifiability seed family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "conflictbench.identifiability-control.v1"


def _load_record(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != SCHEMA:
        raise ValueError(f"{path.name} has unsupported schema")
    if not isinstance(record.get("seed"), int):
        raise ValueError(f"{path.name} has no integer seed")
    if not isinstance(record.get("contract"), dict) or not isinstance(record.get("results"), dict):
        raise ValueError(f"{path.name} is missing contract or results")
    return record


def _mean(records: list[dict[str, Any]], strength: str, policy: str) -> float:
    return float(np.mean([record["results"][strength][policy]["mean_gain"] for record in records]))


def aggregate_results(directory: Path, *, expected_seeds: list[int]) -> dict[str, Any]:
    """Aggregate exactly the requested result files and reject partial families."""

    if len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("expected seeds must be unique")
    paths = sorted(directory.glob("seed-*.json"))
    records = [_load_record(path) for path in paths]
    by_seed = {record["seed"]: record for record in records}
    if len(by_seed) != len(records):
        raise ValueError("duplicate seed outputs")
    missing = sorted(set(expected_seeds) - set(by_seed))
    unexpected = sorted(set(by_seed) - set(expected_seeds))
    if missing:
        raise ValueError(f"missing expected seed outputs: {missing}")
    if unexpected:
        raise ValueError(f"unexpected seed outputs: {unexpected}")
    ordered = [by_seed[seed] for seed in expected_seeds]
    contracts = {json.dumps(record["contract"], sort_keys=True) for record in ordered}
    if len(contracts) != 1:
        raise ValueError("seed outputs use different contracts")
    strengths = set(ordered[0]["results"])
    if not strengths:
        raise ValueError("seed outputs contain no cue strengths")
    required_policies = {"candidate_prior", "random", "cue_policy", "oracle", "risk_aware"}
    for record in ordered:
        if set(record["results"]) != strengths:
            raise ValueError("seed outputs use different cue strength sets")
        for strength in strengths:
            policies = set(record["results"][strength])
            if policies != required_policies:
                raise ValueError(f"seed output policies differ at cue strength {strength}")
    summaries: dict[str, dict[str, float]] = {}
    for strength in sorted(strengths, key=float):
        random_values = np.asarray(
            [record["results"][strength]["random"]["mean_gain"] for record in ordered], dtype=np.float64
        )
        cue_values = np.asarray(
            [record["results"][strength]["cue_policy"]["mean_gain"] for record in ordered], dtype=np.float64
        )
        summaries[strength] = {
            "candidate_prior_mean": _mean(ordered, strength, "candidate_prior"),
            "random_mean": float(random_values.mean()),
            "cue_policy_mean": float(cue_values.mean()),
            "oracle_mean": _mean(ordered, strength, "oracle"),
            "risk_aware_mean": _mean(ordered, strength, "risk_aware"),
            "cue_minus_random_mean": float((cue_values - random_values).mean()),
            "cue_minus_random_sample_sd": float((cue_values - random_values).std(ddof=1))
            if len(ordered) > 1
            else 0.0,
        }
    return {
        "schema": "conflictbench.identifiability-aggregate.v1",
        "contract": ordered[0]["contract"],
        "expected_seeds": expected_seeds,
        "seed_count": len(ordered),
        "results": summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--expected-seeds", required=True, help="comma-separated integer seeds")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing aggregate: {output}")
    seeds = [int(item.strip()) for item in args.expected_seeds.split(",") if item.strip()]
    record = aggregate_results(args.input_dir.resolve(), expected_seeds=seeds)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
