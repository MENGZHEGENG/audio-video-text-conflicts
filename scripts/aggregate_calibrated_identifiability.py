#!/usr/bin/env python3
"""Aggregate a complete calibrated-identifiability seed family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "conflictbench.calibrated-identifiability-control.v1"


def _load_record(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != SCHEMA or not isinstance(record.get("seed"), int):
        raise ValueError(f"{path.name} has unsupported schema or seed")
    if not isinstance(record.get("contract"), dict) or not isinstance(record.get("results"), dict):
        raise ValueError(f"{path.name} is missing contract or results")
    return record


def _source_counts(result: dict[str, Any], candidate: str) -> tuple[int, int]:
    recovery = result.get("source_recovery")
    if not isinstance(recovery, dict) or not isinstance(recovery.get(candidate), dict):
        raise ValueError("source recovery is missing a candidate")
    source = recovery[candidate]
    queried = source.get("query_count")
    helpful = source.get("post_query_helpful_count")
    if not isinstance(queried, int) or not isinstance(helpful, int) or not 0 <= helpful <= queried:
        raise ValueError("source recovery counts are invalid")
    return queried, helpful


def aggregate_results(directory: Path, *, expected_seeds: list[int]) -> dict[str, Any]:
    """Require the exact seed family and preserve per-source post-query recovery."""

    if len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("expected seeds must be unique")
    records = [_load_record(path) for path in sorted(directory.glob("seed-*.json"))]
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
    if len({json.dumps(record["contract"], sort_keys=True) for record in ordered}) != 1:
        raise ValueError("seed outputs use different contracts")
    strengths = set(ordered[0]["results"])
    if not strengths:
        raise ValueError("seed outputs contain no cue strengths")
    summaries: dict[str, dict[str, Any]] = {}
    for strength in sorted(strengths, key=float):
        values: list[float] = []
        query_rates: list[float] = []
        pooled: dict[str, list[int]] = {"0": [0, 0], "1": [0, 0]}
        for record in ordered:
            results = record["results"]
            if set(results) != strengths:
                raise ValueError("seed outputs use different cue strength sets")
            result = results[strength]
            calibrated = result.get("calibrated_risk_aware")
            if not isinstance(calibrated, dict):
                raise ValueError("calibrated policy is missing")
            value = calibrated.get("mean_gain")
            rate = calibrated.get("query_rate")
            if not isinstance(value, (int, float)) or not isinstance(rate, (int, float)) or not 0 <= rate <= 1:
                raise ValueError("calibrated policy summary is invalid")
            values.append(float(value))
            query_rates.append(float(rate))
            for candidate in pooled:
                queried, helpful = _source_counts(result, candidate)
                pooled[candidate][0] += queried
                pooled[candidate][1] += helpful
        source_recovery = {
            candidate: {
                "query_count": counts[0],
                "post_query_helpful_count": counts[1],
                "post_query_helpful_rate": counts[1] / counts[0] if counts[0] else None,
            }
            for candidate, counts in pooled.items()
        }
        summaries[strength] = {
            "calibrated_risk_aware_mean": float(np.mean(values)),
            "calibrated_query_rate_mean": float(np.mean(query_rates)),
            "source_recovery": source_recovery,
        }
    return {
        "schema": "conflictbench.calibrated-identifiability-aggregate.v1",
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
    aggregate = aggregate_results(args.input_dir.resolve(), expected_seeds=seeds)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
