#!/usr/bin/env python3
"""Validate and aggregate the complete matched-selector seed set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conflictbench.matched_selector_value import aggregate_selector_runs, load_seeds  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-plan", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=92309)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()

    seeds = load_seeds(json.loads(args.seed_plan.read_text(encoding="utf-8")))
    records = []
    for seed in seeds:
        path = args.input_dir / f"seed_{seed}.json"
        records.append(json.loads(path.read_text(encoding="utf-8")))
    result = aggregate_selector_runs(
        records,
        expected_seeds=seeds,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"seeds": result["seed_count"], "output": str(output), "schema": result["schema"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
