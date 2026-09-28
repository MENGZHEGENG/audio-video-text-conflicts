#!/usr/bin/env python3
"""Run one controlled pre-query identifiability sweep and write a JSON record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conflictbench.identifiability import (  # noqa: E402
    candidate_prior_actions,
    cue_policy_actions,
    evaluate_gain,
    generate_controlled_value_simulation,
    oracle_actions,
    random_actions,
    risk_aware_actions,
)


def _cue_strengths(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("provide at least one cue strength")
    if any(not 0.0 <= item <= 1.0 for item in values):
        raise argparse.ArgumentTypeError("cue strengths must be between 0 and 1")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("cue strengths must be distinct")
    return values


def _candidate_costs(value: str) -> tuple[float, float]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(values) != 2 or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("candidate costs must be two non-negative values")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--oracle-headroom", type=float, required=True)
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--candidate-costs", type=_candidate_costs, default=(0.0, 0.0))
    parser.add_argument("--cue-strengths", type=_cue_strengths, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, object]] = {}
    for index, strength in enumerate(args.cue_strengths):
        simulation = generate_controlled_value_simulation(
            seed=args.seed,
            samples=args.samples,
            oracle_headroom=args.oracle_headroom,
            cue_strength=strength,
            candidate_costs=args.candidate_costs,
        )
        action_seed = args.seed + 10_000 + index
        results[strength.__format__(".1f")] = {
            "candidate_prior": evaluate_gain(
                simulation, candidate_prior_actions(simulation, budget=args.budget, seed=action_seed)
            ),
            "random": evaluate_gain(simulation, random_actions(simulation, budget=args.budget, seed=action_seed)),
            "cue_policy": evaluate_gain(
                simulation, cue_policy_actions(simulation, budget=args.budget, seed=action_seed)
            ),
            "oracle": evaluate_gain(simulation, oracle_actions(simulation, budget=args.budget, seed=action_seed)),
            "risk_aware": evaluate_gain(simulation, risk_aware_actions(simulation)),
        }
    record = {
        "schema": "conflictbench.identifiability-control.v1",
        "seed": args.seed,
        "contract": {
            "samples": args.samples,
            "oracle_headroom": args.oracle_headroom,
            "budget": args.budget,
            "candidate_costs": list(args.candidate_costs),
        },
        "results": results,
    }
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
