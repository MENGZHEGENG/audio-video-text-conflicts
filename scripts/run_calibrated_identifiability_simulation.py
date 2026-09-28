#!/usr/bin/env python3
"""Evaluate calibration-derived acquisition and abstention on disjoint simulations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conflictbench.identifiability import (  # noqa: E402
    calibrate_candidate_values,
    calibrated_risk_aware_actions,
    candidate_prior_actions,
    cue_policy_actions,
    evaluate_gain,
    evaluate_source_recovery,
    generate_controlled_value_simulation,
    oracle_actions,
    random_actions,
    risk_aware_actions,
)


def _cue_strengths(value: str) -> list[float]:
    strengths = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not strengths or any(not 0.0 <= item <= 1.0 for item in strengths) or len(set(strengths)) != len(strengths):
        raise argparse.ArgumentTypeError("cue strengths must be distinct values between 0 and 1")
    return strengths


def _candidate_costs(value: str) -> tuple[float, float]:
    costs = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(costs) != 2 or any(cost < 0 for cost in costs):
        raise argparse.ArgumentTypeError("candidate costs must be two non-negative values")
    return costs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--calibration-seed", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--calibration-samples", type=int, required=True)
    parser.add_argument("--oracle-headroom", type=float, required=True)
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--candidate-costs", type=_candidate_costs, required=True)
    parser.add_argument("--cue-strengths", type=_cue_strengths, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    calibration_records: dict[str, dict[str, object]] = {}
    results: dict[str, dict[str, object]] = {}
    for index, strength in enumerate(args.cue_strengths):
        calibration = generate_controlled_value_simulation(
            seed=args.calibration_seed + index,
            samples=args.calibration_samples,
            oracle_headroom=args.oracle_headroom,
            cue_strength=strength,
            candidate_costs=args.candidate_costs,
        )
        simulation = generate_controlled_value_simulation(
            seed=args.seed + index,
            samples=args.samples,
            oracle_headroom=args.oracle_headroom,
            cue_strength=strength,
            candidate_costs=args.candidate_costs,
        )
        values = calibrate_candidate_values(calibration)
        actions = calibrated_risk_aware_actions(simulation, values)
        action_seed = args.seed + 10_000 + index
        key = strength.__format__(".1f")
        calibration_records[key] = {
            "lower_expected_gain": values.lower_expected_gain.tolist(),
            "samples_per_candidate": values.samples_per_candidate.tolist(),
        }
        results[key] = {
            "candidate_prior": evaluate_gain(
                simulation, candidate_prior_actions(simulation, budget=args.budget, seed=action_seed)
            ),
            "random": evaluate_gain(simulation, random_actions(simulation, budget=args.budget, seed=action_seed)),
            "cue_policy": evaluate_gain(
                simulation, cue_policy_actions(simulation, budget=args.budget, seed=action_seed)
            ),
            "oracle": evaluate_gain(simulation, oracle_actions(simulation, budget=args.budget, seed=action_seed)),
            "risk_aware": evaluate_gain(simulation, risk_aware_actions(simulation)),
            "calibrated_risk_aware": evaluate_gain(simulation, actions),
            "source_recovery": evaluate_source_recovery(simulation, actions),
        }
    record = {
        "schema": "conflictbench.calibrated-identifiability-control.v1",
        "seed": args.seed,
        "calibration_seed": args.calibration_seed,
        "contract": {
            "samples": args.samples,
            "calibration_samples": args.calibration_samples,
            "oracle_headroom": args.oracle_headroom,
            "budget": args.budget,
            "candidate_costs": list(args.candidate_costs),
        },
        "calibration": calibration_records,
        "results": results,
    }
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
