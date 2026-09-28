#!/usr/bin/env python3
"""Re-score immutable scalar run summaries under three abstention contracts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np


CONTRACTS = ("published", "symmetric_nonidentifiable", "ordinary_selective")
AMBIGUITY = "ambiguity"
SECOND_NONIDENTIFIABLE = "mixed"


class ContractError(ValueError):
    """Raised when a stored scalar record cannot be re-scored safely."""


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} is not numeric")
    value = float(value)
    if not np.isfinite(value):
        raise ContractError(f"{label} is not finite")
    return value


def _rescore_method(method: dict, contract: str, abstain_cost: float) -> float:
    mechanisms = method.get("by_mechanism")
    if not isinstance(mechanisms, dict) or not mechanisms:
        raise ContractError("method lacks per-mechanism summaries")
    total = 0
    weighted_utility = 0.0
    for mechanism, record in mechanisms.items():
        if not isinstance(record, dict):
            raise ContractError(f"{mechanism} summary is invalid")
        count = record.get("n")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ContractError(f"{mechanism}.n is invalid")
        utility = _number(record.get("utility"), f"{mechanism}.utility")
        abstain_rate = _number(record.get("abstain_rate"), f"{mechanism}.abstain_rate")
        if not 0.0 <= abstain_rate <= 1.0:
            raise ContractError(f"{mechanism}.abstain_rate is out of range")
        change = 0.0
        if contract == "symmetric_nonidentifiable" and mechanism == SECOND_NONIDENTIFIABLE:
            change = (2.0 + abstain_cost) * abstain_rate
        elif contract == "ordinary_selective" and mechanism == AMBIGUITY:
            change = -(2.0 + abstain_cost) * abstain_rate
        elif contract not in CONTRACTS:
            raise ContractError(f"unknown contract: {contract}")
        total += count
        weighted_utility += count * (utility + change)
    return weighted_utility / total


def _load(paths: Sequence[Path]) -> dict[int, dict]:
    records: dict[int, dict] = {}
    for path in sorted(paths):
        record = json.loads(path.read_text(encoding="utf-8"))
        seed = record.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed in records:
            raise ContractError(f"invalid or duplicate seed in {path}")
        if record.get("validation", {}).get("ok") is not True:
            raise ContractError(f"unverified record: {path}")
        records[seed] = record
    if not records:
        raise ContractError("run set is empty")
    return records


def analyze(paths: Sequence[Path], abstain_cost: float = 0.2) -> dict[str, object]:
    records = _load(paths)
    methods = sorted(records[next(iter(records))]["splits"]["unseen"]["methods"])
    if any(sorted(record["splits"]["unseen"]["methods"]) != methods for record in records.values()):
        raise ContractError("method sets differ across seeds")
    per_contract: dict[str, object] = {}
    for contract in CONTRACTS:
        summaries: dict[str, object] = {}
        for method in methods:
            values = np.asarray(
                [_rescore_method(records[seed]["splits"]["unseen"]["methods"][method], contract, abstain_cost) for seed in sorted(records)],
                dtype=float,
            )
            summaries[method] = {
                "mean_utility": float(values.mean()),
                "sample_standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "seed_count": int(len(values)),
            }
        per_contract[contract] = summaries
    return {
        "schema": "conflictbench.abstention-contract-reanalysis.v1",
        "contracts": {
            "published": "abstention is correct only for ambiguity; non-ambiguity abstention costs 0.2",
            "symmetric_nonidentifiable": "also treat mixed-mechanism abstention as correct with no abstention cost",
            "ordinary_selective": "treat abstention as incorrect with a 0.2 abstention cost for every mechanism",
        },
        "abstain_cost": abstain_cost,
        "seeds": sorted(records),
        "unseen_utility": per_contract,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--abstain-cost", type=float, default=0.2)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"immutable output exists: {args.output}")
    report = analyze(args.inputs, args.abstain_cost)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "seeds": len(report["seeds"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
