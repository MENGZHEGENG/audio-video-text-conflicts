#!/usr/bin/env python3
"""Summarize immutable CREMA-D comprehensive resplits without pooling actors.

The six deterministic actor reallocations overlap.  This script therefore
reports their means and ranges as a split-robustness description, never as an
independent-population confidence interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence


SCHEMA = "conflictbench.cremad-comprehensive-evaluation.v1"
SUMMARY_SCHEMA = "conflictbench.cremad-comprehensive-summary.v1"
INTERPRETATION = "descriptive; actor reallocations overlap and are not independent population units"
METRICS = (
    "evaluation_request_rate",
    "policy_minus_random_matched_budget",
    "full_multimodal_minus_start",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    if record.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unexpected schema")
    if not isinstance(record.get("seed"), int):
        raise ValueError(f"{path}: missing integer seed")
    if not isinstance(record.get("encoder_grid"), dict) or not record["encoder_grid"]:
        raise ValueError(f"{path}: missing encoder grid")
    return record


def _point_values(point: dict[str, Any], path: Path, combo: str, budget: str) -> tuple[float, float, float]:
    bootstrap = point.get("actor_cluster_bootstrap")
    if not isinstance(bootstrap, dict):
        raise ValueError(f"{path}: {combo}/{budget} lacks actor bootstrap")
    try:
        request_rate = float(point["evaluation_request_rate"])
        margin = float(bootstrap["policy_minus_random_matched_budget"]["estimate"])
        headroom = float(bootstrap["full_multimodal_minus_start"]["estimate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid {combo}/{budget} metrics") from exc
    return request_rate, margin, headroom


def aggregate(paths: Sequence[Path]) -> dict[str, Any]:
    """Validate and summarize reports, retaining each budget's distinct seeds."""
    if not paths:
        raise ValueError("no comprehensive result records were supplied")
    loaded = [(Path(path), _load(Path(path))) for path in paths]
    combinations = sorted(loaded[0][1]["encoder_grid"])
    if not combinations:
        raise ValueError("empty encoder grid")
    points: dict[str, dict[str, list[tuple[int, float, float, float]]]] = {}
    sources: list[dict[str, Any]] = []
    for path, record in loaded:
        grid = record["encoder_grid"]
        if sorted(grid) != combinations:
            raise ValueError(f"{path}: encoder grid differs from the first record")
        budgets = sorted({budget for combo in combinations for budget in grid[combo].get("operating_points", {})}, key=float)
        if not budgets:
            raise ValueError(f"{path}: no operating points")
        for combo in combinations:
            if sorted(grid[combo].get("operating_points", {}), key=float) != budgets:
                raise ValueError(f"{path}: operating budgets differ across encoder pairs")
        sources.append({"filename": path.name, "seed": record["seed"], "sha256": _sha256(path), "budgets": budgets})
        for budget in budgets:
            bucket = points.setdefault(budget, {combo: [] for combo in combinations})
            for combo in combinations:
                request_rate, margin, headroom = _point_values(grid[combo]["operating_points"][budget], path, combo, budget)
                bucket[combo].append((record["seed"], request_rate, margin, headroom))

    by_budget: dict[str, Any] = {}
    for budget, by_combo in sorted(points.items(), key=lambda item: float(item[0])):
        by_budget[budget] = {}
        for combo, entries in by_combo.items():
            entries.sort(key=lambda item: item[0])
            seeds = [seed for seed, _, _, _ in entries]
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"duplicate seed at budget {budget}: {seeds}")
            rates = [rate for _, rate, _, _ in entries]
            margins = [margin for _, _, margin, _ in entries]
            headrooms = [headroom for _, _, _, headroom in entries]
            by_budget[budget][combo] = {
                "resplits": len(entries),
                "seeds": seeds,
                "mean_evaluation_request_rate": fmean(rates),
                "mean_policy_minus_random_matched_budget": fmean(margins),
                "minimum_policy_minus_random_matched_budget": min(margins),
                "maximum_policy_minus_random_matched_budget": max(margins),
                "mean_full_multimodal_minus_start": fmean(headrooms),
            }
    return {
        "schema": SUMMARY_SCHEMA,
        "resplit_interpretation": INTERPRETATION,
        "source_records": sorted(sources, key=lambda item: (float(item["budgets"][0]), item["seed"], item["filename"])),
        "budgets": by_budget,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("immutable output exists")
    report = aggregate(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": _sha256(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
