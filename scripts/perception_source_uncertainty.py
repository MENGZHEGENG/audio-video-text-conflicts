#!/usr/bin/env python3
"""Add descriptive uncertainty to a frozen Perception source-gate result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


class SourceUncertaintyError(ValueError):
    """Raised when the frozen source-gate record is internally inconsistent."""


def wilson_interval(successes: int, total: int) -> list[float]:
    """Return a two-sided 95% Wilson score interval."""

    if total <= 0 or not 0 <= successes <= total:
        raise SourceUncertaintyError(
            "successes and total do not define a binomial rate"
        )
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [center - half_width, center + half_width]


def paired_bootstrap_interval(
    aligned: Sequence[int],
    control: Sequence[int],
    *,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, list[float]]:
    """Return point difference and a paired-component percentile interval."""

    first = np.asarray(aligned, dtype=np.float64)
    second = np.asarray(control, dtype=np.float64)
    if first.ndim != 1 or first.shape != second.shape or len(first) < 2:
        raise SourceUncertaintyError(
            "paired inputs must have the same nontrivial length"
        )
    if replicates < 100:
        raise SourceUncertaintyError("at least 100 bootstrap replicates are required")
    differences = first - second
    indices = rng.integers(0, len(differences), size=(replicates, len(differences)))
    draws = differences[indices].mean(axis=1)
    interval = np.quantile(draws, [0.025, 0.975], method="linear")
    return float(differences.mean()), [float(interval[0]), float(interval[1])]


def _unique_component_outcomes(
    records: Sequence[Mapping[str, Any]],
    *,
    evaluation: str,
    condition: str,
    source_role: str,
    pair_role: str | None,
) -> dict[str, int]:
    outcomes: dict[str, int] = {}
    for record in records:
        if (
            record.get("partition") != "pilot_gate"
            or record.get("evaluation") != evaluation
            or record.get("condition") != condition
            or record.get("source_role") != source_role
            or (pair_role is not None and record.get("pair_role") != pair_role)
        ):
            continue
        component_id = str(record["component_id"])
        value = int(bool(record["is_correct"]))
        if component_id in outcomes and outcomes[component_id] != value:
            raise SourceUncertaintyError("duplicate component outcomes disagree")
        outcomes[component_id] = value
    return outcomes


def analyze(
    payload: Mapping[str, Any],
    *,
    input_sha256: str,
    seed: int = 20260918,
    replicates: int = 10_000,
) -> dict[str, Any]:
    """Compute pair-support Wilson intervals and paired control-gain intervals."""

    gate = payload.get("gate")
    records = payload.get("records")
    if not isinstance(gate, Mapping) or not isinstance(records, list):
        raise SourceUncertaintyError("source-gate result must contain gate and records")

    pair_support: dict[str, Any] = {}
    for condition, metrics in sorted(gate["metrics"].items()):
        pair_support[condition] = {}
        for pair_role, cell in sorted(metrics["by_pair_role"].items()):
            successes = int(cell["pair_support_count"])
            total = int(cell["pair_count"])
            pair_support[condition][pair_role] = {
                "successes": successes,
                "total": total,
                "rate": successes / total,
                "wilson_interval_95": wilson_interval(successes, total),
            }

    rng = np.random.default_rng(seed)
    margins: dict[str, Any] = {}
    aligned_cells = gate["control_metrics"]["aligned_over_control_margin"]
    for condition, condition_cells in sorted(aligned_cells.items()):
        margins[condition] = {}
        requested_cells: list[tuple[str, str | None, Mapping[str, Any]]] = [
            ("target", None, condition_cells["target"])
        ]
        requested_cells.extend(
            ("donor", pair_role, cell)
            for pair_role, cell in sorted(
                condition_cells["donor"]["by_pair_role"].items()
            )
        )
        for source_role, pair_role, cell in requested_cells:
            aligned_map = _unique_component_outcomes(
                records,
                evaluation="source_sufficiency",
                condition=condition,
                source_role=source_role,
                pair_role=pair_role,
            )
            control_type = str(cell["strongest_control_type"])
            control_condition = (
                "question_only" if control_type == "question_only" else condition
            )
            control_map = _unique_component_outcomes(
                records,
                evaluation=control_type,
                condition=control_condition,
                source_role=source_role,
                pair_role=pair_role,
            )
            if set(aligned_map) != set(control_map) or len(aligned_map) != 20:
                raise SourceUncertaintyError(
                    "control comparison must contain 20 paired components"
                )
            component_ids = sorted(aligned_map)
            point, interval = paired_bootstrap_interval(
                [aligned_map[item] for item in component_ids],
                [control_map[item] for item in component_ids],
                replicates=replicates,
                rng=rng,
            )
            if not math.isclose(point, float(cell["margin"]), abs_tol=1e-12):
                raise SourceUncertaintyError(
                    "recomputed margin does not match frozen gate"
                )
            key = source_role if pair_role is None else f"{source_role}.{pair_role}"
            margins[condition][key] = {
                "components": len(component_ids),
                "control_type": control_type,
                "point_difference": point,
                "paired_component_bootstrap_interval_95": interval,
            }

    return {
        "schema": "conflictbench.perception-source-uncertainty.v1",
        "input_sha256": input_sha256,
        "bootstrap": {
            "unit": "paired_decision_component",
            "replicates": replicates,
            "seed": seed,
            "interval": "percentile_95",
        },
        "pair_support": pair_support,
        "aligned_over_control_margin": margins,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()

    raw = args.input.read_bytes()
    payload = json.loads(raw)
    report = analyze(
        payload,
        input_sha256=hashlib.sha256(raw).hexdigest(),
        seed=args.seed,
        replicates=args.replicates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
