#!/usr/bin/env python3
"""Validate and aggregate the complete temporal shortcut diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from typing import Any

SCHEMA = "conflictbench.temporal_shortcut_check.v2"
CONDITIONS = (
    "identity",
    "shared_permutation",
    "independent_channel_permutation",
    "per_example_circular_shift",
)
METHODS = ("temporal_template", "temporal_linear", "temporal_fusion", "temporal_gated")
METRICS = (
    "decision_accuracy",
    "nonambiguous_accuracy",
    "coverage",
    "selective_accuracy",
    "utility",
)


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _load_expected(config_path: pathlib.Path) -> dict[tuple[str, int], dict[str, Any]]:
    spec = json.loads(config_path.read_text(encoding="utf-8"))
    if spec.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA} configuration")
    cells = spec.get("cells")
    seeds = spec.get("seeds")
    if not isinstance(cells, list) or not isinstance(seeds, list):
        raise TypeError("configuration must define list cells and seeds")
    expected: dict[tuple[str, int], dict[str, Any]] = {}
    for cell in cells:
        cell_id = cell.get("id") if isinstance(cell, dict) else None
        if not isinstance(cell_id, str):
            raise TypeError("each cell needs a string id")
        for seed in seeds:
            key = (cell_id, int(seed))
            if key in expected:
                raise ValueError(f"duplicate expected run: {key}")
            expected[key] = cell
    return expected


def _validated(
    path: pathlib.Path, expected: dict[tuple[str, int], dict[str, Any]]
) -> tuple[tuple[str, int], dict[str, Any]]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        record.get("schema") != SCHEMA
        or record.get("validation", {}).get("ok") is not True
    ):
        raise ValueError("schema or validation failed")
    key = (str(record.get("config", {}).get("id")), int(record.get("seed")))
    if key not in expected:
        raise ValueError(f"unexpected cell/seed: {key}")
    if record.get("config") != expected[key]:
        raise ValueError("resolved cell configuration mismatch")
    if record.get("environment", {}).get("used_device") != "cuda":
        raise ValueError("record did not use CUDA")
    conditions = record.get("conditions")
    if not isinstance(conditions, dict) or set(conditions) != set(CONDITIONS):
        raise ValueError("condition set mismatch")
    for condition in CONDITIONS:
        splits = conditions[condition].get("splits", {})
        if set(splits) != {"seen", "unseen"}:
            raise ValueError(f"{condition}: split set mismatch")
        for split in splits.values():
            methods = split.get("methods", {})
            if set(methods) != set(METHODS):
                raise ValueError(f"{condition}: method set mismatch")
            for method in methods.values():
                if method.get("status") != "verified":
                    raise ValueError(f"{condition}: unverified method")
                for metric in METRICS:
                    value = method.get("metrics", {}).get(metric)
                    if (
                        value is None
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                    ):
                        raise ValueError(f"{condition}: invalid {metric}")
    return key, record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    expected = _load_expected(pathlib.Path(args.config))
    records: dict[tuple[str, int], dict[str, Any]] = {}
    rejected: list[dict[str, str]] = []
    for path in sorted(
        pathlib.Path(args.input_dir).glob("temporal_shortcut_falsification-gpu-*.json")
    ):
        try:
            key, record = _validated(path, expected)
            if key in records:
                raise ValueError(f"duplicate cell/seed: {key}")
            records[key] = record
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            rejected.append(
                {"file": str(path), "reason": f"{type(exc).__name__}: {exc}"}
            )
    missing = sorted(set(expected) - set(records))
    if missing or rejected:
        raise SystemExit(
            f"diagnostic incomplete: missing={len(missing)}, rejected={len(rejected)}, first_missing={missing[:3]}"
        )

    summary: dict[str, Any] = {
        "schema": "conflictbench.temporal_shortcut_aggregate.v1",
        "runs": len(records),
        "conditions": {},
    }
    for condition in CONDITIONS:
        condition_summary: dict[str, Any] = {}
        for cell_id, _seed in expected:
            condition_summary.setdefault(cell_id, {"splits": {}})
        for cell_id in condition_summary:
            cell_records = [
                record
                for (key_cell, _), record in records.items()
                if key_cell == cell_id
            ]
            for split_name in ("seen", "unseen"):
                methods: dict[str, Any] = {}
                for method_name in METHODS:
                    methods[method_name] = {
                        metric: _summary(
                            [
                                float(
                                    record["conditions"][condition]["splits"][
                                        split_name
                                    ]["methods"][method_name]["metrics"][metric]
                                )
                                for record in cell_records
                            ]
                        )
                        for metric in METRICS
                    }
                condition_summary[cell_id]["splits"][split_name] = {"methods": methods}
        summary["conditions"][condition] = condition_summary
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(output), "runs": len(records), "rejected": 0}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
