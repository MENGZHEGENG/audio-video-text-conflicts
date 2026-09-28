#!/usr/bin/env python3
"""Validate and aggregate a complete temporal factorial campaign."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import statistics
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from conflictbench.temporal_campaign import expand_campaign_spec, load_campaign_spec  # noqa: E402


def _load_validator() -> Any:
    path = ROOT / "scripts" / "verify_run.py"
    spec = importlib.util.spec_from_file_location("conflictbench_verify_run", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary(values: list[float], undefined: int = 0) -> dict[str, Any]:
    result: dict[str, Any] = {"n": len(values)}
    if values:
        result["mean"] = statistics.fmean(values)
        result["std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        result["ci95"] = 1.96 * result["std"] / math.sqrt(len(values))
    if undefined:
        result["undefined"] = int(undefined)
    return result


def _collect_metric(records: list[dict[str, Any]], location: tuple[str, ...]) -> dict[str, Any]:
    values: list[float] = []
    undefined = 0
    for record in records:
        value: Any = record
        for key in location:
            value = value.get(key) if isinstance(value, dict) else None
        if value is None:
            undefined += 1
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            values.append(float(value))
        else:
            raise ValueError(f"non-numeric metric at {'/'.join(location)}")
    return _summary(values, undefined)


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"runs": len(records), "splits": {}}
    if not records:
        return output
    for split_name in ("seen", "unseen"):
        method_names = sorted(
            {
                method
                for record in records
                for method in record["splits"][split_name]["methods"]
                if record["splits"][split_name]["methods"][method].get("status") == "verified"
            }
        )
        split_output: dict[str, Any] = {"methods": {}}
        for method_name in method_names:
            method_records = [
                record["splits"][split_name]["methods"][method_name]
                for record in records
                if record["splits"][split_name]["methods"].get(method_name, {}).get("status") == "verified"
            ]
            metric_names = sorted({metric for row in method_records for metric in row.get("metrics", {})})
            metrics = {
                metric: _collect_metric(method_records, ("metrics", metric)) for metric in metric_names
            }
            mechanisms = sorted(
                {
                    mechanism
                    for row in method_records
                    for mechanism in row.get("by_mechanism", {})
                }
            )
            by_mechanism: dict[str, Any] = {}
            for mechanism in mechanisms:
                mechanism_records = [
                    row["by_mechanism"][mechanism]
                    for row in method_records
                    if mechanism in row.get("by_mechanism", {})
                ]
                by_mechanism[mechanism] = {
                    metric: _collect_metric(mechanism_records, (metric,))
                    for metric in sorted(
                        {metric for row in mechanism_records for metric in row if metric != "n"}
                    )
                }
            split_output["methods"][method_name] = {
                "runs": len(method_records),
                "metrics": metrics,
                "by_mechanism": by_mechanism,
            }
        output["splits"][split_name] = split_output

    comparisons: dict[str, Any] = {}
    for split_name in ("seen", "unseen"):
        for candidate in ("temporal_fusion", "temporal_gated", "active_diagnostic", "majority"):
            for baseline in ("majority", "active_diagnostic"):
                if candidate == baseline:
                    continue
                deltas: dict[str, list[float]] = {"decision_accuracy": [], "utility": []}
                for record in records:
                    methods = record["splits"][split_name]["methods"]
                    left = methods.get(candidate, {})
                    right = methods.get(baseline, {})
                    if left.get("status") != "verified" or right.get("status") != "verified":
                        continue
                    for metric in deltas:
                        a, b = left.get("metrics", {}).get(metric), right.get("metrics", {}).get(metric)
                        if a is not None and b is not None:
                            deltas[metric].append(float(a) - float(b))
                if any(deltas.values()):
                    comparisons[f"{split_name}:{candidate}-vs-{baseline}"] = {
                        metric: _summary(values) for metric, values in deltas.items()
                    }
    output["paired_deltas"] = comparisons
    return output


def _campaign_record_error(data: dict[str, Any], expected: Any) -> str | None:
    """Return a provenance error for a record resolved from the campaign spec."""

    config = data.get("config")
    if config != expected.config:
        return "resolved config does not match the campaign job"
    environment = data.get("environment")
    expected_device = "cuda" if expected.seed_group == "primary" else "cpu"
    if not isinstance(environment, dict) or environment.get("used_device") != expected_device:
        return f"recorded device does not match {expected.seed_group} seed group"
    return None


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Temporal Factorial Campaign: {summary['campaign']['name']}",
        "",
        f"Verified runs: {summary['runs']} / {summary['expected_runs']}",
        "",
        "| Cell | Split | Method | Decision accuracy | Utility | Query rate |",
        "|---|---|---|---:|---:|---:|",
    ]
    for cell_id, cell in summary["cells"].items():
        for split_name, split in cell["splits"].items():
            for method, data in split["methods"].items():
                metrics = data["metrics"]

                def fmt(name: str) -> str:
                    item = metrics.get(name, {})
                    return "NA" if "mean" not in item else f"{item['mean']:.4f} +/- {item.get('std', 0.0):.4f}"

                lines.append(
                    f"| {cell_id} | {split_name} | {method} | {fmt('decision_accuracy')} | "
                    f"{fmt('utility')} | {fmt('query_rate')} |"
                )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-spec", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--pattern", default="temporal_factorial-*.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()

    spec = load_campaign_spec(args.campaign_spec)
    validator = _load_validator()
    expected_jobs = {
        (job.seed_group, job.cell_id, job.seed): job
        for seed_group in ("primary", "replication")
        for job in expand_campaign_spec(spec, seed_group)
    }
    input_dir = pathlib.Path(args.input_dir)
    output_path = pathlib.Path(args.output).resolve()
    records_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    rejected: list[dict[str, str]] = []
    for path in sorted(input_dir.glob(args.pattern)):
        if path.resolve() == output_path:
            continue
        try:
            ok, errors = validator.validate(path)
            if not ok:
                rejected.append({"file": str(path), "reason": "; ".join(errors)})
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            config = data.get("config", {})
            key = (
                str(config.get("campaign_seed_group")),
                str(config.get("campaign_cell")),
                int(data["seed"]),
            )
            if str(config.get("campaign_name")) != spec["name"]:
                rejected.append({"file": str(path), "reason": "campaign name mismatch"})
                continue
            if key not in expected_jobs:
                rejected.append({"file": str(path), "reason": f"unexpected campaign job key: {key}"})
                continue
            if key in records_by_key:
                rejected.append({"file": str(path), "reason": f"duplicate campaign job key: {key}"})
                continue
            provenance_error = _campaign_record_error(data, expected_jobs[key])
            if provenance_error:
                rejected.append({"file": str(path), "reason": provenance_error})
                continue
            records_by_key[key] = data
        except Exception as exc:
            rejected.append({"file": str(path), "reason": f"analysis error: {type(exc).__name__}: {exc}"})

    missing = sorted(set(expected_jobs) - set(records_by_key))
    if missing:
        raise SystemExit(f"campaign incomplete: {len(missing)} expected jobs missing; first: {missing[:3]}")
    records = list(records_by_key.values())
    cells: dict[str, Any] = {}
    for cell in spec["cells"]:
        cell_id = str(cell["id"])
        cell_records = [record for key, record in records_by_key.items() if key[1] == cell_id]
        cell_summary = _aggregate_records(cell_records)
        cell_summary.update(
            {
                "hypothesis": cell["hypothesis"],
                "overrides": cell.get("overrides", {}),
                "seed_group_runs": {
                    seed_group: sum(
                        1
                        for key in records_by_key
                        if key[0] == seed_group and key[1] == cell_id
                    )
                    for seed_group in ("primary", "replication")
                },
            }
        )
        cells[cell_id] = cell_summary

    summary = {
        "schema": "conflictbench.temporal_aggregate.v2",
        "campaign": {
            "name": spec["name"],
            "campaign_schema": spec["schema"],
            "cells": len(spec["cells"]),
            "primary_seeds": len(spec["primary_seeds"]),
            "replication_seeds": len(spec["replication_seeds"]),
        },
        "expected_runs": len(expected_jobs),
        "runs": len(records),
        "rejected": rejected,
        "missing": missing,
        "cells": cells,
    }
    summary["paired_deltas"] = {cell_id: cells[cell_id].pop("paired_deltas", {}) for cell_id in cells}
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        md = pathlib.Path(args.markdown)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps({"output": str(out), "runs": len(records), "rejected": len(rejected)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
