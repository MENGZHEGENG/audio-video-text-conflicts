#!/usr/bin/env python3
"""Validate a benchmark result without treating a partial run as success."""

from __future__ import annotations

import json
import math
import pathlib
import sys


BASE_METHODS = {"majority", "weighted", "median", "active_diagnostic", "mlp", "gated"}
TEMPORAL_METHODS = {"temporal_fusion", "temporal_gated"}
REQUIRED_SPLITS = {"seen", "unseen"}
TEMPORAL_PROTOCOL = "zero_mean_marker_v2"


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"non-finite JSON constant: {token}")


def _check_metric_values(prefix: str, metrics: object, errors: list[str]) -> None:
    if not isinstance(metrics, dict):
        errors.append(f"{prefix}: metrics missing")
        return
    for metric_name, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(f"{prefix}/{metric_name}: metric is not finite or null")


def validate(path: pathlib.Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    except Exception as exc:  # pragma: no cover - command-line diagnostics
        return False, [f"invalid JSON: {exc}"]
    if data.get("schema") != "conflictbench.run.v1":
        errors.append("unexpected schema")
    if not isinstance(data.get("seed"), int):
        errors.append("seed is missing")
    validation = data.get("validation")
    if not isinstance(validation, dict) or validation.get("ok") is not True:
        errors.append("runner validation flag is not true")
    config = data.get("config")
    dataset = str(config.get("dataset", "synthetic")).lower() if isinstance(config, dict) else "synthetic"
    temporal_result = dataset == "synthetic_temporal"
    if temporal_result and (not isinstance(validation, dict) or validation.get("temporal_methods_verified") is not True):
        errors.append("temporal_methods_verified flag is not true")
    if temporal_result:
        temporal_metadata = (data.get("dataset") or {}).get("temporal") if isinstance(data.get("dataset"), dict) else None
        if not isinstance(temporal_metadata, dict):
            errors.append("temporal dataset metadata is missing")
        elif temporal_metadata.get("protocol") != TEMPORAL_PROTOCOL:
            errors.append(f"temporal protocol must be {TEMPORAL_PROTOCOL}")
    splits = data.get("splits")
    if not isinstance(splits, dict) or not REQUIRED_SPLITS.issubset(splits):
        errors.append("required splits are missing")
    for split_name in REQUIRED_SPLITS:
        split = (splits or {}).get(split_name, {})
        methods = set((split.get("methods") or {}).keys())
        required_methods = set(BASE_METHODS)
        if isinstance(config, dict) and config.get("name") == "synthetic_acquisition_campaign":
            required_methods.add("learned_acquisition")
        if temporal_result:
            required_methods.update(TEMPORAL_METHODS)
        if not required_methods.issubset(methods):
            missing_methods = sorted(required_methods - methods)
            qualifier = "temporal " if temporal_result and TEMPORAL_METHODS.intersection(missing_methods) else ""
            errors.append(f"{split_name}: {qualifier}required methods are missing ({missing_methods})")
        for method_name, record in (split.get("methods") or {}).items():
            if temporal_result and method_name in TEMPORAL_METHODS and record.get("status") != "verified":
                errors.append(f"{split_name}/{method_name}: temporal method is not verified")
            if record.get("status") == "verified":
                _check_metric_values(f"{split_name}/{method_name}", record.get("metrics"), errors)
                by_mechanism = record.get("by_mechanism", {})
                if not isinstance(by_mechanism, dict):
                    errors.append(f"{split_name}/{method_name}: by_mechanism missing")
                else:
                    for mechanism_name, mechanism_record in by_mechanism.items():
                        if not isinstance(mechanism_record, dict):
                            errors.append(f"{split_name}/{method_name}/{mechanism_name}: record missing")
                            continue
                        _check_metric_values(
                            f"{split_name}/{method_name}/{mechanism_name}",
                            mechanism_record,
                            errors,
                        )
    return not errors, errors


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} RESULT.json", file=sys.stderr)
        return 2
    path = pathlib.Path(sys.argv[1])
    ok, errors = validate(path)
    if ok:
        print(f"verified: {path}")
        return 0
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
