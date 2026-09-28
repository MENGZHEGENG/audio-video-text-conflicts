#!/usr/bin/env python3
"""Summarize paired scalar results and clean-excluded mechanism performance."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

NOVEL_MECHANISMS = ("ambiguity", "burst", "mixed")
PRIMARY_METHOD = "active_diagnostic"
REFERENCE_METHOD = "majority"
ACQUISITION_METHOD = "learned_acquisition"


class AnalysisError(ValueError):
    """Raised when run records do not satisfy the paired analysis contract."""


def _read_runs(paths: Sequence[Path], label: str) -> dict[int, dict]:
    if not paths:
        raise AnalysisError(f"{label} run set is empty")
    records: dict[int, dict] = {}
    for path in sorted((Path(item) for item in paths), key=lambda item: item.name):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AnalysisError(f"cannot read {label} run {path}: {error}") from error
        seed = record.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise AnalysisError(f"{label} run {path} has no integer seed")
        validation = record.get("validation")
        if (
            not isinstance(validation, dict)
            or validation.get("ok") is not True
            or validation.get("base_methods_verified") is not True
            or validation.get("learned_methods_verified") is not True
            or validation.get("learned_methods_skipped") != []
        ):
            raise AnalysisError(f"{label} run {path} is not verified")
        if seed in records:
            raise AnalysisError(f"{label} run set has duplicate seed {seed}")
        records[seed] = record
    return records


def _source_set_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(item) for item in paths), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(b"\0")
    return digest.hexdigest()


def _method(record: dict, name: str) -> dict:
    try:
        method = record["splits"]["unseen"]["methods"][name]
    except (KeyError, TypeError) as error:
        raise AnalysisError(f"run lacks unseen method {name}") from error
    if not isinstance(method, dict):
        raise AnalysisError(f"unseen method {name} is not an object")
    return method


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{label} is not numeric")
    number = float(value)
    if not np.isfinite(number):
        raise AnalysisError(f"{label} is not finite")
    return number


def _metric(method: dict, name: str) -> float:
    try:
        value = method["metrics"][name]
    except (KeyError, TypeError) as error:
        raise AnalysisError(f"method lacks metric {name}") from error
    return _finite_number(value, f"metric {name}")


def _novel_rates(method: dict) -> tuple[float, float, float]:
    try:
        mechanisms = method["by_mechanism"]
    except (KeyError, TypeError) as error:
        raise AnalysisError("method lacks by_mechanism results") from error
    total = 0
    accuracy_sum = 0.0
    utility_sum = 0.0
    macro_accuracy = 0.0
    for mechanism in NOVEL_MECHANISMS:
        try:
            cell = mechanisms[mechanism]
            count = cell["n"]
        except (KeyError, TypeError) as error:
            raise AnalysisError(f"method lacks {mechanism} results") from error
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise AnalysisError(f"{mechanism}.n must be a positive integer")
        accuracy = _finite_number(
            cell.get("decision_accuracy"), f"{mechanism}.decision_accuracy"
        )
        utility = _finite_number(cell.get("utility"), f"{mechanism}.utility")
        if not 0.0 <= accuracy <= 1.0:
            raise AnalysisError(f"{mechanism}.decision_accuracy is out of range")
        total += count
        accuracy_sum += count * accuracy
        utility_sum += count * utility
        macro_accuracy += accuracy
    return (
        accuracy_sum / total,
        utility_sum / total,
        macro_accuracy / len(NOVEL_MECHANISMS),
    )


def _summary(
    values: Sequence[float], rng: np.random.Generator, replicates: int
) -> dict:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise AnalysisError("summary values must be a finite nonempty vector")
    if replicates <= 0:
        raise AnalysisError("bootstrap_replicates must be positive")
    indices = rng.integers(0, vector.size, size=(replicates, vector.size))
    bootstrap_means = vector[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, (0.025, 0.975))
    return {
        "mean": float(vector.mean()),
        "sample_standard_deviation": float(vector.std(ddof=1))
        if vector.size > 1
        else 0.0,
        "paired_seed_bootstrap_interval_95": [float(low), float(high)],
        "n": int(vector.size),
    }


def analyze_runs(
    campaign_paths: Sequence[Path],
    acquisition_paths: Sequence[Path],
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20_270_918,
) -> dict:
    campaign = _read_runs(campaign_paths, "campaign")
    acquisition = _read_runs(acquisition_paths, "acquisition")
    if set(campaign) != set(acquisition):
        raise AnalysisError("campaign and acquisition seed sets differ")
    seeds = sorted(campaign)
    rng = np.random.default_rng(bootstrap_seed)

    paired: dict[str, dict] = {}
    for metric_name in ("decision_accuracy", "utility"):
        differences = [
            _metric(_method(campaign[seed], PRIMARY_METHOD), metric_name)
            - _metric(_method(campaign[seed], REFERENCE_METHOD), metric_name)
            for seed in seeds
        ]
        paired[metric_name] = _summary(differences, rng, bootstrap_replicates)

    campaign_method_names = sorted(campaign[seeds[0]]["splits"]["unseen"]["methods"])
    novel: dict[str, dict] = {}
    for method_name in campaign_method_names:
        per_seed = [
            _novel_rates(_method(campaign[seed], method_name)) for seed in seeds
        ]
        novel[method_name] = {
            "decision_accuracy": _summary(
                [item[0] for item in per_seed], rng, bootstrap_replicates
            ),
            "utility": _summary(
                [item[1] for item in per_seed], rng, bootstrap_replicates
            ),
            "mechanism_macro_accuracy": _summary(
                [item[2] for item in per_seed], rng, bootstrap_replicates
            ),
        }

    acquisition_per_seed = [
        _novel_rates(_method(acquisition[seed], ACQUISITION_METHOD)) for seed in seeds
    ]
    novel[ACQUISITION_METHOD] = {
        "decision_accuracy": _summary(
            [item[0] for item in acquisition_per_seed], rng, bootstrap_replicates
        ),
        "utility": _summary(
            [item[1] for item in acquisition_per_seed], rng, bootstrap_replicates
        ),
        "mechanism_macro_accuracy": _summary(
            [item[2] for item in acquisition_per_seed], rng, bootstrap_replicates
        ),
    }

    return {
        "schema_version": 1,
        "run_count": len(seeds),
        "seeds": seeds,
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "unit": "paired_run_seed",
            "interval": "percentile_95",
        },
        "input_source_sets_sha256": {
            "campaign": _source_set_sha256(campaign_paths),
            "acquisition": _source_set_sha256(acquisition_paths),
        },
        "paired_active_minus_majority": paired,
        "novel_only": novel,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, nargs="+", required=True)
    parser.add_argument("--acquisition", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_270_918)
    args = parser.parse_args(argv)
    report = analyze_runs(
        args.campaign,
        args.acquisition,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
