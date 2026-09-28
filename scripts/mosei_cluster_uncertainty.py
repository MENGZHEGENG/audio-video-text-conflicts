#!/usr/bin/env python3
"""Video-cluster bootstrap intervals for fixed CMU-MOSEI score policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from conflictbench.core import policy_actions
from conflictbench.mosei import load_mosei_cache


class ClusterUncertaintyError(ValueError):
    """Raised when a fixed-fold cluster analysis is malformed."""


def video_id(sample_id: str) -> str:
    """Map an utterance identifier to its video-level sampling unit."""

    match = re.match(r"^(.*?)(?:\[\d+\]|_\d+)$", sample_id)
    return match.group(1) if match else sample_id


def cluster_bootstrap(
    outcomes: Mapping[str, Sequence[int]],
    sample_ids: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap accuracies and paired differences by whole video."""

    if replicates < 100:
        raise ClusterUncertaintyError("at least 100 bootstrap replicates are required")
    names = tuple(sorted(outcomes))
    arrays = {name: np.asarray(outcomes[name], dtype=np.float64) for name in names}
    if not names or any(
        values.shape != (len(sample_ids),) for values in arrays.values()
    ):
        raise ClusterUncertaintyError("every outcome vector must match sample IDs")
    if any(not np.isin(values, (0.0, 1.0)).all() for values in arrays.values()):
        raise ClusterUncertaintyError("outcomes must be binary")

    groups: dict[str, list[int]] = {}
    for index, sample_id in enumerate(sample_ids):
        groups.setdefault(video_id(str(sample_id)), []).append(index)
    group_ids = tuple(sorted(groups))
    if len(group_ids) < 2:
        raise ClusterUncertaintyError("at least two video clusters are required")
    counts = np.asarray([len(groups[item]) for item in group_ids], dtype=np.int64)
    successes = {
        name: np.asarray([arrays[name][groups[item]].sum() for item in group_ids])
        for name in names
    }
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(group_ids), size=(replicates, len(group_ids)))
    denominators = counts[sampled].sum(axis=1)
    draws = {
        name: successes[name][sampled].sum(axis=1) / denominators for name in names
    }

    def summarize(values: np.ndarray, point: float) -> dict[str, Any]:
        interval = np.quantile(values, [0.025, 0.975], method="linear")
        return {
            "point": point,
            "video_cluster_bootstrap_interval_95": [
                float(interval[0]),
                float(interval[1]),
            ],
        }

    methods = {
        name: summarize(draws[name], float(arrays[name].mean())) for name in names
    }
    paired: dict[str, Any] = {}
    for left, right in (("threshold", "majority"),):
        if left in draws and right in draws:
            paired[f"{left}_minus_{right}"] = summarize(
                draws[left] - draws[right],
                float(arrays[left].mean() - arrays[right].mean()),
            )
    return {
        "utterances": len(sample_ids),
        "video_clusters": len(group_ids),
        "methods": methods,
        "paired": paired,
    }


def analyze_cache(cache: Path, *, replicates: int, seed: int) -> dict[str, Any]:
    """Load the pinned cache and analyze fixed-policy test decisions."""

    bundle = load_mosei_cache(cache)
    test = bundle.splits["test"]
    actions = {
        "threshold": policy_actions("active_diagnostic", test.x, threshold=0.35).action,
        "majority": policy_actions("majority", test.x).action,
        "median": policy_actions("median", test.x).action,
        "weighted": policy_actions("weighted", test.x).action,
    }
    outcomes = {
        name: (action == test.y).astype(np.int8) for name, action in actions.items()
    }
    summary = cluster_bootstrap(
        outcomes,
        bundle.sample_ids["test"],
        replicates=replicates,
        seed=seed,
    )
    return {
        "schema": "conflictbench.mosei-fixed-policy-cluster-uncertainty.v1",
        "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "bootstrap": {
            "unit": "official_test_video",
            "replicates": replicates,
            "seed": seed,
            "interval": "percentile_95",
        },
        "dataset": bundle.metadata,
        "test": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    report = analyze_cache(args.cache, replicates=args.replicates, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
