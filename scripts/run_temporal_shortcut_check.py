#!/usr/bin/env python3
"""Run one full-information temporal shortcut diagnostic using the configured device."""

from __future__ import annotations

import argparse
import json
import pathlib
from datetime import datetime, timezone
from typing import Any

from conflictbench.core import evaluate_actions, torch_status
from conflictbench.temporal import (
    fit_and_predict_temporal_torch,
    generate_temporal_dataset,
    temporal_linear_actions,
    temporal_summary,
    temporal_template_actions,
    transform_temporal_dataset,
)

SCHEMA = "conflictbench.temporal_shortcut_check.v2"
CONDITIONS = (
    ("identity", "identity"),
    ("shared_permutation", "shared_permutation"),
    ("independent_channel_permutation", "independent_channel_permutation"),
    ("per_example_circular_shift", "per_example_circular_shift"),
)


def _load_job(
    path: pathlib.Path, index: int
) -> tuple[dict[str, Any], dict[str, Any], int]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema") != SCHEMA or not isinstance(spec.get("name"), str):
        raise ValueError(f"expected {SCHEMA} specification")
    seeds = spec.get("seeds")
    cells = spec.get("cells")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty unique list")
    if not isinstance(cells, list) or not cells:
        raise ValueError("cells must be a non-empty list")
    expected = len(seeds) * len(cells)
    if index < 0 or index >= expected:
        raise IndexError(f"index {index} is outside 0..{expected - 1}")
    cell = cells[index // len(seeds)]
    if not isinstance(cell, dict) or not isinstance(cell.get("id"), str):
        raise TypeError("each cell requires a string id")
    return spec, cell, int(seeds[index % len(seeds)])


def _generated_splits(cell: dict[str, Any], seed: int) -> tuple[Any, dict[str, Any]]:
    common = {
        "seq_len": int(cell["seq_len"]),
        "strength": float(cell["strength"]),
        "noise": float(cell["noise"]),
    }
    train = generate_temporal_dataset(
        seed, int(cell["train_size"]), cell["train_mechanisms"], **common
    )
    splits = {
        "seen": generate_temporal_dataset(
            seed + 100_003, int(cell["eval_size"]), cell["seen_mechanisms"], **common
        ),
        "unseen": generate_temporal_dataset(
            seed + 200_003, int(cell["eval_size"]), cell["unseen_mechanisms"], **common
        ),
    }
    return train, splits


def _transform(
    train: Any, splits: dict[str, Any], *, mode: str, seed: int
) -> tuple[Any, dict[str, Any]]:
    if mode in {"identity", "shared_permutation"}:
        transform_seed = 700_001
        return (
            transform_temporal_dataset(train, mode=mode, seed=transform_seed),
            {
                name: transform_temporal_dataset(split, mode=mode, seed=transform_seed)
                for name, split in splits.items()
            },
        )
    return (
        transform_temporal_dataset(train, mode=mode, seed=seed + 300_003),
        {
            name: transform_temporal_dataset(
                split, mode=mode, seed=seed + 400_003 + offset
            )
            for offset, (name, split) in enumerate(splits.items())
        },
    )


def _evaluate(output: Any, split: Any) -> dict[str, Any]:
    return {"status": "verified", **evaluate_actions(output, temporal_summary(split))}


def _condition_result(
    train: Any, splits: dict[str, Any], cell: dict[str, Any], seed: int, device: str
) -> dict[str, Any]:
    outputs: dict[str, dict[str, Any]] = {
        "temporal_template": {},
        "temporal_linear": {},
    }
    for name, split in splits.items():
        outputs["temporal_template"][name] = temporal_template_actions(train, split)
        outputs["temporal_linear"][name] = temporal_linear_actions(train, split)
    for method in ("temporal_fusion", "temporal_gated"):
        method_outputs, fit = fit_and_predict_temporal_torch(
            method,
            train,
            splits,
            seed=seed,
            max_steps=int(cell["max_steps"]),
            batch_size=int(cell["batch_size"]),
            learning_rate=float(cell["learning_rate"]),
            width=int(cell["width"]),
            depth=int(cell["depth"]),
            device=device,
        )
        outputs[method] = {
            name: {"output": method_outputs[name], "fit": fit} for name in splits
        }
    result: dict[str, Any] = {"splits": {}}
    for split_name, split in splits.items():
        methods: dict[str, Any] = {}
        for method in ("temporal_template", "temporal_linear"):
            methods[method] = _evaluate(outputs[method][split_name], split)
        for method in ("temporal_fusion", "temporal_gated"):
            item = outputs[method][split_name]
            methods[method] = {**item["fit"], **_evaluate(item["output"], split)}
        result["splits"][split_name] = {"n": len(split.y), "methods": methods}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    status = torch_status()
    if not status.get("cuda_available"):
        raise SystemExit("CUDA is required for this diagnostic")
    spec, cell, seed = _load_job(pathlib.Path(args.config), args.index)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{spec['name']}-gpu-{cell['id']}-seed{seed}.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")

    raw_train, raw_splits = _generated_splits(cell, seed)
    conditions: dict[str, Any] = {}
    for condition_name, transform_name in CONDITIONS:
        train, splits = _transform(
            raw_train, raw_splits, mode=transform_name, seed=seed
        )
        conditions[condition_name] = {
            "transform": transform_name,
            **_condition_result(train, splits, cell, seed, args.device),
        }
    result = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "config": cell,
        "specification": {"name": spec["name"], "index": int(args.index)},
        "environment": {"torch": status, "used_device": args.device},
        "conditions": conditions,
        "validation": {
            "ok": True,
            "methods": [
                "temporal_template",
                "temporal_linear",
                "temporal_fusion",
                "temporal_gated",
            ],
        },
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(output), "cell": cell["id"], "seed": seed}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
