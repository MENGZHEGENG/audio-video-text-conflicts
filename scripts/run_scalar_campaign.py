#!/usr/bin/env python3
"""Run the paired scalar and learned-acquisition seed roster."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from conflictbench.runner import run_from_config


def _seed_group(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    seeds = tuple(value)
    if any(
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        for seed in seeds
    ):
        raise ValueError(f"{name} must contain nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{name} must contain unique seeds")
    return seeds


def load_seed_plan(path: Path) -> tuple[int, ...]:
    """Load the fixed primary and replication seed groups."""
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    primary = _seed_group(plan.get("primary_seeds"), "primary_seeds")
    replication = _seed_group(plan.get("replication_seeds"), "replication_seeds")
    if set(primary).intersection(replication):
        raise ValueError("primary and replication seeds must be disjoint")
    return primary + replication


def _load_config(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"configuration must be an object: {path}")
    return value


def run_campaign(
    campaign_config: Path,
    acquisition_config: Path,
    seeds: Sequence[int],
    output_dir: Path,
    device: str,
) -> None:
    """Run both configurations for each seed using write-once outputs."""
    campaign = _load_config(campaign_config)
    acquisition = _load_config(acquisition_config)
    output_dir = Path(output_dir)
    targets = [
        output_dir / lane / f"seed-{seed}.json"
        for seed in seeds
        for lane in ("campaign", "acquisition")
    ]
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite {existing[0]}")

    for seed in seeds:
        for lane, template in (("campaign", campaign), ("acquisition", acquisition)):
            config = dict(template)
            config["seed"] = seed
            result = run_from_config(config, device_preference=device)
            target = output_dir / lane / f"seed-{seed}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
            if result.get("validation", {}).get("ok") is not True:
                raise RuntimeError(f"validation failed for {lane} seed {seed}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-plan", type=Path, required=True)
    parser.add_argument("--campaign-config", type=Path, required=True)
    parser.add_argument("--acquisition-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    run_campaign(
        args.campaign_config,
        args.acquisition_config,
        load_seed_plan(args.seed_plan),
        args.output_dir,
        args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
