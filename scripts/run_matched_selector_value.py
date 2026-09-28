#!/usr/bin/env python3
"""Run one seed of the matched-budget selector-value study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conflictbench.matched_selector_value import (  # noqa: E402
    DEFAULT_TEST_MECHANISMS,
    DEFAULT_TRAIN_MECHANISMS,
    SelectorRunConfig,
    load_seeds,
    run_selector_value,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--seed", type=int)
    seed_group.add_argument("--seed-index", type=int)
    parser.add_argument("--seed-plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("name") != "matched_selector_value":
        raise ValueError("unexpected experiment configuration name")
    if config.get("train_mechanisms") != list(DEFAULT_TRAIN_MECHANISMS):
        raise ValueError("training mechanisms differ from the frozen scalar protocol")
    if config.get("evaluation_mechanisms") != list(DEFAULT_TEST_MECHANISMS):
        raise ValueError("evaluation mechanisms differ from the frozen scalar protocol")
    if int(config.get("random_seed_offset", -1)) != 500_003:
        raise ValueError("random-selector seed offset differs from the frozen protocol")
    if args.seed_index is not None:
        if args.seed_plan is None:
            raise ValueError("--seed-plan is required with --seed-index")
        seed_plan = json.loads(args.seed_plan.read_text(encoding="utf-8"))
        seeds = load_seeds(seed_plan)
        if not 0 <= args.seed_index < len(seeds):
            raise ValueError("seed index is outside the declared seed plan")
        seed = seeds[args.seed_index]
    else:
        seed = args.seed
    run_config = SelectorRunConfig(
        seed=seed,
        train_size=int(config["train_size"]),
        eval_size=int(config["eval_size"]),
        strength=float(config["strength"]),
        noise=float(config["noise"]),
        threshold=float(config["score_threshold"]),
        query_cost=float(config["query_cost"]),
        abstain_cost=float(config["abstain_cost"]),
        budgets=tuple(float(value) for value in config["budgets"]),
        ridge_alpha=float(config["ridge_alpha"]),
    )
    result = run_selector_value(run_config)
    result["config_name"] = config["name"]

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"seed": seed, "output": str(output), "schema": result["schema"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
