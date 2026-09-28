#!/usr/bin/env python3
"""Run the isolated CMU-MOSEI singleton action baselines."""

from __future__ import annotations

import argparse
import json
import pathlib

from conflictbench.mosei_singleton_baselines import (
    run_singleton_baselines,
    validate_singleton_baseline_result,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with pathlib.Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    result = run_singleton_baselines(args.cache, config, seed=args.seed)
    errors = validate_singleton_baseline_result(
        result,
        expected_seed=args.seed,
        expected_cache_sha256=config.get("expected_cache_sha256"),
        expected_config=config,
    )
    result["validation"] = {"ok": not errors, "errors": errors}
    destination = pathlib.Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {"output": str(destination), "validation": result["validation"]},
            sort_keys=True,
        )
    )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
