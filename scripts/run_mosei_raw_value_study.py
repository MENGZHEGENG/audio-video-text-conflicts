#!/usr/bin/env python3
"""Run one leakage-safe study over pooled released CMU-MOSEI descriptors."""

from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Sequence

from conflictbench.mosei_raw_value_study import (
    run_raw_value_study,
    validate_raw_value_result,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--mode", required=True, choices=("singleton", "pair"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    config_path = pathlib.Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    record = run_raw_value_study(args.cache, config, mode=args.mode, seed=args.seed)
    errors = validate_raw_value_result(
        record,
        expected_config=config,
        expected_mode=args.mode,
        expected_seed=args.seed,
        expected_cache_sha256=config.get("expected_cache_sha256"),
    )
    record["validation"] = {"ok": not errors, "errors": errors}

    destination = pathlib.Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(destination), "validation": record["validation"]}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
