#!/usr/bin/env python3
"""Validate and deterministically replay one singleton baseline result."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from conflictbench.mosei_singleton_baselines import (
    run_singleton_baselines,
    validate_singleton_baseline_result,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    with pathlib.Path(args.path).open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    with pathlib.Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    errors = validate_singleton_baseline_result(
        saved,
        expected_seed=args.seed,
        expected_cache_sha256=config.get("expected_cache_sha256"),
        expected_config=config,
    )
    replay = run_singleton_baselines(args.cache, config, seed=args.seed)
    if replay != saved:
        errors.append("saved result does not exactly match deterministic cache replay")
    message = json.dumps(
        {"ok": not errors, "errors": errors, "path": args.path}, sort_keys=True
    )
    print(message, file=sys.stderr if errors else sys.stdout)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
