#!/usr/bin/env python3
"""Recompute and verify one singleton-baseline aggregate."""

from __future__ import annotations

import argparse
import json
import pathlib

from conflictbench.mosei_singleton_baselines import (
    aggregate_singleton_baseline_results,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", action="append", required=True)
    args = parser.parse_args()

    with pathlib.Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    with pathlib.Path(args.path).open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    records = []
    for name in args.input:
        with pathlib.Path(name).open("r", encoding="utf-8") as handle:
            records.append(json.load(handle))
    expected = aggregate_singleton_baseline_results(records, config)
    if saved != expected:
        raise SystemExit("aggregate does not match the validated seed set")
    print(json.dumps({"ok": True, "path": args.path}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
