#!/usr/bin/env python3
"""Validate and aggregate the complete singleton-baseline seed set."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import tempfile

from conflictbench.mosei_singleton_baselines import (
    aggregate_singleton_baseline_results,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = pathlib.Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    records = []
    for name in args.input:
        with pathlib.Path(name).open("r", encoding="utf-8") as handle:
            records.append(json.load(handle))
    aggregate = aggregate_singleton_baseline_results(records, config)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to replace aggregate: {output}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(aggregate, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o400)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
