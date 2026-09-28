#!/usr/bin/env python3
"""Run ConflictBench from a JSON configuration."""

from __future__ import annotations

import argparse
import json
import pathlib

from conflictbench.runner import run_from_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    config = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
    if args.seed is not None:
        config["seed"] = args.seed
    result = run_from_config(config, device_preference=args.device)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "validation": result["validation"]}, sort_keys=True))
    return 0 if result["validation"]["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
