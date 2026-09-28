#!/usr/bin/env python3
"""Validate a natural-data source registry and its staged files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from conflictbench.source_registry import verify_source_registry_files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="Directory containing the registry-listed source files.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    try:
        source = json.loads(args.registry.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read registry: {exc}")
    if not isinstance(source, dict):
        parser.error("registry must be a JSON object")

    try:
        validated = verify_source_registry_files(source, args.source_root)
    except (OSError, ValueError) as exc:
        parser.error(f"source preflight failed: {exc}")

    result = {
        "schema": "conflictbench.natural-source-registry-validation.v1",
        "dataset_name": validated.dataset_name,
        "group_count": validated.group_count,
        "file_count": validated.file_count,
        "role_counts": validated.role_counts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
