#!/usr/bin/env python3
"""Validate a saved CMU-MOSEI value-cache audit record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Mapping


EXPECTED_COUNTS = {"train": 16_327, "valid": 1_871, "test": 4_662}
REQUIRED_CHECKS = {
    "arrays_exact": True,
    "finite_numeric_values": True,
    "sample_ids_unique": True,
    "utterance_ids_valid": True,
    "video_groups_disjoint": True,
}


def validate_record(record: Mapping[str, Any], expected_sha256: str) -> list[str]:
    errors: list[str] = []
    if record.get("schema") != "conflictbench.mosei-value-cache-audit.v1":
        errors.append("unexpected audit schema")
    cache = record.get("cache")
    if not isinstance(cache, Mapping):
        errors.append("cache summary is missing")
    else:
        if cache.get("schema") != "conflictbench.mosei-cache.v2":
            errors.append("unexpected cache schema")
        digest = cache.get("sha256")
        if digest != expected_sha256 or re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None:
            errors.append("cache SHA256 does not match the locked value")
    if record.get("checks") != REQUIRED_CHECKS:
        errors.append("audit checks are incomplete")
    splits = record.get("splits")
    if not isinstance(splits, Mapping):
        errors.append("split summary is missing")
    else:
        for name, count in EXPECTED_COUNTS.items():
            split = splits.get(name)
            if not isinstance(split, Mapping) or split.get("samples") != count:
                errors.append(f"{name} split count is invalid")
            elif not isinstance(split.get("video_groups"), int) or split["video_groups"] <= 0:
                errors.append(f"{name} video-group count is invalid")
    if record.get("test_split_use") != "structural_audit_only":
        errors.append("test split use is not structural-only")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    with Path(args.path).open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    errors = validate_record(record, args.expected_sha256)
    print(json.dumps({"ok": not errors, "errors": errors, "path": args.path}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
