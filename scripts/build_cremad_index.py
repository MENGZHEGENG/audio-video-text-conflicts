#!/usr/bin/env python3
"""Build a CREMA-D paired-media index after source admission."""
import argparse
import json
from pathlib import Path
from conflictbench.cremad_index import build_index

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repository-root", type=Path, required=True)
parser.add_argument("--role-allocation", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    parser.error("refusing to overwrite output")
try:
    result = build_index(args.repository_root, args.role_allocation)
except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
    parser.error(f"index build failed: {exc}")
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result["summary"], sort_keys=True))
