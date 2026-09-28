#!/usr/bin/env python3
"""Run one resolved job from a temporal factorial campaign specification."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from conflictbench.runner import run_from_config  # noqa: E402
from conflictbench.temporal_campaign import load_campaign_spec, resolve_campaign_job  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-spec", required=True, help="temporal_campaign.v2 JSON campaign specification")
    parser.add_argument(
        "--seed-group", choices=("primary", "replication"), required=True
    )
    parser.add_argument("--index", type=int, required=True, help="stable expanded run index")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    spec = load_campaign_spec(args.campaign_spec)
    job = resolve_campaign_job(spec, args.seed_group, args.index)
    requested_device = args.device or (
        "cuda" if args.seed_group == "primary" else "cpu"
    )
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        f"{spec['name']}-{args.seed_group}-{job.cell_id}-seed{job.seed}.json"
    )
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")

    result = run_from_config(job.config, device_preference=requested_device)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    used_device = result.get("environment", {}).get("used_device")
    if requested_device == "cuda" and used_device != "cuda":
        raise SystemExit(f"requested CUDA but runner used {used_device!r}; output left for diagnosis: {output}")
    if requested_device == "cpu" and used_device != "cpu":
        raise SystemExit(f"requested CPU but runner used {used_device!r}; output left for diagnosis: {output}")
    if result.get("validation", {}).get("ok") is not True:
        raise SystemExit(f"runner validation failed; inspect {output}")
    print(
        json.dumps(
            {
                "output": str(output),
                "campaign": spec["name"],
                "cell": job.cell_id,
                "seed": job.seed,
                "seed_group": args.seed_group,
                "device": used_device,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
