#!/usr/bin/env python3
"""Create an immutable input contract for the Perception nuisance gate."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from collections.abc import Sequence

from conflictbench.perception_nuisance_gate import build_nuisance_contract


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration-sha256", required=True)
    parser.add_argument("--pilot-index-sha256", required=True)
    parser.add_argument("--media-receipt-sha256", required=True)
    parser.add_argument("--media-set-sha256", required=True)
    parser.add_argument("--implementation-bundle-sha256", required=True)
    parser.add_argument("--shortcut-preregistration-sha256")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)

    contract = build_nuisance_contract(
        configuration_sha256=args.configuration_sha256,
        pilot_index_sha256=args.pilot_index_sha256,
        media_receipt_sha256=args.media_receipt_sha256,
        media_set_sha256=args.media_set_sha256,
        implementation_bundle_sha256=args.implementation_bundle_sha256,
        shortcut_preregistration_sha256=args.shortcut_preregistration_sha256,
    )
    payload = (
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    except FileExistsError as exc:
        raise SystemExit(f"output already exists: {args.output}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
