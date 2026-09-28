#!/usr/bin/env python3
"""Extract legacy positional-convolution tensors from one pinned checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from conflictbench.posconv_compat import checkpoint_file, prepare_patch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        parser.error("immutable receipt already exists")
    if args.snapshot.name != args.revision:
        parser.error("snapshot directory does not match pinned revision")
    checkpoint = checkpoint_file(args.snapshot)
    tensor_digest = prepare_patch(checkpoint, args.model_id, args.revision, args.output)
    receipt = {
        "schema": "conflictbench.cremad-posconv-patch.v1",
        "model_id": args.model_id,
        "revision": args.revision,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "patch": str(args.output),
        "patch_sha256": sha256(args.output),
        "tensor_sha256": tensor_digest,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
