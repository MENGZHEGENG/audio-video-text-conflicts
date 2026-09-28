#!/usr/bin/env python3
"""Pool CMU-MOSEI descriptors into a train/validation-only cache."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from conflictbench.mosei_raw import (
    RAW_CACHE_SCHEMA,
    build_raw_mosei_bundle,
    save_raw_mosei_cache,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="COVAREP CSD file")
    parser.add_argument("--video", required=True, help="OpenFace 2 CSD file")
    parser.add_argument("--text", required=True, help="timestamped word-vector CSD file")
    parser.add_argument("--labels", required=True, help="sentiment-label CSD file")
    parser.add_argument("--splits", required=True, help="JSON file with official video folds")
    parser.add_argument("--output", required=True, help="destination .npz file")
    parser.add_argument(
        "--sentiment-mode", choices=("nonnegative", "positive"), default="nonnegative"
    )
    parser.add_argument("--max-samples-per-split", type=int)
    args = parser.parse_args(argv)

    bundle = build_raw_mosei_bundle(
        {"audio": args.audio, "video": args.video, "text": args.text},
        args.labels,
        args.splits,
        sentiment_mode=args.sentiment_mode,
        max_samples_per_split=args.max_samples_per_split,
    )
    output = save_raw_mosei_cache(bundle, args.output)
    print(
        json.dumps(
            {
                "cache": str(output),
                "cache_schema": RAW_CACHE_SCHEMA,
                "split_sizes": {name: len(split.y) for name, split in bundle.splits.items()},
                "source_provenance": bundle.metadata["source_provenance"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
