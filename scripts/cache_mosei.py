#!/usr/bin/env python3
"""Project aligned CMU-MOSEI CSD files once into a compact scalar cache."""

from __future__ import annotations

import argparse
import json

from conflictbench.mosei import load_mosei_bundle, save_mosei_cache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="COVAREP CSD path")
    parser.add_argument("--video", required=True, help="OpenFace 2 CSD path")
    parser.add_argument("--text", required=True, help="timestamped word-vector CSD path")
    parser.add_argument("--labels", required=True, help="sentiment-label CSD path")
    parser.add_argument("--splits", required=True, help="JSON file containing train/valid/test video IDs")
    parser.add_argument("--output", required=True, help="destination .npz cache")
    parser.add_argument("--sentiment-mode", choices=("nonnegative", "positive"), default="nonnegative")
    parser.add_argument("--max-samples-per-split", type=int)
    args = parser.parse_args()
    bundle = load_mosei_bundle(
        {"audio": args.audio, "video": args.video, "text": args.text},
        args.labels,
        args.splits,
        sentiment_mode=args.sentiment_mode,
        max_samples_per_split=args.max_samples_per_split,
    )
    output = save_mosei_cache(bundle, args.output)
    print(
        json.dumps(
            {
                "cache": str(output),
                "metadata": bundle.metadata,
                "split_sizes": {name: len(split.y) for name, split in bundle.splits.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
