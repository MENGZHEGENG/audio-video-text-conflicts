"""Build the compact ConflictBench cache from an MMSA pickle."""

from __future__ import annotations

import argparse

from conflictbench.multibench import load_mmsa_bundle, save_descriptor_cache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sentiment-mode", choices=("nonnegative", "positive"), default="nonnegative")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bundle = load_mmsa_bundle(args.feature_path, dataset_name=args.dataset, sentiment_mode=args.sentiment_mode)
    output = save_descriptor_cache(bundle, args.output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
