#!/usr/bin/env python3
"""Matched protocol controls; grid is descriptive, never test-set selection."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from conflictbench.core import Dataset, evaluate_actions, generate_dataset
from conflictbench.controls import protocol_controls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    seeds = [11, 23, 37, 41, 53]
    records = []
    for noise in [0.11, 0.22, 0.44]:
        for seed in seeds:
            data = generate_dataset(seed, 4000, ['clean', 'mixed', 'burst', 'ambiguity'], noise=noise)
            for threshold in [0.175, 0.35, 0.7]:
                for subset in ['all', 'novel_only']:
                    mask = np.ones(len(data.y), dtype=bool) if subset == 'all' else data.mechanism != 'clean'
                    selected = Dataset(data.x[mask], data.y[mask], data.ambiguous[mask], data.mechanism[mask])
                    for method, output in protocol_controls(selected.x, threshold).items():
                        records.append(dict(seed=seed, noise=noise, threshold=threshold,
                                            subset=subset, method=method,
                                            **evaluate_actions(output, selected, initial_threshold=threshold)))
    result = dict(seeds=seeds, eval_size=4000, query_cost=0.1, abstain_cost=0.2, records=records)
    with Path(args.output).open('x') as handle:
        json.dump(result, handle, allow_nan=False, indent=2)
    print(json.dumps(dict(records=len(records), output=args.output)))


if __name__ == '__main__':
    main()
