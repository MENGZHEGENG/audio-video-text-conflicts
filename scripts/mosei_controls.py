#!/usr/bin/env python3
"""Calibrate matched policies on validation only, then evaluate fixed test folds."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from conflictbench.controls import protocol_controls, select_control_thresholds
from conflictbench.core import evaluate_actions
from conflictbench.mosei import load_mosei_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    bundle = load_mosei_cache(args.cache)
    thresholds = [0.01, 0.05, 0.1, 0.175, 0.25, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0]
    selected, validation = select_control_thresholds(bundle.splits['valid'], thresholds)
    test = bundle.splits['test']
    records = []
    for mode in ['fixed', 'calibrated']:
        for method in selected:
            threshold = 0.35 if mode == 'fixed' else selected[method]
            output = protocol_controls(test.x, threshold)[method]
            records.append(dict(mode=mode, method=method, threshold=threshold,
                                **evaluate_actions(output, test, initial_threshold=threshold)))
    result = dict(schema='conflictbench.mosei_controls.v1', thresholds=thresholds,
                  selection='maximum validation utility; ties choose smallest threshold',
                  query_cost=0.1, abstain_cost=0.2, selected=selected,
                  validation=validation, test=records, dataset=bundle.metadata)
    with Path(args.output).open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(dict(selected=selected, test_records=len(records))))


if __name__ == '__main__':
    main()
