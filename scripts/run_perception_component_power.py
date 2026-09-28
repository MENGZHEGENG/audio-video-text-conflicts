#!/usr/bin/env python3
"""Run the hash-bound Perception Test component-cluster power approximation."""

from __future__ import annotations

import argparse
import pathlib
from collections.abc import Sequence

import conflictbench.perception_component_power as power_module
from conflictbench.perception_component_power import (
    run_power_simulation,
    write_power_output,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--component-event-summary", required=True, type=pathlib.Path)
    parser.add_argument("--component-event-summary-sha256", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)

    result = run_power_simulation(
        config_path=args.config,
        expected_config_sha256=args.config_sha256,
        summary_path=args.component_event_summary,
        expected_summary_sha256=args.component_event_summary_sha256,
        source_paths={
            "perception_component_power.py": pathlib.Path(
                power_module.__file__
            ).resolve(),
            "run_perception_component_power.py": pathlib.Path(__file__).resolve(),
        },
    )
    write_power_output(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
