#!/usr/bin/env python3
"""Run frozen Qwen2.5-Omni inference on a locked training-only pilot index."""

from __future__ import annotations

import argparse
import pathlib
from collections.abc import Sequence

import conflictbench.perception_omni_gate as gate_module
from conflictbench.perception_omni_gate import (
    QwenOmniBackend,
    load_gate_configuration,
    run_source_sufficiency_gate,
    write_gate_output,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--pilot-index", required=True, type=pathlib.Path)
    parser.add_argument("--pilot-index-sha256", required=True)
    parser.add_argument("--media-root", required=True, type=pathlib.Path)
    parser.add_argument("--model-cache", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)

    config = load_gate_configuration(args.config, args.config_sha256)
    backend = QwenOmniBackend(
        model_config=config["model"],
        inference_config=config["inference"],
        cache_dir=args.model_cache,
    )
    output = run_source_sufficiency_gate(
        config_path=args.config,
        expected_config_sha256=args.config_sha256,
        pilot_index_path=args.pilot_index,
        expected_pilot_index_sha256=args.pilot_index_sha256,
        media_root=args.media_root,
        backend=backend,
        source_paths=[
            pathlib.Path(gate_module.__file__).resolve(),
            pathlib.Path(__file__).resolve(),
        ],
    )
    write_gate_output(
        args.output,
        output,
        expected_config_sha256=args.config_sha256,
        expected_pilot_index_sha256=args.pilot_index_sha256,
    )
    # A scientifically negative gate is a valid, preserved result. Nonzero
    # exits are reserved for structural or runtime failures raised above.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
