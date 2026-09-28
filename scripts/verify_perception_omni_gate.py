#!/usr/bin/env python3
"""Replay-verify a source-gate result without invoking the model."""

from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

from conflictbench.perception_omni_gate import (
    GateValidationError,
    _reject_json_constant,
    _unique_object,
    verify_gate_output_replay,
)


def _load_output(path: pathlib.Path) -> Mapping[str, Any]:
    try:
        payload = pathlib.Path(path).read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateValidationError(
            "gate output is not readable strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise GateValidationError("gate output must be a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--pilot-index", required=True, type=pathlib.Path)
    parser.add_argument("--pilot-index-sha256", required=True)
    parser.add_argument("--media-root", required=True, type=pathlib.Path)
    parser.add_argument("--source", required=True, action="append", type=pathlib.Path)
    parser.add_argument("--transcript", required=True, type=pathlib.Path)
    parser.add_argument("--transcript-sha256", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)

    replayed = verify_gate_output_replay(
        _load_output(args.output),
        config_path=args.config,
        expected_config_sha256=args.config_sha256,
        pilot_index_path=args.pilot_index,
        expected_pilot_index_sha256=args.pilot_index_sha256,
        media_root=args.media_root,
        source_paths=args.source,
        transcript_path=args.transcript,
        expected_transcript_sha256=args.transcript_sha256,
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "payload_sha256": replayed["payload_sha256"],
                "inference_transcript_sha256": replayed["inference_transcript_sha256"],
                "record_count": replayed["counts"]["record_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
