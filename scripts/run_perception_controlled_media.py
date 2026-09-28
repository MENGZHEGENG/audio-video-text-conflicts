#!/usr/bin/env python3
"""Build or verify deterministic Perception Test controlled-media edits."""

from __future__ import annotations

import argparse
import pathlib
from collections.abc import Sequence

from conflictbench.perception_media_edits import (
    construct_controlled_media,
    verify_controlled_media,
)


def _source_bindings(
    values: Sequence[str], *, label: str = "implementation sources"
) -> dict[str, pathlib.Path]:
    bindings: dict[str, pathlib.Path] = {}
    for value in values:
        role, separator, path = value.partition("=")
        if not separator or not role or not path or role in bindings:
            raise argparse.ArgumentTypeError(f"{label} must be unique ROLE=PATH values")
        bindings[role] = pathlib.Path(path)
    return bindings


def _digest_bindings(
    values: Sequence[str], *, label: str = "source digests"
) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for value in values:
        role, separator, digest = value.partition("=")
        if not separator or not role or not digest or role in bindings:
            raise argparse.ArgumentTypeError(
                f"{label} must be unique ROLE=SHA256 values"
            )
        bindings[role] = digest
    return bindings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", required=True, type=pathlib.Path)
    parser.add_argument("--configuration-sha256", required=True)
    parser.add_argument("--pilot-index", required=True, type=pathlib.Path)
    parser.add_argument("--pilot-index-sha256", required=True)
    parser.add_argument("--media-receipt", required=True, type=pathlib.Path)
    parser.add_argument("--media-receipt-sha256", required=True)
    parser.add_argument("--media-set-sha256", required=True)
    parser.add_argument("--media-root", required=True, type=pathlib.Path)
    parser.add_argument("--nuisance-configuration", required=True, type=pathlib.Path)
    parser.add_argument("--nuisance-configuration-sha256", required=True)
    parser.add_argument(
        "--nuisance-implementation-source", required=True, action="append"
    )
    parser.add_argument("--nuisance-implementation-bundle-sha256", required=True)
    parser.add_argument("--nuisance-contract", required=True, type=pathlib.Path)
    parser.add_argument("--nuisance-contract-sha256", required=True)
    parser.add_argument("--source-gate-configuration", required=True, type=pathlib.Path)
    parser.add_argument("--source-gate-configuration-sha256", required=True)
    parser.add_argument("--source-gate-output", required=True, type=pathlib.Path)
    parser.add_argument("--source-gate-output-sha256", required=True)
    parser.add_argument(
        "--source-gate-implementation-source", required=True, action="append"
    )
    parser.add_argument(
        "--source-gate-implementation-source-sha256",
        required=True,
        action="append",
    )
    parser.add_argument("--source-gate-implementation-bundle-sha256", required=True)
    parser.add_argument("--source-gate-transcript", required=True, type=pathlib.Path)
    parser.add_argument("--source-gate-transcript-sha256", required=True)
    parser.add_argument(
        "--temporal-implementation-source", required=True, action="append"
    )
    parser.add_argument(
        "--temporal-implementation-source-sha256",
        required=True,
        action="append",
    )
    parser.add_argument("--temporal-implementation-bundle-sha256", required=True)
    parser.add_argument("--temporal-calibration", required=True, type=pathlib.Path)
    parser.add_argument("--temporal-calibration-sha256", required=True)
    parser.add_argument("--ffmpeg", required=True, type=pathlib.Path)
    parser.add_argument("--ffprobe", required=True, type=pathlib.Path)
    parser.add_argument("--output-root", required=True, type=pathlib.Path)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args(argv)
    try:
        nuisance_sources = _source_bindings(
            args.nuisance_implementation_source, label="nuisance implementation sources"
        )
        source_gate_sources = _source_bindings(
            args.source_gate_implementation_source,
            label="source-gate implementation sources",
        )
        source_gate_source_digests = _digest_bindings(
            args.source_gate_implementation_source_sha256,
            label="source-gate implementation source digests",
        )
        temporal_sources = _source_bindings(
            args.temporal_implementation_source,
            label="temporal implementation sources",
        )
        temporal_source_digests = _digest_bindings(
            args.temporal_implementation_source_sha256,
            label="temporal implementation source digests",
        )
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    arguments = {
        "configuration_path": args.configuration,
        "expected_configuration_sha256": args.configuration_sha256,
        "pilot_index_path": args.pilot_index,
        "expected_pilot_index_sha256": args.pilot_index_sha256,
        "media_receipt_path": args.media_receipt,
        "expected_media_receipt_sha256": args.media_receipt_sha256,
        "expected_media_set_sha256": args.media_set_sha256,
        "media_root": args.media_root,
        "nuisance_configuration_path": args.nuisance_configuration,
        "expected_nuisance_configuration_sha256": (args.nuisance_configuration_sha256),
        "nuisance_implementation_sources": nuisance_sources,
        "expected_nuisance_implementation_bundle_sha256": (
            args.nuisance_implementation_bundle_sha256
        ),
        "nuisance_contract_path": args.nuisance_contract,
        "expected_nuisance_contract_sha256": args.nuisance_contract_sha256,
        "source_gate_configuration_path": args.source_gate_configuration,
        "expected_source_gate_configuration_sha256": (
            args.source_gate_configuration_sha256
        ),
        "source_gate_output_path": args.source_gate_output,
        "expected_source_gate_output_sha256": args.source_gate_output_sha256,
        "source_gate_implementation_sources": source_gate_sources,
        "expected_source_gate_implementation_source_sha256": (
            source_gate_source_digests
        ),
        "expected_source_gate_implementation_bundle_sha256": (
            args.source_gate_implementation_bundle_sha256
        ),
        "source_gate_transcript_path": args.source_gate_transcript,
        "expected_source_gate_transcript_sha256": (args.source_gate_transcript_sha256),
        "temporal_implementation_sources": temporal_sources,
        "expected_temporal_implementation_source_sha256": (temporal_source_digests),
        "expected_temporal_implementation_bundle_sha256": (
            args.temporal_implementation_bundle_sha256
        ),
        "temporal_calibration_path": args.temporal_calibration,
        "expected_temporal_calibration_sha256": args.temporal_calibration_sha256,
        "ffmpeg_path": args.ffmpeg,
        "ffprobe_path": args.ffprobe,
        "output_root": args.output_root,
    }
    if args.verify_existing:
        result = verify_controlled_media(**arguments)
    else:
        result = construct_controlled_media(**arguments)
    return 0 if result["continuation_allowed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
