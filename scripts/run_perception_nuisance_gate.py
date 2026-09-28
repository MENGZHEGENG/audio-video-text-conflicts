#!/usr/bin/env python3
"""Run the hash-bound Perception Test edit-role nuisance gate."""

from __future__ import annotations

import argparse
import pathlib
from collections.abc import Sequence

import conflictbench as package_module
import conflictbench.perception_candidate_audit as audit_module
import conflictbench.perception_media_pilot as pilot_module
import conflictbench.perception_nuisance_gate as gate_module
import conflictbench.perception_omni_gate as omni_module
from conflictbench.perception_nuisance_gate import run_nuisance_gate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--pilot-index", required=True, type=pathlib.Path)
    parser.add_argument("--pilot-index-sha256", required=True)
    parser.add_argument("--media-receipt", required=True, type=pathlib.Path)
    parser.add_argument("--media-receipt-sha256", required=True)
    parser.add_argument("--media-root", required=True, type=pathlib.Path)
    parser.add_argument("--media-set-sha256", required=True)
    parser.add_argument("--implementation-bundle-sha256", required=True)
    parser.add_argument("--nuisance-contract", required=True, type=pathlib.Path)
    parser.add_argument("--nuisance-contract-sha256", required=True)
    parser.add_argument("--derived-media-root", type=pathlib.Path)
    parser.add_argument("--shortcut-preregistration", type=pathlib.Path)
    parser.add_argument("--shortcut-preregistration-sha256")
    parser.add_argument("--shortcut-output", type=pathlib.Path)
    parser.add_argument("--shortcut-output-sha256")
    parser.add_argument("--power-record", type=pathlib.Path)
    parser.add_argument("--power-record-sha256")
    for prefix in (
        "raw-conflict-index",
        "source-only-output",
        "semantic-gate-output",
        "derived-media-v4-index",
    ):
        parser.add_argument(f"--{prefix}", type=pathlib.Path)
        parser.add_argument(f"--{prefix}-sha256")
        parser.add_argument(f"--{prefix}-verification", type=pathlib.Path)
        parser.add_argument(f"--{prefix}-verification-sha256")
    parser.add_argument("--replay-worker", type=pathlib.Path)
    parser.add_argument("--replay-worker-sha256")
    parser.add_argument("--replay-worker-configuration", type=pathlib.Path)
    parser.add_argument("--replay-worker-configuration-sha256")
    parser.add_argument("--replay-worker-model", type=pathlib.Path)
    parser.add_argument("--replay-worker-model-sha256")
    parser.add_argument("--ffprobe", required=True, type=pathlib.Path)
    parser.add_argument("--report", required=True, type=pathlib.Path)
    parser.add_argument("--result", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)

    dependency_fields = {
        "raw_conflict_index": "raw_conflict_index",
        "source_only": "source_only_output",
        "semantic_gate": "semantic_gate_output",
        "derived_media_v4": "derived_media_v4_index",
    }
    subject_paths = {
        role: getattr(args, field) for role, field in dependency_fields.items()
    }
    subject_sha256s = {
        role: getattr(args, f"{field}_sha256")
        for role, field in dependency_fields.items()
    }
    verification_paths = {
        role: getattr(args, f"{field}_verification")
        for role, field in dependency_fields.items()
    }
    verification_sha256s = {
        role: getattr(args, f"{field}_verification_sha256")
        for role, field in dependency_fields.items()
    }
    if not all(subject_paths.values()) or not all(subject_sha256s.values()):
        subject_paths = None
        subject_sha256s = None
    if not all(verification_paths.values()) or not all(verification_sha256s.values()):
        verification_paths = None
        verification_sha256s = None

    run_nuisance_gate(
        config_path=args.config,
        expected_config_sha256=args.config_sha256,
        pilot_index_path=args.pilot_index,
        expected_pilot_index_sha256=args.pilot_index_sha256,
        media_receipt_path=args.media_receipt,
        expected_media_receipt_sha256=args.media_receipt_sha256,
        media_root=args.media_root,
        expected_media_set_sha256=args.media_set_sha256,
        expected_implementation_bundle_sha256=args.implementation_bundle_sha256,
        nuisance_contract_path=args.nuisance_contract,
        expected_nuisance_contract_sha256=args.nuisance_contract_sha256,
        derived_media_root=args.derived_media_root,
        shortcut_preregistration_path=args.shortcut_preregistration,
        expected_shortcut_preregistration_sha256=(args.shortcut_preregistration_sha256),
        shortcut_output_path=args.shortcut_output,
        expected_shortcut_output_sha256=args.shortcut_output_sha256,
        power_record_path=args.power_record,
        expected_power_record_sha256=args.power_record_sha256,
        dependency_subject_paths=subject_paths,
        dependency_subject_sha256s=subject_sha256s,
        dependency_record_paths=verification_paths,
        dependency_record_sha256s=verification_sha256s,
        replay_worker_path=args.replay_worker,
        expected_replay_worker_sha256=args.replay_worker_sha256,
        replay_worker_configuration_path=args.replay_worker_configuration,
        expected_replay_worker_configuration_sha256=(
            args.replay_worker_configuration_sha256
        ),
        replay_worker_model_path=args.replay_worker_model,
        expected_replay_worker_model_sha256=args.replay_worker_model_sha256,
        ffprobe_path=args.ffprobe,
        report_path=args.report,
        result_path=args.result,
        source_paths={
            "__init__.py": pathlib.Path(package_module.__file__).resolve(),
            "runner.py": pathlib.Path(__file__).resolve(),
            "perception_nuisance_gate.py": pathlib.Path(gate_module.__file__).resolve(),
            "perception_media_pilot.py": pathlib.Path(pilot_module.__file__).resolve(),
            "perception_candidate_audit.py": pathlib.Path(
                audit_module.__file__
            ).resolve(),
            "perception_omni_gate.py": pathlib.Path(omni_module.__file__).resolve(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
