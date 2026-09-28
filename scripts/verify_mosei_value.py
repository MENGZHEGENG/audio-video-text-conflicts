#!/usr/bin/env python3
"""Validate one saved CMU-MOSEI acquisition-value result."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

from conflictbench.mosei_value_study import (
    configuration_sha256,
    implementation_sha256,
    run_value_study,
    validate_value_result,
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--mode", choices=("singleton", "pair"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cache-sha256")
    parser.add_argument("--cache")
    parser.add_argument("--config")
    attestation = parser.add_mutually_exclusive_group()
    attestation.add_argument("--write-attestation")
    attestation.add_argument("--check-attestation")
    args = parser.parse_args()

    with pathlib.Path(args.path).open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    expected_configuration_sha256 = None
    expected_config = None
    if args.config:
        with pathlib.Path(args.config).open("r", encoding="utf-8") as handle:
            expected_config = json.load(handle)
        expected_configuration_sha256 = configuration_sha256(expected_config)
    errors = validate_value_result(
        record,
        expected_mode=args.mode,
        expected_seed=args.seed,
        expected_cache_sha256=args.cache_sha256,
        expected_configuration_sha256=expected_configuration_sha256,
        expected_config=expected_config,
    )
    source = pathlib.Path(args.path).resolve()
    if args.cache:
        if expected_config is None or args.mode is None or args.seed is None:
            errors.append("cache replay requires --config, --mode, and --seed")
        else:
            replay = run_value_study(args.cache, expected_config, mode=args.mode, seed=args.seed)
            if replay != record:
                errors.append("saved result does not exactly match deterministic cache replay")
    if args.check_attestation:
        attestation_path = pathlib.Path(args.check_attestation).resolve()
        try:
            with attestation_path.open("r", encoding="utf-8") as handle:
                saved_attestation = json.load(handle)
            expected_attestation = {
                "schema": "conflictbench.mosei-value-attestation.v1",
                "result_file": source.name,
                "result_sha256": _sha256(source),
                "configuration_sha256": configuration_sha256(expected_config),
                "implementation_sha256": implementation_sha256(),
                "mode": args.mode,
                "seed": args.seed,
            }
            if saved_attestation != expected_attestation:
                errors.append("verification record does not match the result")
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
            errors.append("verification record is missing or invalid")
    if args.write_attestation and not errors:
        if expected_config is None or args.mode is None or args.seed is None or not args.cache:
            errors.append("writing a verification record requires cache replay and locked arguments")
        else:
            attestation_path = pathlib.Path(args.write_attestation).resolve()
            attestation_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": "conflictbench.mosei-value-attestation.v1",
                "result_file": source.name,
                "result_sha256": _sha256(source),
                "configuration_sha256": configuration_sha256(expected_config),
                "implementation_sha256": implementation_sha256(),
                "mode": args.mode,
                "seed": args.seed,
            }
            with attestation_path.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
    print(json.dumps({"ok": not errors, "errors": errors, "path": args.path}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
