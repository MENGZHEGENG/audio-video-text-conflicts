#!/usr/bin/env python3
"""Validate released unprojected CMU-MOSEI CSD descriptors, not raw media."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

from conflictbench.mosei_raw_value_study import (
    configuration_sha256,
    implementation_sha256,
    run_raw_value_study,
    validate_raw_value_result,
)

VERIFICATION_SCHEMA = "conflictbench.mosei-raw-value-verification.v1"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_verification(
    source: pathlib.Path,
    record: Mapping[str, Any],
    config: Mapping[str, Any],
    cache_sha256: str,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema": VERIFICATION_SCHEMA,
        "result_file": source.name,
        "result_sha256": _sha256(source),
        "configuration_sha256": configuration_sha256(config),
        "implementation_sha256": implementation_sha256(),
        "raw_cache_sha256": cache_sha256,
        "projection_attestation_sha256": record["projection_fit"]["attestation_sha256"],
        "core_protocol": record["core_protocol"],
        "mode": mode,
        "seed": seed,
    }


def _write_new_json(path: pathlib.Path, record: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, destination)
        os.unlink(temporary_name)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--mode", choices=("singleton", "pair"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--replay", action="store_true")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--write-verification")
    action.add_argument("--check-verification")
    args = parser.parse_args(argv)

    source = pathlib.Path(args.path).expanduser().resolve()
    config_path = pathlib.Path(args.config).expanduser().resolve()
    cache_path = pathlib.Path(args.cache).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        record = json.load(handle)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    errors: list[str] = []
    cache_sha256 = _sha256(cache_path)
    if config.get("expected_cache_sha256") != cache_sha256:
        errors.append("raw cache SHA256 does not match the locked configuration")
    errors.extend(
        validate_raw_value_result(
            record,
            expected_config=config,
            expected_mode=args.mode,
            expected_seed=args.seed,
            expected_cache_sha256=cache_sha256,
        )
    )
    if record.get("validation") != {"ok": True, "errors": []}:
        errors.append("saved structural validation did not pass")
    if args.replay and not errors:
        replay = run_raw_value_study(
            cache_path,
            config,
            mode=args.mode,
            seed=args.seed,
        )
        if replay != record:
            errors.append("saved result does not exactly match deterministic cache replay")

    expected: dict[str, Any] | None = None
    if not errors:
        expected = _expected_verification(
            source,
            record,
            config,
            cache_sha256,
            args.mode,
            args.seed,
        )
    if args.check_verification:
        verification_path = pathlib.Path(args.check_verification).expanduser().resolve()
        try:
            saved = json.loads(verification_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            saved = None
        if expected is None or saved != expected:
            errors.append("verification record does not match the full result")
    if args.write_verification and not errors:
        assert expected is not None
        _write_new_json(pathlib.Path(args.write_verification), expected)

    print(json.dumps({"ok": not errors, "errors": errors, "path": str(source)}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
