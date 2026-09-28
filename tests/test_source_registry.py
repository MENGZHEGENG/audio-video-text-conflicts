from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from conflictbench.source_registry import validate_source_registry
from conflictbench.source_registry import verify_source_registry_files


ROOT = Path(__file__).parents[1]


def _registry() -> dict[str, object]:
    index = b'{"release":"revised"}\n'
    release = b"verified supervised source bytes\n"
    return {
        "schema": "conflictbench.natural-source-registry.v4",
        "dataset": {"name": "CH-SIMS revised", "release": "v2"},
        "terms": {
            "url": "https://example.org/terms",
            "reviewed_on": "2026-09-18",
            "research_use_permitted": True,
        },
        "files": [
            {
                "filename": "release-index.json",
                "size_bytes": len(index),
                "source_url": "https://example.org/release-index.json",
                "expected_sha256": hashlib.sha256(index).hexdigest(),
                "observed_sha256": hashlib.sha256(index).hexdigest(),
                "roles": ["release_index"],
            },
            {
                "filename": "supervised-release.pkl",
                "size_bytes": len(release),
                "source_url": "https://example.org/supervised-release.pkl",
                "expected_sha256": hashlib.sha256(release).hexdigest(),
                "observed_sha256": hashlib.sha256(release).hexdigest(),
                "roles": ["data", "labels", "role_allocation", "group_mapping"],
            },
        ],
        "labels": {
            "integrated": "label",
            "audio": "audio_label",
            "video": "video_label",
            "text": "text_label",
            "file_role": "labels",
        },
        "group_field": "video_id",
        "grouping": {
            "unit": "speaker",
            "provenance_url": "https://example.org/speaker-grouping",
            "file_role": "group_mapping",
        },
        "role_allocation": {"method": "source-defined split", "file_role": "role_allocation"},
        "role_groups": {
            "task_fit": ["v1", "v2"],
            "router_fit": ["v3"],
            "calibration": ["v4"],
            "evaluation": ["v5", "v6"],
        },
    }


def _write_registry_files(registry: dict[str, object], root: Path) -> None:
    (root / "release-index.json").write_bytes(b'{"release":"revised"}\n')
    (root / "supervised-release.pkl").write_bytes(b"verified supervised source bytes\n")


def test_valid_registry_requires_terms_digest_independent_labels_and_disjoint_roles():
    result = validate_source_registry(_registry())

    assert result.dataset_name == "CH-SIMS revised"
    assert result.group_count == 6
    assert result.file_count == 2
    assert result.role_counts == {
        "task_fit": 2,
        "router_fit": 1,
        "calibration": 1,
        "evaluation": 2,
    }


def test_optional_same_label_donor_role_does_not_exclude_four_role_sources():
    four_role = _registry()
    assert validate_source_registry(four_role).group_count == 6
    with_donor = copy.deepcopy(four_role)
    with_donor["role_groups"]["same_label_donor"] = ["v7"]
    validated = validate_source_registry(with_donor)
    assert validated.group_count == 7
    assert validated.role_counts["same_label_donor"] == 1
    with_donor["role_groups"]["unexpected"] = ["v8"]
    with pytest.raises(ValueError, match="groups"):
        validate_source_registry(with_donor)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda record: record["terms"].update({"research_use_permitted": False}), "terms"),
        (lambda record: record["files"][0].update({"observed_sha256": "b" * 64}), "files"),
        (lambda record: record["labels"].pop("video"), "labels"),
        (lambda record: record["files"][1].update({"roles": ["data", "labels"]}), "files"),
        (lambda record: record.pop("grouping"), "grouping"),
        (lambda record: record["grouping"].update({"unit": "segment"}), "grouping"),
        (lambda record: record["role_groups"].update({"evaluation": ["v2", "v6"]}), "overlap"),
    ],
)
def test_registry_rejects_each_phase_zero_gate(mutate, message):
    registry = copy.deepcopy(_registry())
    mutate(registry)

    with pytest.raises(ValueError, match=message):
        validate_source_registry(registry)


def test_registry_cli_writes_only_a_validated_summary(tmp_path):
    registry_path = tmp_path / "source_registry.json"
    output_path = tmp_path / "validated.json"
    registry = _registry()
    _write_registry_files(registry, tmp_path)
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_source_registry.py"),
            "--registry",
            str(registry_path),
            "--source-root",
            str(tmp_path),
            "--output",
            str(output_path),
        ],
        env={"PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "dataset_name": "CH-SIMS revised",
        "group_count": 6,
        "file_count": 2,
        "role_counts": {
            "calibration": 1,
            "evaluation": 2,
            "router_fit": 1,
            "task_fit": 2,
        },
        "schema": "conflictbench.natural-source-registry-validation.v1",
    }


def test_file_verifier_recomputes_index_and_payload_digests(tmp_path):
    registry = _registry()
    _write_registry_files(registry, tmp_path)

    result = verify_source_registry_files(registry, tmp_path)

    assert result.file_count == 2
    (tmp_path / "supervised-release.pkl").write_bytes(b"tampered bytes\n")
    with pytest.raises(ValueError, match="file bytes"):
        verify_source_registry_files(registry, tmp_path)


def test_file_verifier_rejects_path_escape(tmp_path):
    registry = _registry()
    _write_registry_files(registry, tmp_path)
    registry["files"][0]["filename"] = "../release-index.json"  # type: ignore[index]

    with pytest.raises(ValueError, match="files"):
        verify_source_registry_files(registry, tmp_path)
