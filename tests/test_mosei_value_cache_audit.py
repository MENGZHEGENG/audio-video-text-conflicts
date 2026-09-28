import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_mosei_value_cache.py"
VERIFY_SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_mosei_value_cache_audit.py"
EXPECTED_COUNTS = {"train": 16_327, "valid": 1_871, "test": 4_662}
SPEC = importlib.util.spec_from_file_location("audit_mosei_value_cache", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


def _write_valid_cache(path: Path) -> None:
    arrays = {
        "cache_schema": np.asarray("conflictbench.mosei-cache.v2"),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "dataset": "CMU-MOSEI",
                    "alignment": "positive_interval_overlap_mean",
                    "loaded_split_counts": EXPECTED_COUNTS,
                },
                sort_keys=True,
            )
        ),
    }
    for split, count in EXPECTED_COUNTS.items():
        arrays[f"{split}_x"] = np.zeros((count, 3), dtype=np.float32)
        arrays[f"{split}_y"] = np.arange(count, dtype=np.int64) % 2
        arrays[f"{split}_sample_ids"] = np.asarray(
            [f"{split}-video-{index:05d}[0]" for index in range(count)],
            dtype="U",
        )
    np.savez_compressed(path, **arrays)


def _small_arrays() -> dict[str, np.ndarray]:
    counts = {"train": 2, "valid": 1, "test": 1}
    arrays = {
        "cache_schema": np.asarray("conflictbench.mosei-cache.v2"),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "dataset": "CMU-MOSEI",
                    "alignment": "positive_interval_overlap_mean",
                    "loaded_split_counts": counts,
                },
                sort_keys=True,
            )
        ),
    }
    for split, count in counts.items():
        arrays[f"{split}_x"] = np.zeros((count, 3), dtype=np.float32)
        arrays[f"{split}_y"] = np.arange(count, dtype=np.int64) % 2
        arrays[f"{split}_sample_ids"] = np.asarray(
            [f"{split}-video-{index}[0]" for index in range(count)], dtype="U"
        )
    return arrays


def _audit_small(tmp_path: Path, monkeypatch, arrays: dict[str, np.ndarray]):
    monkeypatch.setattr(AUDITOR, "EXPECTED_COUNTS", {"train": 2, "valid": 1, "test": 1})
    tmp_path.mkdir(parents=True, exist_ok=True)
    cache = tmp_path / "scores.npz"
    np.savez(cache, **arrays)
    return AUDITOR.audit_cache(cache)


def test_cli_writes_structural_audit_with_digest_exclusively(tmp_path):
    cache = tmp_path / "scores.npz"
    output = tmp_path / "audit.json"
    _write_valid_cache(cache)

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--cache", str(cache), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record == {
        "schema": "conflictbench.mosei-value-cache-audit.v1",
        "cache": {
            "filename": "scores.npz",
            "schema": "conflictbench.mosei-cache.v2",
            "sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        },
        "checks": {
            "arrays_exact": True,
            "finite_numeric_values": True,
            "sample_ids_unique": True,
            "utterance_ids_valid": True,
            "video_groups_disjoint": True,
        },
        "splits": {
            split: {"samples": count, "video_groups": count}
            for split, count in EXPECTED_COUNTS.items()
        },
        "test_split_use": "structural_audit_only",
    }
    assert json.loads(completed.stdout) == record

    repeated = subprocess.run(
        [sys.executable, str(SCRIPT), "--cache", str(cache), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert repeated.returncode != 0
    assert "FileExistsError" in repeated.stderr


def test_rejects_whitespace_video_group_in_utterance_id(tmp_path, monkeypatch):
    arrays = _small_arrays()
    arrays["train_sample_ids"][0] = " [0]"

    with pytest.raises(ValueError, match="invalid utterance sample ID"):
        _audit_small(tmp_path, monkeypatch, arrays)


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        ("conflictbench.mosei-cache.v1", "unsupported MOSEI cache schema"),
        ("conflictbench.mosei-cache.v3", "unsupported MOSEI cache schema"),
    ],
)
def test_rejects_any_cache_schema_except_v2(tmp_path, monkeypatch, schema, message):
    arrays = _small_arrays()
    arrays["cache_schema"] = np.asarray(schema)

    with pytest.raises(ValueError, match=message):
        _audit_small(tmp_path, monkeypatch, arrays)


def test_rejects_missing_or_unexpected_arrays(tmp_path, monkeypatch):
    missing = _small_arrays()
    missing.pop("valid_y")
    with pytest.raises(ValueError, match=r"missing=\['valid_y'\]"):
        _audit_small(tmp_path / "missing", monkeypatch, missing)

    unexpected = _small_arrays()
    unexpected["debug_values"] = np.zeros(1, dtype=np.float32)
    with pytest.raises(ValueError, match=r"unexpected=\['debug_values'\]"):
        _audit_small(tmp_path / "unexpected", monkeypatch, unexpected)


def test_rejects_wrong_shapes_and_dtypes(tmp_path, monkeypatch):
    wrong_shape = _small_arrays()
    wrong_shape["valid_x"] = np.zeros((1, 2), dtype=np.float32)
    with pytest.raises(ValueError, match=r"valid_x.*shape"):
        _audit_small(tmp_path / "shape", monkeypatch, wrong_shape)

    wrong_dtype = _small_arrays()
    wrong_dtype["train_y"] = wrong_dtype["train_y"].astype(np.int32)
    with pytest.raises(ValueError, match=r"train_y.*dtype int64"):
        _audit_small(tmp_path / "dtype", monkeypatch, wrong_dtype)


def test_rejects_nonfinite_scores_and_nonbinary_labels(tmp_path, monkeypatch):
    nonfinite = _small_arrays()
    nonfinite["test_x"][0, 2] = np.nan
    with pytest.raises(ValueError, match="test contains non-finite"):
        _audit_small(tmp_path / "nonfinite", monkeypatch, nonfinite)

    nonbinary = _small_arrays()
    nonbinary["valid_y"][0] = 2
    with pytest.raises(ValueError, match="valid_y must contain only binary labels"):
        _audit_small(tmp_path / "nonbinary", monkeypatch, nonbinary)


def test_rejects_duplicate_sample_ids_and_cross_split_video_groups(tmp_path, monkeypatch):
    duplicate = _small_arrays()
    duplicate["train_sample_ids"][1] = duplicate["train_sample_ids"][0]
    with pytest.raises(ValueError, match="train contains duplicate sample IDs"):
        _audit_small(tmp_path / "duplicate", monkeypatch, duplicate)

    overlap = _small_arrays()
    overlap["valid_sample_ids"][0] = "train-video-0[7]"
    with pytest.raises(ValueError, match="video groups overlap between train and valid"):
        _audit_small(tmp_path / "overlap", monkeypatch, overlap)


def test_allows_multiple_utterances_from_one_video_within_one_split(tmp_path, monkeypatch):
    arrays = _small_arrays()
    arrays["train_sample_ids"] = np.asarray(["same-video[0]", "same-video[1]"], dtype="U")

    record = _audit_small(tmp_path, monkeypatch, arrays)

    assert record["splits"]["train"] == {"samples": 2, "video_groups": 1}


def test_expected_sha256_is_an_optional_integrity_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(AUDITOR, "EXPECTED_COUNTS", {"train": 2, "valid": 1, "test": 1})
    cache = tmp_path / "scores.npz"
    np.savez(cache, **_small_arrays())
    digest = hashlib.sha256(cache.read_bytes()).hexdigest()

    assert AUDITOR.audit_cache(cache, expected_sha256=digest.upper())["cache"]["sha256"] == digest
    with pytest.raises(ValueError, match="cache SHA256 mismatch"):
        AUDITOR.audit_cache(cache, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="exactly 64 hexadecimal"):
        AUDITOR.audit_cache(cache, expected_sha256="not-a-digest")


def test_rejects_metadata_that_does_not_describe_aligned_mosei(tmp_path, monkeypatch):
    arrays = _small_arrays()
    arrays["metadata_json"] = np.asarray(
        json.dumps(
            {
                "dataset": "CMU-MOSEI",
                "alignment": "presegmented_entries",
                "loaded_split_counts": {"train": 2, "valid": 1, "test": 1},
            }
        )
    )

    with pytest.raises(ValueError, match="utterance alignment"):
        _audit_small(tmp_path, monkeypatch, arrays)


def test_saved_audit_verifier_rejects_missing_checks(tmp_path):
    cache = tmp_path / "scores.npz"
    output = tmp_path / "audit.json"
    _write_valid_cache(cache)
    record = AUDITOR.audit_cache(cache)
    output.write_text(json.dumps(record), encoding="utf-8")
    digest = hashlib.sha256(cache.read_bytes()).hexdigest()

    valid = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), str(output), "--expected-sha256", digest],
        capture_output=True,
        text=True,
        check=False,
    )
    assert valid.returncode == 0, valid.stderr

    record["checks"].pop("video_groups_disjoint")
    output.write_text(json.dumps(record), encoding="utf-8")
    invalid = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), str(output), "--expected-sha256", digest],
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid.returncode == 2
    assert "audit checks are incomplete" in invalid.stdout
