from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import unicodedata
import zipfile

import pytest
from conflictbench.perception_candidate_audit import (
    AuditError,
    audit_archives,
    candidate_group_key,
    main,
    normalize_text,
)


def _question(*, answer_id: int, options: list[str] | None = None) -> dict:
    return {
        "id": 0,
        "question": "Which event happened?",
        "options": options or ["alpha", "beta", "gamma"],
        "answer_id": answer_id,
        "area": "semantics",
        "reasoning": "descriptive",
        "tag": ["event", "ordering"],
    }


def _video(video_id: str, split: str, question: dict) -> dict:
    return {
        "metadata": {
            "split": split,
            "video_id": video_id,
            "frame_rate": 30.0,
            "num_frames": 90,
            "resolution": [720, 1280],
            "audio_samples": 144000,
            "audio_sample_rate": 48000.0,
            "is_cup_game": 0,
            "is_camera_moving": 1,
        },
        "mc_question": [question],
    }


def _write_archive(path: pathlib.Path, member: str, payload: dict) -> tuple[int, str]:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_BZIP2) as archive:
        archive.writestr(member, encoded)
    data = path.read_bytes()
    return len(data), hashlib.sha256(data).hexdigest()


def _write_config(
    path: pathlib.Path,
    train_lock: tuple[int, str],
    validation_lock: tuple[int, str],
    *,
    minimum: int = 2,
    expected_counts: dict | None = None,
) -> None:
    if expected_counts is None:
        expected_counts = {
            "split_minimum_two_answer_group_count": {
                "train": 2,
                "validation": 2,
            },
            "cross_split_minimum_two_answer_group_count": 2,
            "cross_split_all_three_option_group_count": 1,
            "duplicate_option_exclusion_count": {
                "train": 0,
                "validation": 1,
            },
        }
    config = {
        "schema_version": 2,
        "audit": "perception_test_mcqa_candidate_pool",
        "purpose": "candidate_pool_feasibility_only",
        "source_repository": {
            "url": "https://github.com/google-deepmind/perception_test",
            "commit": "3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf",
            "annotation_license": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/legalcode",
        },
        "normalization": {
            "unicode": "NFC",
            "whitespace": "strip_and_collapse",
            "case_sensitive": True,
            "punctuation_sensitive": True,
        },
        "eligibility": {
            "min_unique_videos_per_supported_answer_per_split": minimum,
            "min_supported_answers_per_group": 2,
            "require_same_supported_answer_vocabulary_across_splits": True,
            "report_all_three_option_tier": True,
        },
        "expected_snapshot_counts": expected_counts,
        "archives": {
            "train": {
                "url": "https://storage.googleapis.com/dm-perception-test/zip_data/mc_question_train_annotations.zip?generation=1686575054862298",
                "generation": "1686575054862298",
                "size_bytes": train_lock[0],
                "sha256": train_lock[1],
                "member": "mc_question_train.json",
                "metadata_split": "train",
            },
            "validation": {
                "url": "https://storage.googleapis.com/dm-perception-test/zip_data/mc_question_valid_annotations.zip?generation=1686575054970958",
                "generation": "1686575054970958",
                "size_bytes": validation_lock[0],
                "sha256": validation_lock[1],
                "member": "mc_question_valid.json",
                "metadata_split": "valid",
            },
        },
    }
    path.write_text(json.dumps(config), encoding="utf-8")


def test_normalization_is_nfc_and_whitespace_only():
    decomposed = "  Cafe\u0301\tQUESTION\n"

    assert normalize_text(decomposed) == unicodedata.normalize("NFC", "Café QUESTION")
    assert normalize_text("Case, punctuation!") == "Case, punctuation!"


def test_official_audit_config_pins_the_reviewed_snapshot():
    config_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "configs"
        / "perception_test_mcqa_audit.json"
    )
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)

    assert hashlib.sha256(config_bytes).hexdigest() == (
        "0ed19f8193f4872897b572553869ac0ca3732a02f63afd4ca18794425000fe3a"
    )
    assert config["expected_snapshot_counts"] == {
        "cross_split_all_three_option_group_count": 30,
        "cross_split_minimum_two_answer_group_count": 59,
        "duplicate_option_exclusion_count": {"train": 0, "validation": 4},
        "split_minimum_two_answer_group_count": {
            "train": 252,
            "validation": 382,
        },
    }


def test_group_key_sorts_options_and_tags_but_uses_answer_text_before_sorting():
    first = _question(answer_id=0, options=["zeta", "alpha"])
    second = copy.deepcopy(first)
    second["options"] = ["alpha", "zeta"]
    second["answer_id"] = 1
    second["tag"] = list(reversed(second["tag"]))

    first_key, first_answer = candidate_group_key(first)
    second_key, second_answer = candidate_group_key(second)

    assert first_key == second_key
    assert first_answer == second_answer == "zeta"


def test_audit_reports_broad_and_all_option_candidate_tiers(tmp_path: pathlib.Path):
    train = {}
    validation = {}
    for split_name, destination in (("train", train), ("valid", validation)):
        for answer_id, answer in enumerate(("alpha", "beta", "gamma")):
            for index in range(2):
                video_id = f"{split_name}_{answer}_{index}"
                destination[video_id] = _video(
                    video_id, split_name, _question(answer_id=answer_id)
                )

    duplicate = _question(answer_id=0, options=["same", " same\n", "different"])
    validation["valid_duplicate"] = _video("valid_duplicate", "valid", duplicate)

    two_answer_options = ["left", "right", "unclear"]
    for split_name, destination in (("train", train), ("valid", validation)):
        for answer_id, answer in enumerate(("left", "right")):
            for index in range(2):
                video_id = f"{split_name}_two_{answer}_{index}"
                destination[video_id] = _video(
                    video_id,
                    split_name,
                    _question(answer_id=answer_id, options=two_answer_options),
                )

    train_path = tmp_path / "train.zip"
    validation_path = tmp_path / "validation.zip"
    train_lock = _write_archive(train_path, "mc_question_train.json", train)
    validation_lock = _write_archive(
        validation_path, "mc_question_valid.json", validation
    )
    config_path = tmp_path / "config.json"
    _write_config(config_path, train_lock, validation_lock)

    report = audit_archives(config_path, train_path, validation_path)

    assert report["schema_version"] == 2
    assert report["status"] == "pass"
    assert report["scope"] == "annotation_candidate_pool_feasibility_only"
    assert report["annotation_use_disclosure"]["validation_labels_inspected"] is True
    assert (
        report["annotation_use_disclosure"][
            "validation_media_or_model_outputs_inspected"
        ]
        is False
    )
    assert report["source_attribution"]["materials_license"] == "CC-BY-4.0"
    assert report["splits"]["train"] == {
        "at_least_two_supported_answers": {
            "group_key_count": 2,
            "question_count": 10,
            "unique_video_count": 10,
        },
        "candidate_group_key_count": 2,
        "duplicate_option_exclusion_count": 0,
        "duplicate_option_exclusions": [],
        "raw_question_count": 10,
        "structurally_valid_question_count": 10,
        "video_count": 10,
        "videos_with_structurally_valid_questions": 10,
    }
    assert report["splits"]["validation"]["raw_question_count"] == 11
    assert report["splits"]["validation"]["structurally_valid_question_count"] == 10
    assert report["splits"]["validation"]["duplicate_option_exclusion_count"] == 1
    assert report["splits"]["validation"]["duplicate_option_exclusions"] == [
        {
            "question_id": 0,
            "reason": "duplicate_option_text_after_normalization",
            "video_id": "valid_duplicate",
        }
    ]
    cross = report["cross_split"]
    assert cross["shared_group_key_count"] == 2
    broad = cross["minimum_two_supported_answers"]
    assert broad["group_key_count"] == 2
    assert broad["question_count"] == {"train": 10, "validation": 10}
    assert broad["unique_video_count"] == {"train": 10, "validation": 10}
    assert broad["supported_answer_cardinality"] == {"2": 1, "3": 1}
    strict = cross["all_three_options_supported"]
    assert strict["group_key_count"] == 1
    assert strict["question_count"] == {"train": 6, "validation": 6}
    assert strict["unique_video_count"] == {"train": 6, "validation": 6}
    assert strict["candidate_groups"][0]["option_vocabulary"] == [
        "alpha",
        "beta",
        "gamma",
    ]
    assert strict["candidate_groups"][0]["supported_answer_vocabulary"] == [
        "alpha",
        "beta",
        "gamma",
    ]
    assert strict["candidate_groups"][0]["answer_video_counts"] == {
        "train": {"alpha": 2, "beta": 2, "gamma": 2},
        "validation": {"alpha": 2, "beta": 2, "gamma": 2},
    }
    assert str(tmp_path) not in json.dumps(report)


def test_audit_fails_closed_on_hash_mismatch_before_opening_archive(
    tmp_path: pathlib.Path,
):
    train_path = tmp_path / "train.zip"
    validation_path = tmp_path / "validation.zip"
    train_lock = _write_archive(train_path, "mc_question_train.json", {})
    validation_lock = _write_archive(validation_path, "mc_question_valid.json", {})
    config_path = tmp_path / "config.json"
    _write_config(config_path, train_lock, validation_lock)
    train_path.write_bytes(b"not the pinned archive")

    with pytest.raises(AuditError, match="train archive size does not match"):
        audit_archives(config_path, train_path, validation_path)


def test_audit_fails_closed_on_unknown_schema_field(tmp_path: pathlib.Path):
    train = {"train_0": _video("train_0", "train", _question(answer_id=0))}
    train["train_0"]["metadata"]["unexpected"] = "value"
    validation = {"valid_0": _video("valid_0", "valid", _question(answer_id=0))}
    train_path = tmp_path / "train.zip"
    validation_path = tmp_path / "validation.zip"
    train_lock = _write_archive(train_path, "mc_question_train.json", train)
    validation_lock = _write_archive(
        validation_path, "mc_question_valid.json", validation
    )
    config_path = tmp_path / "config.json"
    _write_config(config_path, train_lock, validation_lock, minimum=1)

    with pytest.raises(AuditError, match="metadata.*keys"):
        audit_archives(config_path, train_path, validation_path)


def test_audit_fails_closed_on_cross_split_video_id_overlap(tmp_path: pathlib.Path):
    train = {"shared_video": _video("shared_video", "train", _question(answer_id=0))}
    validation = {
        "shared_video": _video("shared_video", "valid", _question(answer_id=0))
    }
    train_path = tmp_path / "train.zip"
    validation_path = tmp_path / "validation.zip"
    train_lock = _write_archive(train_path, "mc_question_train.json", train)
    validation_lock = _write_archive(
        validation_path, "mc_question_valid.json", validation
    )
    config_path = tmp_path / "config.json"
    _write_config(config_path, train_lock, validation_lock, minimum=1)

    with pytest.raises(AuditError, match="video IDs overlap.*shared_video"):
        audit_archives(config_path, train_path, validation_path)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        (
            "repository_url",
            "https://github.com/deepmind/perception_test",
            "source_repository.url is not the canonical official URL",
        ),
        (
            "repository_commit",
            "0" * 40,
            "source_repository.commit is not the pinned official commit",
        ),
        (
            "train_url",
            "https://storage.googleapis.com/dm-perception-test/zip_data/mc_question_train_annotations.zip?generation=1",
            "archives.train.url is not the pinned official URL",
        ),
        (
            "validation_generation",
            "1",
            "archives.validation.generation is not the pinned official generation",
        ),
    ],
)
def test_audit_rejects_noncanonical_source_identity(
    tmp_path: pathlib.Path, field: str, replacement: str, message: str
):
    config_path = tmp_path / "config.json"
    _write_config(config_path, (1, "0" * 64), (1, "1" * 64))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if field == "repository_url":
        config["source_repository"]["url"] = replacement
    elif field == "repository_commit":
        config["source_repository"]["commit"] = replacement
    elif field == "train_url":
        config["archives"]["train"]["url"] = replacement
    else:
        config["archives"]["validation"]["generation"] = replacement
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(AuditError, match=message):
        audit_archives(config_path, tmp_path / "train.zip", tmp_path / "validation.zip")


def test_audit_fails_when_pinned_snapshot_counts_do_not_match(tmp_path: pathlib.Path):
    train = {}
    validation = {}
    for split_name, destination in (("train", train), ("valid", validation)):
        for answer_id, answer in enumerate(("alpha", "beta", "gamma")):
            for index in range(2):
                video_id = f"{split_name}_{answer}_{index}"
                destination[video_id] = _video(
                    video_id, split_name, _question(answer_id=answer_id)
                )
    train_path = tmp_path / "train.zip"
    validation_path = tmp_path / "validation.zip"
    train_lock = _write_archive(train_path, "mc_question_train.json", train)
    validation_lock = _write_archive(
        validation_path, "mc_question_valid.json", validation
    )
    config_path = tmp_path / "config.json"
    incorrect = {
        "split_minimum_two_answer_group_count": {"train": 99, "validation": 1},
        "cross_split_minimum_two_answer_group_count": 1,
        "cross_split_all_three_option_group_count": 1,
        "duplicate_option_exclusion_count": {"train": 0, "validation": 0},
    }
    _write_config(
        config_path,
        train_lock,
        validation_lock,
        minimum=2,
        expected_counts=incorrect,
    )

    with pytest.raises(AuditError, match="snapshot counts differ"):
        audit_archives(config_path, train_path, validation_path)


def test_audit_fails_closed_on_test_split_configuration(tmp_path: pathlib.Path):
    config_path = tmp_path / "config.json"
    _write_config(config_path, (1, "0" * 64), (1, "1" * 64))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["archives"]["test"] = copy.deepcopy(config["archives"]["validation"])
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(AuditError, match="exactly train and validation"):
        audit_archives(config_path, tmp_path / "train.zip", tmp_path / "validation.zip")


def test_audit_fails_closed_on_duplicate_json_key(tmp_path: pathlib.Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"schema_version": 1, "schema_version": 1}', encoding="utf-8"
    )

    with pytest.raises(AuditError, match="duplicate JSON key"):
        audit_archives(config_path, tmp_path / "train.zip", tmp_path / "validation.zip")


def test_cli_failure_does_not_create_output(tmp_path: pathlib.Path):
    config_path = tmp_path / "config.json"
    _write_config(config_path, (1, "0" * 64), (1, "1" * 64))
    output_path = tmp_path / "result.json"

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--config",
                str(config_path),
                "--train-archive",
                str(tmp_path / "missing-train.zip"),
                "--validation-archive",
                str(tmp_path / "missing-validation.zip"),
                "--output",
                str(output_path),
            ]
        )

    assert exc_info.value.code == 2
    assert not output_path.exists()
