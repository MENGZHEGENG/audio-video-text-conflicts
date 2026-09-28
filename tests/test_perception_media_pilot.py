from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import zipfile

import conflictbench.perception_media_pilot as pilot_module
import pytest
from conflictbench.perception_media_pilot import (
    PilotConstructionError,
    _collect_eligible_entries,
    _entry_from_question,
    _maximum_unique_assignment,
    build_media_pilot_index,
)

_TRAIN_URL = (
    "https://storage.googleapis.com/dm-perception-test/zip_data/"
    "mc_question_train_annotations.zip?generation=1686575054862298"
)
_VALIDATION_URL = (
    "https://storage.googleapis.com/dm-perception-test/zip_data/"
    "mc_question_valid_annotations.zip?generation=1686575054970958"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _question(question_id: int, answer_id: int, question: str) -> dict:
    return {
        "id": question_id,
        "question": question,
        "options": ["alpha", "beta", "unused"],
        "answer_id": answer_id,
        "area": "semantics",
        "reasoning": "descriptive",
        "tag": ["event", "ordering"],
    }


def _video(video_id: str, question: dict) -> dict:
    return {
        "metadata": {
            "split": "train",
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


def _write_train_archive(path: pathlib.Path) -> tuple[dict, dict]:
    payload: dict[str, dict] = {}
    group_records: list[dict] = []
    for group_index in range(2):
        question = f"Which event happened in group {group_index}?"
        for answer_id, answer in enumerate(("alpha", "beta")):
            for video_index in range(75):
                video_id = f"train_g{group_index}_{answer}_{video_index:03d}"
                payload[video_id] = _video(
                    video_id,
                    _question(0, answer_id, question),
                )
        key_record = {
            "area": "semantics",
            "option_vocabulary": ["alpha", "beta", "unused"],
            "question": question,
            "reasoning": "descriptive",
            "tags": ["event", "ordering"],
        }
        key_bytes = json.dumps(
            key_record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        group_records.append(
            {
                **key_record,
                "answer_video_counts": {
                    "train": {"alpha": 75, "beta": 75, "unused": 0},
                    "validation": {"alpha": 5, "beta": 5, "unused": 0},
                },
                "key_sha256": _sha256(key_bytes),
                "question_counts": {"train": 150, "validation": 10},
                "supported_answer_vocabulary": ["alpha", "beta"],
                "unique_video_counts": {"train": 150, "validation": 10},
            }
        )

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_BZIP2) as archive:
        archive.writestr("mc_question_train.json", encoded)
    archive_bytes = path.read_bytes()
    return {
        "generation": "1686575054862298",
        "member": "mc_question_train.json",
        "metadata_split": "train",
        "sha256": _sha256(archive_bytes),
        "size_bytes": len(archive_bytes),
        "url": _TRAIN_URL,
    }, {"payload": payload, "candidate_groups": group_records}


def _write_inputs(tmp_path: pathlib.Path) -> dict[str, pathlib.Path | str]:
    train_path = tmp_path / "train.zip"
    train_lock, fixture = _write_train_archive(train_path)

    # Deliberately not a ZIP: the builder may authenticate this sealed input,
    # but must not open validation annotations for a training-only pilot.
    validation_path = tmp_path / "validation.sealed"
    validation_bytes = b"sealed validation annotation archive bytes"
    validation_path.write_bytes(validation_bytes)
    validation_lock = {
        "generation": "1686575054970958",
        "member": "mc_question_valid.json",
        "metadata_split": "valid",
        "sha256": _sha256(validation_bytes),
        "size_bytes": len(validation_bytes),
        "url": _VALIDATION_URL,
    }

    audit_configuration_sha256 = "a" * 64
    candidate_groups = fixture["candidate_groups"]
    audit = {
        "annotation_use_disclosure": {
            "test_annotations_inspected": False,
            "train_labels_inspected": True,
            "validation_labels_inspected": True,
            "validation_media_or_model_outputs_inspected": False,
            "validation_use": "schema_and_candidate_pool_counts_only",
        },
        "configuration_sha256": audit_configuration_sha256,
        "cross_split": {
            "all_three_options_supported": {
                "candidate_groups": [],
                "group_key_count": 0,
                "question_count": {"train": 0, "validation": 0},
                "supported_answer_cardinality": {},
                "unique_video_count": {"train": 0, "validation": 0},
            },
            "minimum_two_supported_answers": {
                "candidate_groups": candidate_groups,
                "group_key_count": 2,
                "question_count": {"train": 300, "validation": 20},
                "supported_answer_cardinality": {"2": 2},
                "unique_video_count": {"train": 300, "validation": 20},
            },
            "shared_group_key_count": 2,
        },
        "eligibility": {
            "min_supported_answers_per_group": 2,
            "min_unique_videos_per_supported_answer_per_split": 5,
            "report_all_three_option_tier": True,
            "require_same_supported_answer_vocabulary_across_splits": True,
        },
        "limitations": [
            "Structural annotation counts do not prove audio or video sufficiency.",
            "Validation labels were inspected for structural counts, so validation annotations are not fully unseen.",
        ],
        "normalization": {
            "case_sensitive": True,
            "punctuation_sensitive": True,
            "unicode": "NFC",
            "whitespace": "strip_and_collapse",
        },
        "schema_version": 2,
        "scope": "annotation_candidate_pool_feasibility_only",
        "snapshot_counts": {
            "cross_split_all_three_option_group_count": 0,
            "cross_split_minimum_two_answer_group_count": 2,
            "duplicate_option_exclusion_count": {"train": 0, "validation": 0},
            "split_minimum_two_answer_group_count": {
                "train": 2,
                "validation": 2,
            },
        },
        "source_archives": {
            "train": train_lock,
            "validation": validation_lock,
        },
        "source_attribution": {
            "authors": "Pătrăucean et al. (2023)",
            "copyright": "Copyright 2022 DeepMind Technologies Limited",
            "dataset": "Perception Test: A Diagnostic Benchmark for Multimodal Video Models",
            "license_url": "https://creativecommons.org/licenses/by/4.0/legalcode",
            "materials_license": "CC-BY-4.0",
        },
        "source_repository": {
            "annotation_license": "CC-BY-4.0",
            "commit": "3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf",
            "license_url": "https://creativecommons.org/licenses/by/4.0/legalcode",
            "url": "https://github.com/google-deepmind/perception_test",
        },
        "splits": {
            "train": {
                "at_least_two_supported_answers": {
                    "group_key_count": 2,
                    "question_count": 300,
                    "unique_video_count": 300,
                },
                "candidate_group_key_count": 2,
                "duplicate_option_exclusion_count": 0,
                "duplicate_option_exclusions": [],
                "raw_question_count": 300,
                "structurally_valid_question_count": 300,
                "video_count": 300,
                "videos_with_structurally_valid_questions": 300,
            },
            "validation": {
                "at_least_two_supported_answers": {
                    "group_key_count": 2,
                    "question_count": 20,
                    "unique_video_count": 20,
                },
                "candidate_group_key_count": 2,
                "duplicate_option_exclusion_count": 0,
                "duplicate_option_exclusions": [],
                "raw_question_count": 20,
                "structurally_valid_question_count": 20,
                "video_count": 20,
                "videos_with_structurally_valid_questions": 20,
            },
        },
        "status": "pass",
    }
    audit_path = tmp_path / "audit.json"
    audit_bytes = (
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    audit_path.write_bytes(audit_bytes)

    config = {
        "builder": "perception_test_training_media_pilot",
        "construction": {
            "component_partition_lock": True,
            "min_supported_answers_per_key": 2,
            "min_unique_train_videos_per_supported_answer": 5,
            "official_split": "train",
            "pair_roles": [
                "same_answer_nuisance",
                "opposite_answer_candidate",
            ],
            "prefer_unique_donors": True,
            "require_every_eligible_key": True,
            "target_and_donor_video_disjoint_when_possible": True,
        },
        "counts": {
            "expected_eligible_key_count": 2,
            "expected_eligible_key_count_by_supported_answer_count": {
                "2": 2,
                "3": 0,
            },
            "pair_count": 200,
            "pair_count_bounds": [200, 500],
            "partition_target_counts": {
                "pilot_gate": 20,
                "scorer_fit": 60,
                "threshold_calibration": 20,
            },
            "target_count": 100,
        },
        "gate_contract": {
            "evaluation_partition": "pilot_gate",
            "fit_partition": "scorer_fit",
            "nuisance_detection_required": True,
            "required_modalities": ["audio", "video"],
            "scores_in_builder_forbidden": True,
            "source_answer_scoring_required": True,
            "status": "not_run",
            "threshold_partition": "threshold_calibration",
            "thresholds_must_be_frozen_before_evaluation": True,
            "version": 1,
        },
        "implementation_sha256": pilot_module.implementation_sha256(),
        "input_locks": {
            "audit_configuration_sha256": audit_configuration_sha256,
            "train_archive": train_lock,
            "validation_archive": validation_lock,
        },
        "purpose": "training_only_source_sufficiency_construction",
        "schema_version": 1,
        "seed": 20270917,
        "source": {
            "annotation_license": "CC-BY-4.0",
            "citation": "Pătrăucean et al. (2023), Perception Test",
            "license_url": "https://creativecommons.org/licenses/by/4.0/legalcode",
            "repository_commit": "3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf",
            "repository_url": "https://github.com/google-deepmind/perception_test",
        },
    }
    config_path = tmp_path / "pilot.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "audit": audit_path,
        "audit_sha256": _sha256(audit_bytes),
        "config": config_path,
        "train": train_path,
        "validation": validation_path,
    }


def _build(inputs: dict[str, pathlib.Path | str]) -> dict:
    return build_media_pilot_index(
        config_path=pathlib.Path(inputs["config"]),
        structural_audit_path=pathlib.Path(inputs["audit"]),
        expected_structural_audit_sha256=str(inputs["audit_sha256"]),
        train_archive=pathlib.Path(inputs["train"]),
        validation_archive=pathlib.Path(inputs["validation"]),
    )


def test_builds_exact_deterministic_training_only_candidate_index(
    tmp_path: pathlib.Path,
):
    inputs = _write_inputs(tmp_path)

    first = _build(inputs)
    second = _build(inputs)

    assert first == second
    assert first["status"] == "construction_complete_scores_not_run"
    assert first["scope"] == "train_candidate_index_from_authenticated_train_labels"
    assert first["annotation_use_disclosure"] == {
        "builder_test_annotations_loaded": False,
        "builder_train_labels_loaded": True,
        "builder_validation_labels_loaded": False,
        "eligible_keys_derived_from_authenticated_train_archive": True,
        "structural_audit_authenticated": True,
        "structural_audit_train_labels_inspected": True,
        "structural_audit_used_for_candidate_selection": False,
        "structural_audit_validation_labels_inspected": True,
        "structural_audit_validation_use": "schema_and_candidate_pool_counts_only",
        "train_eligibility_min_supported_answers": 2,
        "train_eligibility_min_unique_videos_per_supported_answer": 5,
    }
    assert first["counts"] == {
        "component_count": 100,
        "eligible_key_count": 2,
        "eligible_key_count_by_supported_answer_count": {"2": 2, "3": 0},
        "exclusion_count": 0,
        "opposite_answer_pair_count": 100,
        "pair_count": 200,
        "same_answer_pair_count": 100,
        "target_count": 100,
        "unique_donor_count": 200,
        "unique_target_video_count": 100,
    }
    assert first["source_use"] == {
        "media_loaded": False,
        "model_outputs_loaded": False,
        "official_splits_in_candidate_index": ["train"],
        "builder_test_annotations_loaded": False,
        "builder_validation_annotations_loaded": False,
        "validation_archive_sha256_verified": True,
    }
    assert first["gate_contract"]["status"] == "not_run"
    assert first["gate_contract"]["required_modalities"] == ["audio", "video"]
    assert first["gate_contract"]["required_later_records"] == {
        "nuisance_detection": [
            "component_id",
            "pair_id",
            "partition",
            "predicted_role",
            "role_scores",
            "detector_id",
            "detector_config_sha256",
        ],
        "source_answer_scoring": [
            "component_id",
            "pair_id",
            "partition",
            "source_role",
            "modality",
            "predicted_answer",
            "answer_scores",
            "answer_margin",
            "scorer_id",
            "scorer_config_sha256",
        ],
    }

    pairs = first["candidate_index"]
    assert len(pairs) == 200
    by_target: dict[str, list[dict]] = {}
    for pair in pairs:
        assert pair["official_split"] == "train"
        by_target.setdefault(pair["target"]["video_id"], []).append(pair)
    assert len(by_target) == 100
    assert {len(target_pairs) for target_pairs in by_target.values()} == {2}

    donor_ids: list[str] = []
    target_ids = set(by_target)
    for target_pairs in by_target.values():
        roles = {pair["role"] for pair in target_pairs}
        assert roles == {"same_answer_nuisance", "opposite_answer_candidate"}
        first_target = target_pairs[0]["target"]
        assert target_pairs[1]["target"] == first_target
        for pair in target_pairs:
            donor_ids.append(pair["donor"]["video_id"])
            assert pair["donor"]["question"] == first_target["question"]
            assert pair["donor"]["options"] == first_target["options"]
            if pair["role"] == "same_answer_nuisance":
                assert pair["donor"]["answer"] == first_target["answer"]
            else:
                assert pair["donor"]["answer"] != first_target["answer"]
    assert len(set(donor_ids)) == 200
    assert set(donor_ids).isdisjoint(target_ids)

    assert first["partition_counts"] == {
        "pilot_gate": {"component_count": 20, "pair_count": 40, "target_count": 20},
        "scorer_fit": {"component_count": 60, "pair_count": 120, "target_count": 60},
        "threshold_calibration": {
            "component_count": 20,
            "pair_count": 40,
            "target_count": 20,
        },
    }
    component_partitions: dict[str, set[str]] = {}
    for pair in pairs:
        component_partitions.setdefault(pair["component_id"], set()).add(
            pair["partition"]
        )
    assert all(len(partitions) == 1 for partitions in component_partitions.values())

    serialized = json.dumps(first, ensure_ascii=False)
    for forbidden in (".mp4", "/tmp/", "answer_probability", "model_score"):
        assert forbidden not in serialized
    assert first["source_attribution"]["materials_license"] == "CC-BY-4.0"
    assert "No media was copied or modified" in first["change_note"]
    provenance = pilot_module.implementation_provenance()
    assert first["implementation_sha256"] == provenance["implementation_sha256"]
    assert first["implementation_source_sha256"] == provenance["source_sha256"]
    pilot_module.validate_media_pilot_attestation(first)
    unsigned = dict(first)
    attestation_sha256 = unsigned.pop("attestation_sha256")
    encoded = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert attestation_sha256 == _sha256(encoded)


def test_rejects_wrong_structural_audit_digest_before_construction(
    tmp_path: pathlib.Path,
):
    inputs = _write_inputs(tmp_path)
    inputs["audit_sha256"] = "0" * 64

    with pytest.raises(PilotConstructionError, match="structural audit SHA-256"):
        _build(inputs)


def test_rejects_unknown_audit_fields_instead_of_accepting_outcomes(
    tmp_path: pathlib.Path,
):
    inputs = _write_inputs(tmp_path)
    audit_path = pathlib.Path(inputs["audit"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["model_scores"] = {"peeked": True}
    audit_bytes = (
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    audit_path.write_bytes(audit_bytes)
    inputs["audit_sha256"] = _sha256(audit_bytes)

    with pytest.raises(PilotConstructionError, match="structural audit keys differ"):
        _build(inputs)


def test_rejects_validation_archive_hash_drift_without_opening_it(
    tmp_path: pathlib.Path,
):
    inputs = _write_inputs(tmp_path)
    validation_path = pathlib.Path(inputs["validation"])
    validation_path.write_bytes(validation_path.read_bytes() + b"drift")

    with pytest.raises(PilotConstructionError, match="validation archive size"):
        _build(inputs)


def test_rejects_configuration_outside_the_bounded_pair_range(
    tmp_path: pathlib.Path,
):
    inputs = _write_inputs(tmp_path)
    config_path = pathlib.Path(inputs["config"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["counts"]["pair_count"] = 198
    config["counts"]["target_count"] = 99
    config["counts"]["partition_target_counts"]["scorer_fit"] = 59
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(PilotConstructionError, match="between 200 and 500"):
        _build(inputs)


def test_rejects_configuration_with_a_different_implementation_digest(
    tmp_path: pathlib.Path,
):
    inputs = _write_inputs(tmp_path)
    config_path = pathlib.Path(inputs["config"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["implementation_sha256"] = "0" * 64
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(PilotConstructionError, match="implementation SHA-256"):
        _build(inputs)


def test_unique_donor_matching_reassigns_an_earlier_flexible_request():
    requests = [
        ("flexible", ["only", "spare"]),
        ("constrained", ["only"]),
    ]

    assignment = _maximum_unique_assignment(requests)

    assert assignment == {"constrained": "only", "flexible": "spare"}


def test_train_only_eligibility_requires_two_answers_with_five_unique_videos():
    payload: dict[str, dict] = {}
    specifications = (
        ("eligible", "alpha", 0, 5),
        ("eligible", "beta", 1, 5),
        ("eligible", "unused", 2, 4),
        ("ineligible", "alpha", 0, 5),
        ("ineligible", "beta", 1, 4),
    )
    for group, answer, answer_id, count in specifications:
        question = f"Which event happened in {group}?"
        for index in range(count):
            video_id = f"{group}_{answer}_{index}"
            payload[video_id] = _video(
                video_id,
                _question(0, answer_id, question),
            )

    eligible, exclusions = _collect_eligible_entries(
        payload,
        min_supported_answers=2,
        min_unique_videos_per_supported_answer=5,
    )

    assert exclusions == []
    assert len(eligible) == 1
    assert set(next(iter(eligible.values()))) == {"alpha", "beta"}


def test_question_ids_reject_numeric_values_that_are_not_integers():
    question = _question(0, 0, "Which event happened?")
    question["id"] = 0.0

    with pytest.raises(PilotConstructionError, match="question IDs"):
        _entry_from_question("train_video", 0, question)


def test_rejects_a_consistently_rewritten_noncanonical_archive_identity(
    tmp_path: pathlib.Path,
):
    inputs = _write_inputs(tmp_path)
    config_path = pathlib.Path(inputs["config"])
    audit_path = pathlib.Path(inputs["audit"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    rewritten_url = "https://example.org/train.zip"
    config["input_locks"]["train_archive"]["url"] = rewritten_url
    audit["source_archives"]["train"]["url"] = rewritten_url
    config_path.write_text(json.dumps(config), encoding="utf-8")
    audit_bytes = json.dumps(audit, sort_keys=True).encode("utf-8")
    audit_path.write_bytes(audit_bytes)
    inputs["audit_sha256"] = _sha256(audit_bytes)

    with pytest.raises(PilotConstructionError, match="pinned official identity"):
        _build(inputs)


def test_implementation_digest_changes_when_either_covered_source_changes(
    tmp_path: pathlib.Path,
):
    builder_path = tmp_path / "perception_media_pilot.py"
    audit_path = tmp_path / "perception_candidate_audit.py"
    builder_path.write_bytes(b"builder implementation\n")
    audit_path.write_bytes(b"candidate audit implementation\n")
    source_paths = {
        "conflictbench.perception_candidate_audit": audit_path,
        "conflictbench.perception_media_pilot": builder_path,
    }

    baseline = pilot_module._implementation_provenance_for_paths(source_paths)
    builder_path.write_bytes(b"changed builder implementation\n")
    builder_changed = pilot_module._implementation_provenance_for_paths(source_paths)
    builder_path.write_bytes(b"builder implementation\n")
    audit_path.write_bytes(b"changed candidate audit implementation\n")
    audit_changed = pilot_module._implementation_provenance_for_paths(source_paths)

    assert baseline["implementation_sha256"] != builder_changed[
        "implementation_sha256"
    ]
    assert baseline["implementation_sha256"] != audit_changed[
        "implementation_sha256"
    ]
    assert (
        baseline["source_sha256"]["conflictbench.perception_candidate_audit"]
        == builder_changed["source_sha256"][
            "conflictbench.perception_candidate_audit"
        ]
    )
    assert (
        baseline["source_sha256"]["conflictbench.perception_media_pilot"]
        == audit_changed["source_sha256"]["conflictbench.perception_media_pilot"]
    )


def test_output_attestation_rejects_content_and_implementation_tampering(
    tmp_path: pathlib.Path,
):
    result = _build(_write_inputs(tmp_path))

    content_tamper = copy.deepcopy(result)
    content_tamper["scope"] = "forged_scope"
    with pytest.raises(PilotConstructionError, match="attestation SHA-256"):
        pilot_module.validate_media_pilot_attestation(content_tamper)

    implementation_tamper = copy.deepcopy(result)
    implementation_tamper["implementation_sha256"] = "0" * 64
    unsigned = dict(implementation_tamper)
    unsigned.pop("attestation_sha256")
    implementation_tamper["attestation_sha256"] = _sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    with pytest.raises(PilotConstructionError, match="implementation SHA-256"):
        pilot_module.validate_media_pilot_attestation(implementation_tamper)


def test_validation_dependent_audit_membership_cannot_change_candidate_selection(
    tmp_path: pathlib.Path,
):
    inputs = _write_inputs(tmp_path)
    baseline = _build(inputs)
    audit_path = pathlib.Path(inputs["audit"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    tier = audit["cross_split"]["minimum_two_supported_answers"]
    retained = tier["candidate_groups"][:1]
    tier["candidate_groups"] = retained
    tier["group_key_count"] = 1
    tier["question_count"] = {"train": 150, "validation": 10}
    tier["supported_answer_cardinality"] = {"2": 1}
    tier["unique_video_count"] = {"train": 150, "validation": 10}
    audit["cross_split"]["shared_group_key_count"] = 1
    audit["snapshot_counts"]["cross_split_minimum_two_answer_group_count"] = 1
    audit_bytes = (
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    audit_path.write_bytes(audit_bytes)
    inputs["audit_sha256"] = _sha256(audit_bytes)

    changed_audit = _build(inputs)

    for field in (
        "candidate_index",
        "components",
        "counts",
        "donor_reuse_events",
        "exclusions",
        "partition_counts",
        "strata",
    ):
        assert changed_audit[field] == baseline[field]
    assert changed_audit["input_digests"]["structural_audit_sha256"] != baseline[
        "input_digests"
    ]["structural_audit_sha256"]


def test_output_writer_refuses_to_replace_an_existing_file(
    tmp_path: pathlib.Path,
):
    output = tmp_path / "pilot-index.json"
    pilot_module._write_json_atomic(output, {"status": "first"})

    with pytest.raises(PilotConstructionError, match="output already exists"):
        pilot_module._write_json_atomic(output, {"status": "second"})

    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "first"}
