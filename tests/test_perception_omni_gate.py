from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import json
import pathlib
import re
import subprocess
import types

import pytest

import conflictbench.perception_omni_gate as gate_module
from conflictbench.perception_media_pilot import (
    _canonical_digest,
    _expanded_gate_contract,
    implementation_provenance,
)
from conflictbench.perception_omni_gate import (
    MODEL_ID,
    MODEL_REVISION,
    GateValidationError,
    InferenceRequest,
    InferenceResult,
    QwenOmniBackend,
    parse_choice,
    run_source_sufficiency_gate,
    validate_gate_output,
    write_gate_output,
)

_REAL_PROBE_MEDIA_STREAMS = gate_module._probe_media_streams


@pytest.fixture(autouse=True)
def _avoid_real_ffprobe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gate_module,
        "_probe_media_streams",
        lambda _path: ["audio", "video"],
    )


@pytest.mark.parametrize("text, expected", [("A", "A"), (" B\n", "B"), ("C", "C")])
def test_parse_choice_accepts_only_one_trimmed_uppercase_label(
    text: str, expected: str
) -> None:
    assert parse_choice(text) == expected


@pytest.mark.parametrize("text", ["", "a", "A.", "Answer: A", "A B", "D"])
def test_parse_choice_rejects_noncanonical_responses(text: str) -> None:
    with pytest.raises(GateValidationError, match="exactly one of A, B, or C"):
        parse_choice(text)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest24(value: object) -> str:
    return _sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )[:24]


def _write_json(path: pathlib.Path, value: dict) -> str:
    data = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    path.write_bytes(data)
    return _sha256(data)


def _partition(target_index: int) -> str:
    if target_index < 60:
        return "scorer_fit"
    if target_index < 80:
        return "threshold_calibration"
    return "pilot_gate"


def _source(video_id: str, key_index: int, answer_id: int) -> dict:
    options = [f"key {key_index} answer A", f"key {key_index} answer B", "neither"]
    return {
        "video_id": video_id,
        "question_id": key_index,
        "question": f"Which event happened for key {key_index}?",
        "options": options,
        "answer_id": answer_id,
        "answer": options[answer_id],
    }


def _pilot_gate_contract() -> dict:
    return _expanded_gate_contract(
        {
            "version": 1,
            "status": "not_run",
            "fit_partition": "scorer_fit",
            "threshold_partition": "threshold_calibration",
            "evaluation_partition": "pilot_gate",
            "required_modalities": ["audio", "video"],
            "source_answer_scoring_required": True,
            "nuisance_detection_required": True,
            "thresholds_must_be_frozen_before_evaluation": True,
            "scores_in_builder_forbidden": True,
        }
    )


def _reattest(pilot: dict) -> dict:
    pilot.pop("attestation_sha256", None)
    provenance = implementation_provenance()
    pilot["implementation_sha256"] = provenance["implementation_sha256"]
    pilot["implementation_source_sha256"] = provenance["source_sha256"]
    pilot["attestation_sha256"] = _canonical_digest(pilot)
    return pilot


def _pilot_index() -> dict:
    pairs: list[dict] = []
    components: list[dict] = []
    key_hashes = [_sha256(f"key:{index}".encode()) for index in range(85)]
    for target_index in range(100):
        key_index = target_index % len(key_hashes)
        target_answer_id = target_index % 3
        target = _source(f"target_{target_index:03d}", key_index, target_answer_id)
        partition = _partition(target_index)
        donors = (
            (
                "same_answer_nuisance",
                _source(f"donor_same_{target_index:03d}", key_index, target_answer_id),
            ),
            (
                "opposite_answer_candidate",
                _source(
                    f"donor_opposite_{target_index:03d}",
                    key_index,
                    (target_answer_id + 1) % 3,
                ),
            ),
        )
        video_ids = sorted(
            [target["video_id"], *(donor[1]["video_id"] for donor in donors)]
        )
        component_id = _digest24(video_ids)
        components.append(
            {
                "component_id": component_id,
                "partition": partition,
                "target_count": 1,
                "video_ids": video_ids,
            }
        )
        for role, donor in donors:
            pairs.append(
                {
                    "pair_id": _digest24(
                        [20270917, target["video_id"], donor["video_id"], role]
                    ),
                    "component_id": component_id,
                    "official_split": "train",
                    "partition": partition,
                    "key_sha256": key_hashes[key_index],
                    "role": role,
                    "anchor": {
                        "question": target["question"],
                        "options": target["options"],
                        "answer_id": target["answer_id"],
                        "answer": target["answer"],
                    },
                    "target": copy.deepcopy(target),
                    "donor": donor,
                }
            )
    pairs.sort(
        key=lambda pair: (pair["partition"], pair["component_id"], pair["pair_id"])
    )
    components.sort(key=lambda item: (item["partition"], item["component_id"]))
    partition_counts = {
        partition: {
            "component_count": count,
            "pair_count": count * 2,
            "target_count": count,
        }
        for partition, count in {
            "scorer_fit": 60,
            "threshold_calibration": 20,
            "pilot_gate": 20,
        }.items()
    }
    pilot = {
        "schema_version": 1,
        "status": "construction_complete_scores_not_run",
        "scope": "train_candidate_index_from_authenticated_train_labels",
        "seed": 20270917,
        "implementation_sha256": "0" * 64,
        "implementation_source_sha256": {
            "conflictbench.perception_candidate_audit": "0" * 64,
            "conflictbench.perception_media_pilot": "0" * 64,
        },
        "input_digests": {
            "configuration_sha256": "2" * 64,
            "structural_audit_sha256": "3" * 64,
            "structural_audit_configuration_sha256": "4" * 64,
            "train_archive_sha256": "5" * 64,
            "validation_archive_sha256": "6" * 64,
        },
        "source_use": {
            "official_splits_in_candidate_index": ["train"],
            "validation_archive_sha256_verified": True,
            "builder_validation_annotations_loaded": False,
            "builder_test_annotations_loaded": False,
            "media_loaded": False,
            "model_outputs_loaded": False,
        },
        "annotation_use_disclosure": {
            "eligible_keys_derived_from_authenticated_train_archive": True,
            "train_eligibility_min_supported_answers": 2,
            "train_eligibility_min_unique_videos_per_supported_answer": 5,
            "structural_audit_authenticated": True,
            "structural_audit_used_for_candidate_selection": False,
            "structural_audit_train_labels_inspected": True,
            "structural_audit_validation_labels_inspected": True,
            "structural_audit_validation_use": "schema_and_candidate_pool_counts_only",
            "builder_train_labels_loaded": True,
            "builder_validation_labels_loaded": False,
            "builder_test_annotations_loaded": False,
        },
        "source_attribution": {
            "dataset": "Perception Test",
            "authors": "Patraucean et al.",
            "copyright": "Perception Test authors",
            "materials_license": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/legalcode",
        },
        "change_note": (
            "Training-only deterministic annotation selection; media is unchanged. "
            "Derived materials retain the CC-BY-4.0 notice."
        ),
        "construction_rules": {
            "official_split": "train",
            "min_supported_answers_per_key": 2,
            "min_unique_train_videos_per_supported_answer": 5,
            "pair_roles": [
                "same_answer_nuisance",
                "opposite_answer_candidate",
            ],
            "require_every_eligible_key": True,
            "prefer_unique_donors": True,
            "target_and_donor_video_disjoint_when_possible": True,
            "component_partition_lock": True,
        },
        "counts": {
            "eligible_key_count": 85,
            "eligible_key_count_by_supported_answer_count": {"2": 55, "3": 30},
            "target_count": 100,
            "pair_count": 200,
            "same_answer_pair_count": 100,
            "opposite_answer_pair_count": 100,
            "unique_target_video_count": 100,
            "unique_donor_count": 200,
            "component_count": 100,
            "exclusion_count": 0,
        },
        "partition_counts": partition_counts,
        "strata": [],
        "exclusions": [],
        "donor_reuse_events": [],
        "components": components,
        "gate_contract": _pilot_gate_contract(),
        "candidate_index": pairs,
    }
    return _reattest(pilot)


def _config() -> dict:
    thresholds = {
        "same_answer_nuisance": {
            "audio_only": 0.7,
            "video_only": 0.7,
            "audiovisual": 0.7,
        },
        "opposite_answer_candidate": {
            "audio_only": 0.7,
            "video_only": 0.7,
            "audiovisual": 0.7,
        },
    }
    return {
        "schema_version": 4,
        "runner": "perception_qwen25_omni_source_sufficiency",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "dtype": "float16",
            "attention_implementation": "sdpa",
            "device_map": "auto",
            "disable_talker": True,
        },
        "inference": {
            "batch_size": 1,
            "maximum_new_tokens": 4,
            "do_sample": False,
            "seed": 20270917,
            "conditions": ["audio_only", "video_only", "audiovisual"],
            "emit_choice_scores_when_single_token": True,
            "video_preprocessing": {
                "fps": 2.0,
                "min_frames": 4,
                "max_frames": 32,
                "min_pixels": 100352,
                "max_pixels": 200704,
            },
        },
        "controls": {
            "types": ["question_only", "same_question_shuffled_media"],
            "source_orientations": ["target", "donor"],
            "same_question_shuffle_policy": "paired_opposite_answer_within_component",
            "option_order_policy": "deterministic_balanced_by_partition_and_orientation",
            "option_order_seed": 20270917,
            "outcome_based_cohort_selection": "forbidden",
        },
        "input": {
            "official_split": "train",
            "media_extension": ".mp4",
            "selection_policy": "entire_candidate_index",
            "validation_split_access": "forbidden",
        },
        "gate": {
            "evaluation_partition": "pilot_gate",
            "minimum_strict_parse_rate": 0.95,
            "minimum_unique_target_accuracy_by_condition": {
                "audio_only": 0.7,
                "video_only": 0.7,
                "audiovisual": 0.7,
            },
            "minimum_donor_accuracy_by_role_and_condition": copy.deepcopy(thresholds),
            "minimum_pair_support_rate_by_role_and_condition": copy.deepcopy(
                thresholds
            ),
            "maximum_question_only_source_answer_accuracy_by_orientation": {
                "target": 0.5,
                "donor": 0.5,
            },
            "maximum_shuffled_media_source_answer_accuracy_by_condition_and_orientation": {
                "target": {
                    "audio_only": 0.5,
                    "video_only": 0.5,
                    "audiovisual": 0.5,
                },
                "donor": {
                    "audio_only": 0.5,
                    "video_only": 0.5,
                    "audiovisual": 0.5,
                },
            },
            "minimum_aligned_over_control_margin_by_condition_and_orientation": {
                "target": {
                    "audio_only": 0.2,
                    "video_only": 0.2,
                    "audiovisual": 0.2,
                },
                "donor": {
                    "audio_only": 0.2,
                    "video_only": 0.2,
                    "audiovisual": 0.2,
                },
            },
            "question_key_sensitivity": (
                "pilot_gate_normalized_questions_absent_from_fit_or_threshold"
            ),
            "minimum_unseen_question_component_count": 5,
            "maximum_unseen_question_margin_attenuation": 0.2,
            "require_choice_scores": False,
            "outcome_based_sample_selection": "forbidden",
        },
    }


def _choice_for_video(video_id: str) -> str:
    match = re.search(r"_(\d{3})$", video_id)
    assert match is not None
    target_answer_id = int(match.group(1)) % 3
    if video_id.startswith("donor_opposite_"):
        target_answer_id = (target_answer_id + 1) % 3
    return ("A", "B", "C")[target_answer_id]


def _answer_text_for_video(video_id: str) -> str:
    return {
        "A": "answer A",
        "B": "answer B",
        "C": "neither",
    }[_choice_for_video(video_id)]


def _choice_for_media_in_prompt(video_id: str, prompt: str) -> str:
    answer_suffix = _answer_text_for_video(video_id)
    for line in prompt.splitlines():
        if line[:3] in {"A. ", "B. ", "C. "} and line.endswith(answer_suffix):
            return line[0]
    raise AssertionError("media answer is absent from prompt")


class _FakeBackend:
    def __init__(self, *, omit_last: bool = False, emit_scores: bool = True):
        self.calls: list[list[InferenceRequest]] = []
        self.omit_last = omit_last
        self.emit_scores = emit_scores

    def _response(self, request: InferenceRequest) -> str:
        if request.evaluation == "question_only":
            return "A"
        assert request.media_video_id is not None
        return _choice_for_media_in_prompt(request.media_video_id, request.prompt)

    def infer(self, requests):
        requests = list(requests)
        self.calls.append(requests)
        results = []
        for request in requests:
            choice = self._response(request)
            results.append(
                InferenceResult(
                    record_id=request.record_id,
                    raw_response=choice,
                    choice_scores={"A": 2.0, "B": 1.0, "C": -1.0}
                    if self.emit_scores
                    else None,
                    choice_score_method=(
                        "diagnostic_contextual_first_generated_token_logit"
                    )
                    if self.emit_scores
                    else None,
                    choice_token_ids={"A": 11, "B": 12, "C": 13}
                    if self.emit_scores
                    else None,
                )
            )
        if self.omit_last and results:
            results.pop()
        return results

    def runtime_details(self):
        return {
            "backend": "qwen2_5_omni",
            "loaded": True,
            "torch_version": "2.11.0+cu126",
            "transformers_version": "4.57.6",
            "qwen_omni_utils_version": "0.0.8",
            "cuda_runtime_version": "12.6",
            "gpu_name": "fixture GPU",
            "model_dtype": "torch.float16",
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "frozen": True,
        }


class _MaskedFailureBackend(_FakeBackend):
    def _response(self, request: InferenceRequest) -> str:
        if (
            request.pair_role == "opposite_answer_candidate"
            and request.source_role == "donor"
            and request.video_id.startswith("donor_opposite_")
            and int(request.video_id.rsplit("_", 1)[1]) % 2
        ):
            expected = super()._response(request)
            return {"A": "B", "B": "C", "C": "A"}[expected]
        return super()._response(request)


class _MutatingBackend(_FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.mutated = False

    def infer(self, requests):
        requests = list(requests)
        if requests and not self.mutated:
            requests[0].media_path.write_bytes(b"mutated-media")
            self.mutated = True
        return super().infer(requests)


class _DisagreeingDuplicateTargetBackend(_FakeBackend):
    def _response(self, request: InferenceRequest) -> str:
        if (
            request.pair_role == "opposite_answer_candidate"
            and request.source_role == "target"
        ):
            expected = super()._response(request)
            return {"A": "B", "B": "C", "C": "A"}[expected]
        return super()._response(request)


class _LeakyControlBackend(_FakeBackend):
    def _response(self, request: InferenceRequest) -> str:
        if request.evaluation in {
            "question_only",
            "same_question_shuffled_media",
        }:
            return _choice_for_media_in_prompt(request.video_id, request.prompt)
        return super()._response(request)


class _DonorOppositeControlLeakBackend(_FakeBackend):
    """Leak only the donor opposite-answer control stratum.

    The companion donor stratum returns the counteranswer, so pooling both
    donor roles produces exactly 0.50 source-answer agreement and masks the
    complete leak.
    """

    def _response(self, request: InferenceRequest) -> str:
        if (
            request.evaluation
            in {
                "question_only",
                "same_question_shuffled_media",
            }
            and request.source_role == "donor"
        ):
            if request.pair_role == "opposite_answer_candidate":
                return _choice_for_media_in_prompt(request.video_id, request.prompt)
            if request.evaluation == "question_only":
                expected = _choice_for_media_in_prompt(request.video_id, request.prompt)
                return {"A": "B", "B": "C", "C": "A"}[expected]
        return super()._response(request)


class _ExactTwentyPointMarginBackend(_FakeBackend):
    """Create an exact 14/20 minus 10/20 donor-role margin."""

    def _response(self, request: InferenceRequest) -> str:
        if (
            request.source_role == "donor"
            and request.pair_role == "same_answer_nuisance"
            and request.video_id.startswith("donor_same_")
        ):
            index = int(request.video_id.rsplit("_", 1)[1])
            expected = _choice_for_media_in_prompt(request.video_id, request.prompt)
            wrong = {"A": "B", "B": "C", "C": "A"}[expected]
            if request.evaluation == "source_sufficiency":
                return expected if index % 20 < 14 else wrong
            if request.evaluation == "question_only":
                return expected if index % 20 < 10 else wrong
        return super()._response(request)


class _UnseenQuestionAttenuationBackend(_FakeBackend):
    """Fail aligned media only for the five question keys unseen in fit/calibration."""

    def _response(self, request: InferenceRequest) -> str:
        match = re.search(r"_(\d{3})$", request.video_id)
        if (
            request.evaluation == "source_sufficiency"
            and match is not None
            and 80 <= int(match.group(1)) <= 84
        ):
            expected = super()._response(request)
            return {"A": "B", "B": "C", "C": "A"}[expected]
        return super()._response(request)


def _write_gate_inputs(
    tmp_path: pathlib.Path,
    *,
    config: dict | None = None,
    pilot: dict | None = None,
) -> tuple[pathlib.Path, str, pathlib.Path, str, pathlib.Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.json"
    pilot_path = tmp_path / "pilot-index.json"
    media_root = tmp_path / "media"
    media_root.mkdir()
    config_sha256 = _write_json(config_path, config or _config())
    pilot_value = pilot or _pilot_index()
    pilot_sha256 = _write_json(pilot_path, pilot_value)
    video_ids = sorted(
        {
            pair[source_role]["video_id"]
            for pair in pilot_value["candidate_index"]
            for source_role in ("target", "donor")
        }
    )
    for video_id in video_ids:
        (media_root / f"{video_id}.mp4").write_bytes(f"fake:{video_id}".encode())
    return config_path, config_sha256, pilot_path, pilot_sha256, media_root


def _run(
    tmp_path: pathlib.Path,
    backend: _FakeBackend,
    *,
    config: dict | None = None,
    pilot: dict | None = None,
) -> dict:
    config_path, config_sha256, pilot_path, pilot_sha256, media_root = (
        _write_gate_inputs(tmp_path, config=config, pilot=pilot)
    )
    return run_source_sufficiency_gate(
        config_path=config_path,
        expected_config_sha256=config_sha256,
        pilot_index_path=pilot_path,
        expected_pilot_index_sha256=pilot_sha256,
        media_root=media_root,
        backend=backend,
        source_paths=[pathlib.Path(__file__)],
    )


def _rehash_output(output: dict) -> None:
    output.pop("payload_sha256", None)
    output["payload_sha256"] = gate_module._digest_value(output)


def test_runs_complete_design_in_deterministic_condition_batches(
    tmp_path: pathlib.Path,
) -> None:
    backend = _FakeBackend()
    output = _run(tmp_path, backend)
    assert output["schema"] == "conflictbench.perception-omni-source-gate.v4"
    assert output["gate"]["source_sufficiency_status"] == "pass"
    assert output["gate"]["nuisance_detection_status"] == "not_run"
    assert output["gate"]["overall_pilot_status"] == "pending_nuisance_detection"
    assert output["counts"] == {
        "condition_count": 3,
        "control_type_count": 2,
        "media_file_count": 300,
        "target_count": 100,
        "pair_count": 200,
        "pair_role_counts": {
            "opposite_answer_candidate": 100,
            "same_answer_nuisance": 100,
        },
        "partition_target_counts": {
            "scorer_fit": 60,
            "threshold_calibration": 20,
            "pilot_gate": 20,
        },
        "source_record_count": 1200,
        "control_record_count": 1600,
        "record_count": 2800,
    }
    assert len(output["records"]) == 2800
    assert len({record["record_id"] for record in output["records"]}) == 2800
    assert output["gate"]["evaluated_record_count"] == 240
    assert output["gate"]["evaluated_unique_target_count"] == 20
    assert all(
        len({request.condition for request in call}) == 1 for call in backend.calls
    )
    assert all(
        len({request.evaluation for request in call}) == 1 for call in backend.calls
    )
    assert str(tmp_path) not in json.dumps(output)
    validate_gate_output(output)


def test_controls_cover_both_orientations_with_balanced_options_and_counteranswer_media(
    tmp_path: pathlib.Path,
) -> None:
    output = _run(tmp_path, _FakeBackend())
    controls = [
        record
        for record in output["records"]
        if record["evaluation"] != "source_sufficiency"
    ]
    assert {record["evaluation"] for record in controls} == {
        "question_only",
        "same_question_shuffled_media",
    }
    assert {record["source_role"] for record in controls} == {"target", "donor"}
    question_only = [
        record for record in controls if record["evaluation"] == "question_only"
    ]
    assert len(question_only) == 400
    assert all(record["media_video_id"] is None for record in question_only)
    assert all(record["media_sha256"] is None for record in question_only)
    shuffled = [
        record
        for record in controls
        if record["evaluation"] == "same_question_shuffled_media"
    ]
    assert len(shuffled) == 1200
    assert all(record["media_video_id"] != record["video_id"] for record in shuffled)
    assert all(
        record["shuffled_media_choice"] != record["expected_choice"]
        for record in shuffled
    )
    for partition in ("scorer_fit", "threshold_calibration", "pilot_gate"):
        for source_role in ("target", "donor"):
            records = [
                record
                for record in question_only
                if record["partition"] == partition
                and record["source_role"] == source_role
            ]
            if source_role == "target":
                records = list(
                    {
                        (record["video_id"], record["question_id"]): record
                        for record in records
                    }.values()
                )
            counts = [
                sum(record["expected_choice"] == choice for record in records)
                for choice in ("A", "B", "C")
            ]
            assert max(counts) - min(counts) <= 1


def test_high_question_only_or_shuffled_media_accuracy_fails_continuation(
    tmp_path: pathlib.Path,
) -> None:
    output = _run(tmp_path, _LeakyControlBackend())
    assert output["gate"]["source_sufficiency_status"] == "fail"
    assert output["gate"]["overall_pilot_status"] == "fail"
    assert any(
        "question_only" in requirement or "shuffled_media" in requirement
        for requirement in output["gate"]["failed_requirements"]
    )


def test_donor_control_leak_is_gated_separately_for_each_pair_role(
    tmp_path: pathlib.Path,
) -> None:
    output = _run(tmp_path, _DonorOppositeControlLeakBackend())

    assert output["gate"]["source_sufficiency_status"] == "fail"
    question_only = output["gate"]["control_metrics"]["question_only"][
        "by_source_orientation"
    ]["donor"]["by_pair_role"]
    assert question_only["same_answer_nuisance"]["source_answer_agreement_rate"] == 0.0
    assert (
        question_only["opposite_answer_candidate"]["source_answer_agreement_rate"]
        == 1.0
    )
    assert (
        "question_only.donor.opposite_answer_candidate."
        "source_answer_agreement_rate" in output["gate"]["failed_requirements"]
    )
    assert (
        "aligned_over_control_margin.audio_only.donor."
        "opposite_answer_candidate" in output["gate"]["failed_requirements"]
    )


def test_exact_twenty_point_margin_passes_without_float_rounding_failure(
    tmp_path: pathlib.Path,
) -> None:
    output = _run(tmp_path, _ExactTwentyPointMarginBackend())

    cell = output["gate"]["control_metrics"]["aligned_over_control_margin"][
        "audio_only"
    ]["donor"]["by_pair_role"]["same_answer_nuisance"]
    assert cell["aligned_source_correct_count"] == 14
    assert cell["aligned_source_count"] == 20
    assert cell["strongest_control_source_answer_agreement_count"] == 10
    assert cell["strongest_control_count"] == 20
    assert cell["margin"] == pytest.approx(0.20)
    assert cell["passes_minimum_margin"] is True
    assert output["gate"]["source_sufficiency_status"] == "pass"


def test_reports_exact_component_outcomes_and_wilson_interval_per_source_cell(
    tmp_path: pathlib.Path,
) -> None:
    output = _run(tmp_path, _MaskedFailureBackend())
    source_cells = output["gate"]["component_source_outcomes"]["audio_only"]

    target = source_cells["target"]
    assert target["component_count"] == 20
    assert target["component_success_count"] == 20
    assert target["component_success_rate"] == 1.0
    assert target["wilson_interval_95"] == pytest.approx([0.8388748419471808, 1.0])
    assert len(target["exact_component_outcomes"]) == 20
    assert all(
        len(item["aligned_record_ids"]) == 2 and item["success"] is True
        for item in target["exact_component_outcomes"]
    )

    donor = source_cells["donor"]["by_pair_role"]["opposite_answer_candidate"]
    assert donor["component_count"] == 20
    assert donor["component_success_count"] == 10
    assert donor["component_success_rate"] == 0.5
    assert donor["wilson_interval_95"] == pytest.approx(
        [0.2992980081982124, 0.7007019918017876]
    )
    assert len(donor["exact_component_outcomes"]) == 20
    assert all(
        len(item["aligned_record_ids"]) == 1
        for item in donor["exact_component_outcomes"]
    )


def test_question_key_sensitivity_is_key_disjoint_and_preserves_primary_cohort(
    tmp_path: pathlib.Path,
) -> None:
    output = _run(tmp_path, _FakeBackend())
    sensitivity = output["gate"]["question_key_sensitivity"]

    assert sensitivity["status"] == "pass"
    assert sensitivity["split_policy"] == (
        "pilot_gate_normalized_questions_absent_from_fit_or_threshold"
    )
    assert sensitivity["normalization"] == "NFC_and_collapsed_whitespace"
    assert sensitivity["component_count"] == 5
    assert sensitivity["minimum_component_count"] == 5
    assert sensitivity["normalized_question_key_count"] == 5
    assert set(sensitivity["held_out_normalized_question_key_sha256"]).isdisjoint(
        sensitivity["fit_or_threshold_normalized_question_key_sha256"]
    )
    assert sensitivity["key_disjoint"] is True
    assert output["gate"]["evaluated_unique_target_count"] == 20
    assert sensitivity["metrics"]["audio_only"]["target"]["aligned_source_count"] == 5
    assert (
        sensitivity["metrics"]["audio_only"]["donor"]["by_pair_role"][
            "opposite_answer_candidate"
        ]["aligned_source_count"]
        == 5
    )


def test_unseen_question_margin_attenuation_fails_closed(
    tmp_path: pathlib.Path,
) -> None:
    output = _run(tmp_path, _UnseenQuestionAttenuationBackend())
    sensitivity = output["gate"]["question_key_sensitivity"]

    assert output["gate"]["metrics"]["audio_only"]["unique_target_accuracy"] == 0.75
    assert (
        sensitivity["metrics"]["audio_only"]["target"]["aligned_source_accuracy"] == 0.0
    )
    assert sensitivity["metrics"]["audio_only"]["target"]["margin_attenuation"] > 0.2
    assert sensitivity["status"] == "fail"
    assert output["gate"]["source_sufficiency_status"] == "fail"
    assert (
        "question_key_sensitivity.audio_only.target.margin_attenuation"
        in output["gate"]["failed_requirements"]
    )


def test_insufficient_unseen_question_count_still_reports_observed_cells(
    tmp_path: pathlib.Path,
) -> None:
    config = _config()
    config["gate"]["minimum_unseen_question_component_count"] = 6
    output = _run(tmp_path, _FakeBackend(), config=config)
    sensitivity = output["gate"]["question_key_sensitivity"]

    assert sensitivity["status"] == "insufficient_components"
    assert sensitivity["component_count"] == 5
    assert sensitivity["metrics"]["audio_only"]["target"]["aligned_source_count"] == 5
    assert (
        sensitivity["metrics"]["audio_only"]["donor"]["by_pair_role"][
            "opposite_answer_candidate"
        ]["question_only_count"]
        == 5
    )
    assert (
        "question_key_sensitivity.insufficient_components"
        in output["gate"]["failed_requirements"]
    )


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda config: config["controls"].update(
                {"same_question_shuffle_policy": "arbitrary"}
            ),
            "shuffle policy",
        ),
        (
            lambda config: config["controls"].update(
                {"source_orientations": ["donor", "target"]}
            ),
            "source orientations",
        ),
        (
            lambda config: config["gate"][
                "maximum_question_only_source_answer_accuracy_by_orientation"
            ].update({"target": 0.2}),
            "question-only maximum",
        ),
    ],
)
def test_configuration_strictly_validates_control_contract(
    tmp_path: pathlib.Path, mutation, message: str
) -> None:
    config = _config()
    mutation(config)
    with pytest.raises(GateValidationError, match=message):
        _run(tmp_path, _FakeBackend(), config=config)


@pytest.mark.parametrize(
    "field, value",
    [
        ("fps", 1.0),
        ("min_frames", 2),
        ("max_frames", 64),
        ("min_pixels", 200704),
        ("max_pixels", 301056),
    ],
)
def test_configuration_locks_bounded_video_preprocessing(
    tmp_path: pathlib.Path, field: str, value: object
) -> None:
    config = _config()
    config["inference"]["video_preprocessing"][field] = value
    with pytest.raises(GateValidationError, match="video preprocessing"):
        _run(tmp_path, _FakeBackend(), config=config)


def test_configuration_locks_single_item_batches_for_effective_video_fps(
    tmp_path: pathlib.Path,
) -> None:
    config = _config()
    config["inference"]["batch_size"] = 2
    with pytest.raises(GateValidationError, match="batch_size must equal 1"):
        _run(tmp_path, _FakeBackend(), config=config)


def test_run_is_byte_deterministic_for_identical_inputs(tmp_path: pathlib.Path) -> None:
    first = _run(tmp_path / "first", _FakeBackend())
    second = _run(tmp_path / "second", _FakeBackend())
    assert first == second


def test_role_stratified_gate_cannot_be_masked_by_same_answer_pairs(
    tmp_path: pathlib.Path,
) -> None:
    config = _config()
    for condition in ("audio_only", "video_only", "audiovisual"):
        config["gate"]["minimum_pair_support_rate_by_role_and_condition"][
            "opposite_answer_candidate"
        ][condition] = 0.6
    output = _run(tmp_path, _MaskedFailureBackend(), config=config)
    assert output["gate"]["source_sufficiency_status"] == "fail"
    assert output["gate"]["overall_pilot_status"] == "fail"
    opposite = output["gate"]["metrics"]["audio_only"]["by_pair_role"][
        "opposite_answer_candidate"
    ]
    same = output["gate"]["metrics"]["audio_only"]["by_pair_role"][
        "same_answer_nuisance"
    ]
    assert opposite["source_accuracy"]["donor"] == 0.5
    assert opposite["pair_support_rate"] == 0.5
    assert same["source_accuracy"]["donor"] == 1.0
    assert (
        "audio_only.opposite_answer_candidate.donor_accuracy"
        in output["gate"]["failed_requirements"]
    )


def test_duplicate_target_statistics_must_agree_before_deduplication(
    tmp_path: pathlib.Path,
) -> None:
    with pytest.raises(
        GateValidationError, match="duplicate target evaluations disagree"
    ):
        _run(tmp_path, _DisagreeingDuplicateTargetBackend())


def test_missing_required_media_fails_before_inference(tmp_path: pathlib.Path) -> None:
    config_path, config_sha256, pilot_path, pilot_sha256, media_root = (
        _write_gate_inputs(tmp_path)
    )
    (media_root / "donor_same_000.mp4").unlink()
    backend = _FakeBackend()
    with pytest.raises(GateValidationError, match="required media file"):
        run_source_sufficiency_gate(
            config_path=config_path,
            expected_config_sha256=config_sha256,
            pilot_index_path=pilot_path,
            expected_pilot_index_sha256=pilot_sha256,
            media_root=media_root,
            backend=backend,
            source_paths=[],
        )
    assert backend.calls == []


def test_media_mutation_after_authentication_is_detected(
    tmp_path: pathlib.Path,
) -> None:
    with pytest.raises(GateValidationError, match="media changed during inference"):
        _run(tmp_path, _MutatingBackend())


def test_rejects_any_index_that_loaded_validation_annotations(
    tmp_path: pathlib.Path,
) -> None:
    pilot = _pilot_index()
    pilot["source_use"]["builder_validation_annotations_loaded"] = True
    _reattest(pilot)
    config_path, config_sha256, pilot_path, pilot_sha256, media_root = (
        _write_gate_inputs(tmp_path, pilot=pilot)
    )
    backend = _FakeBackend()
    with pytest.raises(GateValidationError, match="validation annotations"):
        run_source_sufficiency_gate(
            config_path=config_path,
            expected_config_sha256=config_sha256,
            pilot_index_path=pilot_path,
            expected_pilot_index_sha256=pilot_sha256,
            media_root=media_root,
            backend=backend,
            source_paths=[],
        )
    assert backend.calls == []


def test_rejects_stale_pilot_attestation(tmp_path: pathlib.Path) -> None:
    pilot = _pilot_index()
    pilot["counts"]["target_count"] = 99
    with pytest.raises(GateValidationError, match="attestation"):
        _run(tmp_path, _FakeBackend(), pilot=pilot)


def test_rejects_attested_wrong_exact_target_count(tmp_path: pathlib.Path) -> None:
    pilot = _pilot_index()
    pilot["counts"]["target_count"] = 99
    _reattest(pilot)
    with pytest.raises(GateValidationError, match="exactly 100 targets"):
        _run(tmp_path, _FakeBackend(), pilot=pilot)


def test_rejects_attested_cross_component_video_reuse(tmp_path: pathlib.Path) -> None:
    pilot = _pilot_index()
    first, second = pilot["components"][:2]
    old_second_id = second["component_id"]
    second["video_ids"][0] = first["video_ids"][0]
    second["video_ids"] = sorted(set(second["video_ids"]))
    second["component_id"] = _digest24(second["video_ids"])
    for pair in pilot["candidate_index"]:
        if pair["component_id"] == old_second_id:
            pair["component_id"] = second["component_id"]
    pilot["components"].sort(key=lambda item: (item["partition"], item["component_id"]))
    pilot["candidate_index"].sort(
        key=lambda item: (item["partition"], item["component_id"], item["pair_id"])
    )
    _reattest(pilot)
    with pytest.raises(GateValidationError, match="multiple components"):
        _run(tmp_path, _FakeBackend(), pilot=pilot)


def test_rejects_attested_disconnected_declared_component(
    tmp_path: pathlib.Path,
) -> None:
    pilot = _pilot_index()
    first, second = next(
        (left, right)
        for left, right in zip(pilot["components"], pilot["components"][1:])
        if left["partition"] == right["partition"]
    )
    old_ids = {first["component_id"], second["component_id"]}
    merged_videos = sorted(first["video_ids"] + second["video_ids"])
    merged_id = _digest24(merged_videos)
    merged = {
        "component_id": merged_id,
        "partition": first["partition"],
        "target_count": 2,
        "video_ids": merged_videos,
    }
    pilot["components"] = [
        component
        for component in pilot["components"]
        if component["component_id"] not in old_ids
    ] + [merged]
    for pair in pilot["candidate_index"]:
        if pair["component_id"] in old_ids:
            pair["component_id"] = merged_id
    pilot["components"].sort(key=lambda item: (item["partition"], item["component_id"]))
    pilot["candidate_index"].sort(
        key=lambda item: (item["partition"], item["component_id"], item["pair_id"])
    )
    pilot["counts"]["component_count"] -= 1
    pilot["partition_counts"][first["partition"]]["component_count"] -= 1
    _reattest(pilot)
    with pytest.raises(GateValidationError, match="recomputed donor-connected"):
        _run(tmp_path, _FakeBackend(), pilot=pilot)


def test_rejects_conflicting_payload_for_reused_video_question(
    tmp_path: pathlib.Path,
) -> None:
    pilot = _pilot_index()
    target_id = "target_000"
    target_pairs = [
        pair
        for pair in pilot["candidate_index"]
        if pair["target"]["video_id"] == target_id
    ]
    same_pair = next(
        pair for pair in target_pairs if pair["role"] == "same_answer_nuisance"
    )
    opposite_pair = next(
        pair for pair in target_pairs if pair["role"] == "opposite_answer_candidate"
    )
    old_component_id = same_pair["component_id"]
    opposite_pair["donor"]["video_id"] = same_pair["donor"]["video_id"]
    component = next(
        item for item in pilot["components"] if item["component_id"] == old_component_id
    )
    component["video_ids"] = sorted({target_id, same_pair["donor"]["video_id"]})
    component["component_id"] = _digest24(component["video_ids"])
    for pair in target_pairs:
        pair["component_id"] = component["component_id"]
    pilot["counts"]["unique_donor_count"] -= 1
    pilot["components"].sort(key=lambda item: (item["partition"], item["component_id"]))
    pilot["candidate_index"].sort(
        key=lambda item: (item["partition"], item["component_id"], item["pair_id"])
    )
    _reattest(pilot)
    with pytest.raises(GateValidationError, match="conflicting payloads"):
        _run(tmp_path, _FakeBackend(), pilot=pilot)


def test_rejects_incomplete_backend_results(tmp_path: pathlib.Path) -> None:
    with pytest.raises(GateValidationError, match="backend result IDs differ"):
        _run(tmp_path, _FakeBackend(omit_last=True))


def test_records_auditable_unavailable_diagnostic_scores(
    tmp_path: pathlib.Path,
) -> None:
    output = _run(tmp_path, _FakeBackend(emit_scores=False))
    assert output["gate"]["source_sufficiency_status"] == "pass"
    assert all(
        record["choice_scores"]
        == {
            "method": None,
            "status": "unavailable",
            "token_ids": None,
            "used_for_gate": False,
            "values": None,
        }
        for record in output["records"]
    )


def test_configuration_rejects_required_but_disabled_scores(
    tmp_path: pathlib.Path,
) -> None:
    config = _config()
    config["inference"]["emit_choice_scores_when_single_token"] = False
    config["gate"]["require_choice_scores"] = True
    with pytest.raises(GateValidationError, match="must be enabled"):
        _run(tmp_path, _FakeBackend(), config=config)


def test_output_validator_reconstructs_exact_record_order_and_hashes(
    tmp_path: pathlib.Path,
) -> None:
    output = _run(tmp_path, _FakeBackend())
    output["records"][0]["request_sha256"] = "0" * 64
    _rehash_output(output)
    with pytest.raises(GateValidationError, match="request_sha256 differs"):
        validate_gate_output(output)
    output = _run(tmp_path / "order", _FakeBackend())
    output["records"][0], output["records"][1] = (
        output["records"][1],
        output["records"][0],
    )
    _rehash_output(output)
    with pytest.raises(GateValidationError, match="record .* differs"):
        validate_gate_output(output)


def test_output_validator_rejects_rehashed_control_assignment_tamper(
    tmp_path: pathlib.Path,
) -> None:
    output = _run(tmp_path, _FakeBackend())
    record = next(
        record
        for record in output["records"]
        if record["evaluation"] == "question_only"
    )
    record["media_video_id"] = record["video_id"]
    record["media_sha256"] = next(
        media["sha256"]
        for media in output["media_files"]
        if media["video_id"] == record["video_id"]
    )
    _rehash_output(output)
    with pytest.raises(GateValidationError, match="record .* differs"):
        validate_gate_output(output)


def test_output_validator_rejects_rehashed_option_balance_tamper(
    tmp_path: pathlib.Path,
) -> None:
    output = _run(tmp_path, _FakeBackend())
    source = output["candidate_design"][0]["target"]
    source["option_balance_slot"] = (source["option_balance_slot"] + 1) % 3
    output["input_digests"]["candidate_design_sha256"] = gate_module._digest_value(
        output["candidate_design"]
    )
    output["input_digests"]["run_input_sha256"] = gate_module._digest_value(
        {
            "configuration_sha256": output["input_digests"]["configuration_sha256"],
            "configuration_value_sha256": output["input_digests"][
                "configuration_value_sha256"
            ],
            "pilot_index_sha256": output["input_digests"]["pilot_index_sha256"],
            "source_set_sha256": output["input_digests"]["source_set_sha256"],
            "media_set_sha256": output["input_digests"]["media_set_sha256"],
            "candidate_design_sha256": output["input_digests"][
                "candidate_design_sha256"
            ],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
    )
    _rehash_output(output)
    with pytest.raises(GateValidationError, match="option balance slot"):
        validate_gate_output(output)


def test_independent_replay_reconstructs_output_and_rejects_rehashed_result_tamper(
    tmp_path: pathlib.Path,
) -> None:
    config_path, config_sha256, pilot_path, pilot_sha256, media_root = (
        _write_gate_inputs(tmp_path)
    )
    source_paths = [pathlib.Path(__file__)]
    output = run_source_sufficiency_gate(
        config_path=config_path,
        expected_config_sha256=config_sha256,
        pilot_index_path=pilot_path,
        expected_pilot_index_sha256=pilot_sha256,
        media_root=media_root,
        backend=_FakeBackend(),
        source_paths=source_paths,
    )
    transcript = gate_module.build_inference_transcript(output)
    assert output["inference_transcript_sha256"] == gate_module._digest_value(
        transcript
    )
    transcript_path = tmp_path / "inference-transcript.json"
    transcript_sha256 = _write_json(transcript_path, transcript)

    replayed = gate_module.verify_gate_output_replay(
        output,
        config_path=config_path,
        expected_config_sha256=config_sha256,
        pilot_index_path=pilot_path,
        expected_pilot_index_sha256=pilot_sha256,
        media_root=media_root,
        source_paths=source_paths,
        transcript_path=transcript_path,
        expected_transcript_sha256=transcript_sha256,
    )
    assert replayed == output

    tampered = copy.deepcopy(output)
    record = next(
        item
        for item in tampered["records"]
        if item["partition"] == "pilot_gate"
        and item["evaluation"] == "source_sufficiency"
        and item["source_role"] == "donor"
    )
    original_choice = record["predicted_choice"]
    changed_choice = {"A": "B", "B": "C", "C": "A"}[original_choice]
    record["raw_response"] = changed_choice
    record["predicted_choice"] = changed_choice
    record["parse_status"] = "strict"
    record["is_correct"] = changed_choice == record["expected_choice"]
    tampered["gate"] = gate_module._summarize_gate(
        tampered["records"], tampered["configuration"]["gate"]
    )
    tampered_transcript = copy.deepcopy(transcript)
    transcript_record = next(
        item
        for item in tampered_transcript["results"]
        if item["record_id"] == record["record_id"]
    )
    transcript_record["raw_response"] = changed_choice
    tampered["inference_transcript_sha256"] = gate_module._digest_value(
        tampered_transcript
    )
    _rehash_output(tampered)
    validate_gate_output(tampered)

    with pytest.raises(GateValidationError, match="replay reconstruction differs"):
        gate_module.verify_gate_output_replay(
            tampered,
            config_path=config_path,
            expected_config_sha256=config_sha256,
            pilot_index_path=pilot_path,
            expected_pilot_index_sha256=pilot_sha256,
            media_root=media_root,
            source_paths=source_paths,
            transcript_path=transcript_path,
            expected_transcript_sha256=transcript_sha256,
        )


def test_output_validator_rejects_wrong_external_lock_hash(
    tmp_path: pathlib.Path,
) -> None:
    output = _run(tmp_path, _FakeBackend())
    with pytest.raises(GateValidationError, match="pilot index SHA-256 differs"):
        validate_gate_output(output, expected_pilot_index_sha256="0" * 64)


def test_output_validator_recomputes_media_digest(tmp_path: pathlib.Path) -> None:
    output = _run(tmp_path, _FakeBackend())
    output["media_files"][0]["sha256"] = "0" * 64
    _rehash_output(output)
    with pytest.raises(GateValidationError, match="media set SHA-256"):
        validate_gate_output(output)


def test_output_validator_rejects_rehashed_prompt_tamper(
    tmp_path: pathlib.Path,
) -> None:
    output = _run(tmp_path, _FakeBackend())
    output["candidate_design"][0]["prompt"] += " tampered"
    _rehash_output(output)
    with pytest.raises(GateValidationError, match="prompt SHA-256"):
        validate_gate_output(output)


def test_write_is_non_overwriting_and_preserves_first_result(
    tmp_path: pathlib.Path,
) -> None:
    output = _run(tmp_path / "run", _FakeBackend())
    destination = tmp_path / "result.json"
    write_gate_output(destination, output)
    first_bytes = destination.read_bytes()
    with pytest.raises(GateValidationError, match="already exists"):
        write_gate_output(destination, output)
    assert destination.read_bytes() == first_bytes
    assert not list(tmp_path.glob(".result.json.*.tmp"))


def test_write_preserves_a_racing_winner(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _run(tmp_path / "run", _FakeBackend())
    destination = tmp_path / "result.json"
    real_link = gate_module.os.link

    def racing_link(source, target):
        pathlib.Path(target).write_bytes(b"racing winner")
        return real_link(source, target)

    monkeypatch.setattr(gate_module.os, "link", racing_link)
    with pytest.raises(GateValidationError, match="already exists"):
        write_gate_output(destination, output)
    assert destination.read_bytes() == b"racing winner"
    assert not list(tmp_path.glob(".result.json.*.tmp"))


def test_ffprobe_preflight_is_bounded_and_requires_both_streams(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"streams": [{"codec_type": "video"}, {"codec_type": "audio"}]}
            ),
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _REAL_PROBE_MEDIA_STREAMS(media) == ["audio", "video"]
    assert calls[0][1]["timeout"] == 15
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            returncode=0, stdout=json.dumps({"streams": [{"codec_type": "video"}]})
        ),
    )
    with pytest.raises(GateValidationError, match="audio and video streams"):
        _REAL_PROBE_MEDIA_STREAMS(media)


def test_qwen_backend_is_lazy_and_pinned(tmp_path: pathlib.Path) -> None:
    config = _config()
    backend = QwenOmniBackend(
        model_config=config["model"],
        inference_config=config["inference"],
        cache_dir=tmp_path,
    )
    assert backend.loaded is False
    bad_model = copy.deepcopy(config["model"])
    bad_model["revision"] = "0" * 40
    with pytest.raises(GateValidationError, match="pinned model identity"):
        QwenOmniBackend(
            model_config=bad_model,
            inference_config=config["inference"],
            cache_dir=tmp_path,
        )


def test_qwen_backend_propagates_audio_in_video_to_all_three_apis(
    tmp_path: pathlib.Path,
) -> None:
    config = _config()
    config["inference"]["emit_choice_scores_when_single_token"] = False
    backend = QwenOmniBackend(
        model_config=config["model"],
        inference_config=config["inference"],
        cache_dir=tmp_path,
    )
    calls: dict[str, list] = {"process": [], "processor": [], "generate": []}

    class FakeSequences:
        def __getitem__(self, _key):
            return [[1]]

    class FakeInputs(dict):
        def __init__(self):
            super().__init__(input_ids=types.SimpleNamespace(shape=(1, 4)))
            self.input_ids = self["input_ids"]

        def to(self, _value):
            return self

    class FakeProcessor:
        def apply_chat_template(self, *_args, **_kwargs):
            return "formatted prompt"

        def __call__(self, **kwargs):
            calls["processor"].append(kwargs)
            return FakeInputs()

        def batch_decode(self, *_args, **_kwargs):
            return ["A"]

    class FakeModel:
        device = "cuda"
        dtype = "float16"

        def generate(self, **kwargs):
            calls["generate"].append(kwargs)
            return types.SimpleNamespace(sequences=FakeSequences(), logits=None)

    class FakeTorch:
        @staticmethod
        def inference_mode():
            return contextlib.nullcontext()

    def fake_process(conversation, *, use_audio_in_video, return_video_kwargs):
        assert return_video_kwargs is True
        media = conversation[0]["content"][0]
        calls["process"].append((media, use_audio_in_video))
        if media["type"] == "audio":
            return ["audio"], [], [], {"fps": []}
        if use_audio_in_video:
            return ["audio"], [], ["video"], {"fps": [1.4]}
        return [], [], ["video"], {"fps": [1.4]}

    backend._model = FakeModel()
    backend._processor = FakeProcessor()
    backend._process_mm_info = fake_process
    backend._torch = FakeTorch()
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    cases = (
        ("audio_only", "audio", False, True, False),
        ("video_only", "video", False, False, True),
        ("audiovisual", "video", True, True, True),
    )
    for condition, media_type, expected, has_audio, has_video in cases:
        request = InferenceRequest(
            record_id=_digest24(condition),
            pair_id="1" * 24,
            pair_role="same_answer_nuisance",
            source_role="target",
            condition=condition,
            video_id="clip",
            media_path=media,
            prompt="question",
        )
        assert backend.infer([request])[0].raw_response == "A"
        media_value, observed_flag = calls["process"][-1]
        assert media_value["type"] == media_type
        if media_type == "video":
            assert media_value == {
                "type": "video",
                "video": str(media),
                "fps": 2.0,
                "min_frames": 4,
                "max_frames": 32,
                "min_pixels": 100352,
                "max_pixels": 200704,
            }
        else:
            assert media_value == {"type": "audio", "audio": str(media)}
        assert observed_flag is expected
        assert calls["processor"][-1]["use_audio_in_video"] is expected
        assert ("audio" in calls["processor"][-1]) is has_audio
        assert ("videos" in calls["processor"][-1]) is has_video
        if has_video:
            assert calls["processor"][-1]["fps"] == 1.4
        else:
            assert "fps" not in calls["processor"][-1]
        assert calls["generate"][-1]["use_audio_in_video"] is expected
        assert calls["generate"][-1]["thinker_max_new_tokens"] == 4
        assert calls["generate"][-1]["return_audio"] is False
        assert calls["generate"][-1]["output_logits"] is False


def test_qwen_backend_question_only_uses_text_without_media(
    tmp_path: pathlib.Path,
) -> None:
    config = _config()
    config["inference"]["emit_choice_scores_when_single_token"] = False
    backend = QwenOmniBackend(
        model_config=config["model"],
        inference_config=config["inference"],
        cache_dir=tmp_path,
    )
    seen: dict[str, object] = {}

    class FakeSequences:
        def __getitem__(self, _key):
            return [[1]]

    class FakeInputs(dict):
        def __init__(self):
            super().__init__(input_ids=types.SimpleNamespace(shape=(1, 4)))
            self.input_ids = self["input_ids"]

        def to(self, _value):
            return self

    class FakeProcessor:
        def apply_chat_template(self, conversation, **_kwargs):
            seen["content"] = conversation[0]["content"]
            return "formatted prompt"

        def __call__(self, **kwargs):
            seen["processor"] = kwargs
            return FakeInputs()

        def batch_decode(self, *_args, **_kwargs):
            return ["A"]

    class FakeModel:
        device = "cuda"
        dtype = "float16"

        def generate(self, **_kwargs):
            return types.SimpleNamespace(sequences=FakeSequences(), logits=None)

    class FakeTorch:
        @staticmethod
        def inference_mode():
            return contextlib.nullcontext()

    def fake_process(conversation, *, use_audio_in_video, return_video_kwargs):
        assert use_audio_in_video is False
        assert return_video_kwargs is True
        assert conversation[0]["content"] == [{"type": "text", "text": "question"}]
        return [], [], [], {"fps": []}

    backend._model = FakeModel()
    backend._processor = FakeProcessor()
    backend._process_mm_info = fake_process
    backend._torch = FakeTorch()
    request = InferenceRequest(
        record_id="1" * 24,
        pair_id="2" * 24,
        pair_role="same_answer_nuisance",
        source_role="target",
        condition="question_only",
        video_id="source-video",
        media_path=None,
        prompt="question",
        evaluation="question_only",
        media_video_id=None,
    )
    assert backend.infer([request])[0].raw_response == "A"
    assert seen["content"] == [{"type": "text", "text": "question"}]
    assert "audio" not in seen["processor"]
    assert "videos" not in seen["processor"]


@pytest.mark.parametrize(
    "video_kwargs",
    [
        {},
        {"fps": "1.4"},
        {"fps": []},
        {"fps": [float("nan")]},
        {"fps": [1.4, 1.4]},
    ],
)
def test_qwen_backend_rejects_invalid_sampled_video_metadata(
    tmp_path: pathlib.Path, video_kwargs: object
) -> None:
    config = _config()
    backend = QwenOmniBackend(
        model_config=config["model"],
        inference_config=config["inference"],
        cache_dir=tmp_path,
    )

    class FakeProcessor:
        def apply_chat_template(self, *_args, **_kwargs):
            return "formatted prompt"

    backend._model = object()
    backend._processor = FakeProcessor()
    backend._process_mm_info = lambda *_args, **_kwargs: (
        [],
        [],
        ["video"],
        video_kwargs,
    )
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    request = InferenceRequest(
        record_id="1" * 24,
        pair_id="2" * 24,
        pair_role="same_answer_nuisance",
        source_role="target",
        condition="video_only",
        video_id="source-video",
        media_path=media,
        prompt="question",
    )
    with pytest.raises(
        GateValidationError,
        match="video metadata|sampled video FPS|each video batch",
    ):
        backend.infer([request])


def test_qwen_backend_uses_per_prompt_contextual_ids_for_diagnostic_logits(
    tmp_path: pathlib.Path,
) -> None:
    config = _config()
    backend = QwenOmniBackend(
        model_config=config["model"],
        inference_config=config["inference"],
        cache_dir=tmp_path,
    )
    generated_kwargs = []

    class FakeScalar:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def item(self):
            return self.value

    class FakeVector:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, token_id):
            return FakeScalar(self.values[token_id])

    class FakeMatrix:
        def __getitem__(self, batch_index):
            values = (
                {11: 1.0, 12: 2.0, 13: 3.0}
                if batch_index == 0
                else {21: 4.0, 22: 5.0, 23: 6.0}
            )
            return FakeVector(values)

    class FakeSequences:
        def __getitem__(self, _key):
            return [[1], [2]]

    class FakeInputs(dict):
        def __init__(self):
            super().__init__(input_ids=types.SimpleNamespace(shape=(2, 4)))
            self.input_ids = self["input_ids"]

        def to(self, _value):
            return self

    class FakeTokenizer:
        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            choice = text[-1] if text[-1] in "ABC" else None
            base = text[:-1] if choice else text
            prefix_id = 1 if "question-one" in base else 2
            if choice is None:
                return [prefix_id]
            offset = 10 if prefix_id == 1 else 20
            return [prefix_id, offset + {"A": 1, "B": 2, "C": 3}[choice]]

    class FakeProcessor:
        tokenizer = FakeTokenizer()

        def apply_chat_template(self, conversation, **_kwargs):
            return "formatted:" + conversation[0]["content"][1]["text"] + ":"

        def __call__(self, **_kwargs):
            return FakeInputs()

        def batch_decode(self, *_args, **_kwargs):
            return ["A", "B"]

    class FakeModel:
        device = "cuda"
        dtype = "float16"

        def generate(self, **kwargs):
            generated_kwargs.append(kwargs)
            return types.SimpleNamespace(
                sequences=FakeSequences(), logits=(FakeMatrix(),)
            )

    class FakeTorch:
        @staticmethod
        def inference_mode():
            return contextlib.nullcontext()

    backend._model = FakeModel()
    backend._processor = FakeProcessor()
    backend._process_mm_info = lambda *_args, **_kwargs: (
        ["audio"],
        [],
        [],
        {"fps": []},
    )
    backend._torch = FakeTorch()
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    requests = [
        InferenceRequest(
            record_id=_digest24(prompt),
            pair_id=str(index + 1) * 24,
            pair_role="same_answer_nuisance",
            source_role="target",
            condition="audio_only",
            video_id=f"clip_{index}",
            media_path=media,
            prompt=prompt,
        )
        for index, prompt in enumerate(("question-one", "question-two"))
    ]
    results = backend.infer(requests)
    assert results[0].choice_token_ids == {"A": 11, "B": 12, "C": 13}
    assert results[0].choice_scores == {"A": 1.0, "B": 2.0, "C": 3.0}
    assert results[1].choice_token_ids == {"A": 21, "B": 22, "C": 23}
    assert results[1].choice_scores == {"A": 4.0, "B": 5.0, "C": 6.0}
    assert generated_kwargs[0]["return_dict_in_generate"] is True
    assert generated_kwargs[0]["output_logits"] is True
    assert generated_kwargs[0]["thinker_max_new_tokens"] == 4
    assert "max_new_tokens" not in generated_kwargs[0]


def test_contextual_choice_tokens_require_distinct_one_token_suffixes() -> None:
    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            if text == "prompt":
                return [1, 2]
            return [1, 2, {"A": 11, "B": 12, "C": 13}[text[-1]]]

    assert QwenOmniBackend._contextual_choice_token_ids(Tokenizer(), "prompt") == {
        "A": 11,
        "B": 12,
        "C": 13,
    }

    class MultiTokenTokenizer(Tokenizer):
        def encode(self, text, *, add_special_tokens):
            if text == "prompt":
                return [1, 2]
            return [1, 2, 3, 4]

    class DuplicateTokenizer(Tokenizer):
        def encode(self, text, *, add_special_tokens):
            if text == "prompt":
                return [1, 2]
            return [1, 2, 9]

    class PrefixMismatchTokenizer(Tokenizer):
        def encode(self, text, *, add_special_tokens):
            if text == "prompt":
                return [1, 2]
            return [1, 8, 9]

    assert (
        QwenOmniBackend._contextual_choice_token_ids(MultiTokenTokenizer(), "prompt")
        is None
    )
    assert (
        QwenOmniBackend._contextual_choice_token_ids(DuplicateTokenizer(), "prompt")
        is None
    )
    assert (
        QwenOmniBackend._contextual_choice_token_ids(
            PrefixMismatchTokenizer(), "prompt"
        )
        is None
    )


def test_cli_preserves_valid_negative_output_with_zero_exit(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = (
        pathlib.Path(__file__).parents[1] / "scripts/run_perception_omni_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_perception_omni_gate_test", script_path
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    config = _config()
    output = {"gate": {"source_sufficiency_status": "fail"}}
    written = []
    monkeypatch.setattr(cli, "load_gate_configuration", lambda *_args: config)
    monkeypatch.setattr(cli, "QwenOmniBackend", lambda **_kwargs: object())
    monkeypatch.setattr(cli, "run_source_sufficiency_gate", lambda **_kwargs: output)
    monkeypatch.setattr(
        cli,
        "write_gate_output",
        lambda *args, **kwargs: written.append((args, kwargs)),
    )
    result = cli.main(
        [
            "--config",
            str(tmp_path / "config.json"),
            "--config-sha256",
            "1" * 64,
            "--pilot-index",
            str(tmp_path / "pilot.json"),
            "--pilot-index-sha256",
            "2" * 64,
            "--media-root",
            str(tmp_path / "media"),
            "--model-cache",
            str(tmp_path / "cache"),
            "--output",
            str(tmp_path / "output.json"),
        ]
    )
    assert result == 0
    assert written


def test_replay_verifier_cli_requires_locks_and_reports_reconstructed_digest(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script_path = (
        pathlib.Path(__file__).parents[1] / "scripts/verify_perception_omni_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "verify_perception_omni_gate_test", script_path
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    output_path = tmp_path / "output.json"
    output_path.write_text('{"placeholder":true}\n', encoding="utf-8")
    seen: dict[str, object] = {}

    def fake_verify(output, **kwargs):
        seen["output"] = output
        seen.update(kwargs)
        return {
            "payload_sha256": "a" * 64,
            "inference_transcript_sha256": "b" * 64,
            "counts": {"record_count": 2800},
        }

    monkeypatch.setattr(cli, "verify_gate_output_replay", fake_verify)
    result = cli.main(
        [
            "--config",
            str(tmp_path / "config.json"),
            "--config-sha256",
            "1" * 64,
            "--pilot-index",
            str(tmp_path / "pilot.json"),
            "--pilot-index-sha256",
            "2" * 64,
            "--media-root",
            str(tmp_path / "media"),
            "--source",
            str(tmp_path / "source.py"),
            "--transcript",
            str(tmp_path / "transcript.json"),
            "--transcript-sha256",
            "3" * 64,
            "--output",
            str(output_path),
        ]
    )
    assert result == 0
    assert seen["output"] == {"placeholder": True}
    assert seen["source_paths"] == [tmp_path / "source.py"]
    assert json.loads(capsys.readouterr().out) == {
        "inference_transcript_sha256": "b" * 64,
        "payload_sha256": "a" * 64,
        "record_count": 2800,
        "status": "pass",
    }
