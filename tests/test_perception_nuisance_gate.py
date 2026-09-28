from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import stat
import statistics
import subprocess
import sys

import numpy as np
import pytest

import conflictbench.perception_nuisance_gate as nuisance_gate_module
from conflictbench.perception_nuisance_gate import (
    FEATURE_NAMES,
    NuisanceValidationError,
    SubprocessReplayWorkerSpec,
    _permuted_records,
    build_feature_records,
    build_handoff_result,
    build_nuisance_contract,
    build_post_edit_feature_records,
    evaluate_nuisance_records,
    evaluate_post_edit_diagnostics,
    evaluate_shortcut_controls_v2,
    extract_metadata_features,
    freeze_detector,
    parse_ffprobe_record,
    probe_media_file,
    validate_handoff_result,
    validate_media_receipt,
    validate_nuisance_contract,
    validate_shortcut_output_v2,
    write_gate_outputs,
)

CONFIG_PATH = (
    pathlib.Path(__file__).parents[1] / "configs" / "perception_nuisance_gate.json"
)
V2_CONFIG_PATH = (
    pathlib.Path(__file__).parents[1] / "configs" / "perception_shortcut_gate_v2.json"
)
RUNNER_PATH = (
    pathlib.Path(__file__).parents[1] / "scripts" / "run_perception_nuisance_gate.py"
)
CONTRACT_BUILDER_PATH = (
    pathlib.Path(__file__).parents[1]
    / "scripts"
    / "build_perception_nuisance_contract.py"
)


def _probe(
    *,
    duration: float,
    video_duration: float,
    audio_duration: float,
    width: int,
    height: int,
    frame_rate: str,
    sample_rate: int,
    channels: int,
    video_codec: str = "h264",
    audio_codec: str = "aac",
    pixel_format: str = "yuv420p",
    sample_format: str = "fltp",
) -> dict:
    return {
        "format": {"duration": str(duration)},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": video_codec,
                "pix_fmt": pixel_format,
                "width": width,
                "height": height,
                "avg_frame_rate": frame_rate,
                "duration": str(video_duration),
            },
            {
                "codec_type": "audio",
                "codec_name": audio_codec,
                "sample_fmt": sample_format,
                "sample_rate": str(sample_rate),
                "channels": channels,
                "duration": str(audio_duration),
            },
        ],
    }


def test_parse_ffprobe_record_requires_one_audio_and_one_video_stream() -> None:
    payload = _probe(
        duration=10.0,
        video_duration=9.5,
        audio_duration=10.0,
        width=100,
        height=100,
        frame_rate="25/1",
        sample_rate=48_000,
        channels=2,
    )
    payload["streams"].append(copy.deepcopy(payload["streams"][0]))

    with pytest.raises(
        NuisanceValidationError, match="exactly one audio and one video"
    ):
        parse_ffprobe_record(payload, size_bytes=1_000)


def test_metadata_features_are_scale_free_and_content_blind() -> None:
    target = parse_ffprobe_record(
        _probe(
            duration=10.0,
            video_duration=9.5,
            audio_duration=10.0,
            width=100,
            height=100,
            frame_rate="25/1",
            sample_rate=48_000,
            channels=2,
            video_codec="h264",
            audio_codec="aac",
            pixel_format="yuv420p",
            sample_format="fltp",
        ),
        size_bytes=1_000,
    )
    donor = parse_ffprobe_record(
        _probe(
            duration=20.0,
            video_duration=18.0,
            audio_duration=20.0,
            width=200,
            height=100,
            frame_rate="50/1",
            sample_rate=24_000,
            channels=1,
            video_codec="hevc",
            audio_codec="aac",
            pixel_format="yuv444p",
            sample_format="s16",
        ),
        size_bytes=4_000,
    )

    observed = dict(zip(FEATURE_NAMES, extract_metadata_features(target, donor)))
    assert set(observed) == {
        "duration_relative_difference",
        "byte_rate_relative_difference",
        "target_av_duration_gap_fraction",
        "donor_av_duration_gap_fraction",
        "target_video_donor_audio_gap_fraction",
        "target_audio_donor_video_gap_fraction",
        "width_relative_difference",
        "height_relative_difference",
        "frame_rate_relative_difference",
        "sample_rate_relative_difference",
        "channel_count_relative_difference",
        "video_codec_mismatch",
        "audio_codec_mismatch",
        "pixel_format_mismatch",
        "sample_format_mismatch",
    }
    np.testing.assert_allclose(
        list(observed.values()),
        [
            0.5,
            0.5,
            0.05,
            0.1,
            0.525,
            4.0 / 9.0,
            0.5,
            0.0,
            0.5,
            0.5,
            0.5,
            1.0,
            0.0,
            1.0,
            1.0,
        ],
    )
    forbidden = {"question", "answer", "source_score", "source_sufficiency"}
    assert forbidden.isdisjoint(observed)


def _config(*, minimum_unseen_components: int = 2) -> dict:
    return {
        "schema_version": 1,
        "runner": "perception_edit_role_nuisance_detector",
        "feature_set": {
            "name": "container_and_stream_diagnostics_v1",
            "names": list(FEATURE_NAMES),
            "forbidden_inputs": [
                "question",
                "options",
                "answer",
                "semantic_embedding",
                "source_answer_score",
                "source_sufficiency_outcome",
            ],
        },
        "model": {
            "families": [
                "ridge_linear",
                "univariate_linear",
                "pairwise_interaction",
            ],
            "ridge_l2": 1.0,
            "selection": "fit_only_max_absolute_covariance",
        },
        "protocol": {
            "fit_partition": "scorer_fit",
            "threshold_partition": "threshold_calibration",
            "evaluation_partition": "pilot_gate",
            "positive_role": "opposite_answer_candidate",
            "negative_role": "same_answer_nuisance",
            "threshold_selection": "maximum_balanced_accuracy_then_closest_to_zero",
            "question_key_sensitivity": "evaluation_keys_unseen_in_fit_or_threshold",
            "minimum_unseen_question_components": minimum_unseen_components,
            "human_evaluation": "forbidden",
            "outcome_based_sample_selection": "forbidden",
        },
        "uncertainty": {
            "method": "component_cluster_percentile_wilson_and_max_statistic",
            "confidence_level": 0.95,
            "bootstrap_repetitions": 128,
            "bootstrap_seed": 2301,
        },
        "permutation": {
            "method": "within_component_role_swap_with_refit",
            "repetitions": 128,
            "seed": 4129,
        },
        "gate": {
            "maximum_primary_upper_confidence_bound": 0.95,
            "maximum_unseen_question_upper_confidence_bound": 0.95,
            "maximum_point_lift_over_structural_chance": 0.10,
            "maximum_structural_chance_balanced_accuracy": 0.60,
            "minimum_permutation_p_value_exclusive": 0.05,
        },
    }


def _records(
    *,
    role_signal: float = 0.0,
    repeated_pilot_key: bool = False,
    partition_sizes: dict[str, int] | None = None,
) -> list[dict]:
    records = []
    if partition_sizes is None:
        partition_sizes = {
            "scorer_fit": 6,
            "threshold_calibration": 3,
            "pilot_gate": 4,
        }
    component_index = 0
    for partition, count in partition_sizes.items():
        for local_index in range(count):
            key = f"key-{component_index:02d}"
            if repeated_pilot_key and partition == "pilot_gate":
                key = "key-00"
            target_id = f"target-{component_index:02d}"
            component_id = f"component-{component_index:02d}"
            for label, role in (
                (0.0, "same_answer_nuisance"),
                (1.0, "opposite_answer_candidate"),
            ):
                features = np.zeros(len(FEATURE_NAMES), dtype=float)
                features[0] = role_signal * label
                features[1] = ((component_index * 17 + int(label) * 3) % 11) / 100.0
                records.append(
                    {
                        "pair_id": f"pair-{component_index:02d}-{int(label)}",
                        "target_id": target_id,
                        "component_id": component_id,
                        "partition": partition,
                        "key_sha256": key,
                        "target_video_id": target_id,
                        "donor_video_id": f"donor-{component_index:02d}-{int(label)}",
                        "role": role,
                        "features": features.tolist(),
                    }
                )
            component_index += 1
    return records


def test_detector_and_threshold_never_depend_on_pilot_gate_features() -> None:
    records = _records(role_signal=0.4)
    frozen = freeze_detector(records, _config())
    changed = copy.deepcopy(records)
    for record in changed:
        if record["partition"] == "pilot_gate":
            record["features"] = [99.0] * len(FEATURE_NAMES)

    repeated = freeze_detector(changed, _config())
    assert repeated["fit_record_ids"] == frozen["fit_record_ids"]
    assert repeated["threshold_record_ids"] == frozen["threshold_record_ids"]
    assert repeated["threshold"] == frozen["threshold"]
    np.testing.assert_allclose(repeated["coefficients"], frozen["coefficients"])


def test_component_or_media_crossing_partitions_is_rejected() -> None:
    records = _records()
    pilot = next(record for record in records if record["partition"] == "pilot_gate")
    pilot["target_video_id"] = "target-00"

    with pytest.raises(NuisanceValidationError, match="video.*crosses partitions"):
        freeze_detector(records, _config())


def test_permutation_swaps_all_targets_in_a_component_together() -> None:
    records = _records(partition_sizes={"scorer_fit": 2})
    for record in records[2:]:
        record["component_id"] = records[0]["component_id"]

    class AlternatingGenerator:
        calls = 0

        def integers(self, _low: int, _high: int) -> int:
            value = self.calls % 2
            self.calls += 1
            return value

    generator = AlternatingGenerator()
    permuted = _permuted_records(records, generator)  # type: ignore[arg-type]
    swapped = {
        original["target_id"]: original["role"] != changed["role"]
        for original, changed in zip(records, permuted)
    }

    assert generator.calls == 1
    assert len(set(swapped.values())) == 1


def test_question_key_sensitivity_has_a_locked_minimum() -> None:
    records = _records(repeated_pilot_key=True)
    report = evaluate_nuisance_records(records, _config(minimum_unseen_components=2))

    assert report["question_key_sensitivity"]["status"] == "insufficient_components"
    assert report["decision"]["nuisance_detection_status"] == "fail"
    assert "question_key_sensitivity_too_small" in report["decision"]["reason_codes"]


def test_recoverable_edit_role_fails_the_locked_gate() -> None:
    report = evaluate_nuisance_records(_records(role_signal=1.0), _config())

    assert report["evaluation"]["detector"]["pair_ranking_accuracy"] == 1.0
    assert report["decision"]["nuisance_detection_status"] == "fail"
    assert "primary_upper_bound_exceeds_ceiling" in report["decision"]["reason_codes"]


def test_pairwise_interaction_detector_rejects_xor_role_signal() -> None:
    records = []
    component_index = 0
    for partition, count in {
        "scorer_fit": 8,
        "threshold_calibration": 4,
        "pilot_gate": 4,
    }.items():
        for local_index in range(count):
            same = (0.0, 0.0) if local_index % 2 == 0 else (1.0, 1.0)
            opposite = (0.0, 1.0) if local_index % 2 == 0 else (1.0, 0.0)
            for role, values in (
                ("same_answer_nuisance", same),
                ("opposite_answer_candidate", opposite),
            ):
                features = [0.0] * len(FEATURE_NAMES)
                features[0], features[1] = values
                records.append(
                    {
                        "pair_id": f"xor-{component_index}-{role}",
                        "target_id": f"xor-target-{component_index}",
                        "component_id": f"xor-component-{component_index}",
                        "partition": partition,
                        "key_sha256": f"xor-key-{component_index}",
                        "target_video_id": f"xor-target-video-{component_index}",
                        "donor_video_id": f"xor-donor-{component_index}-{role}",
                        "role": role,
                        "features": features,
                    }
                )
            component_index += 1

    report = evaluate_nuisance_records(records, _config())

    assert set(report["frozen_detector_suite"]["models"]) == {
        "ridge_linear",
        "univariate_linear",
        "pairwise_interaction",
    }
    nonlinear = report["evaluation"]["detector_suite"]["models"]["pairwise_interaction"]
    assert nonlinear["role_recovery_point_estimate"] == 1.0
    assert report["decision"]["nuisance_detection_status"] == "fail"


def test_indistinguishable_roles_pass_with_source_and_permutation_controls() -> None:
    records = _records(role_signal=0.0)
    for record in records:
        record["features"] = [0.0] * len(FEATURE_NAMES)
    report = evaluate_nuisance_records(records, _config())

    assert report["human_evaluation_used"] is False
    assert report["outcome_based_sample_selection"] == "forbidden"
    assert report["evaluation"]["detector"]["pair_ranking_accuracy"] == 0.5
    assert report["evaluation"]["detector"]["balanced_accuracy"] == 0.5
    assert report["evaluation"]["detector"]["role_recovery_point_estimate"] == 0.5
    assert (
        report["evaluation"]["detector"]["conservative_primary_upper_confidence_bound"]
        >= report["evaluation"]["detector"]["conservative_upper_confidence_bound"]
    )
    assert report["evaluation"]["detector"]["exact_outcome_counts"] == {
        "correct": 0,
        "incorrect": 0,
        "ties": 4,
    }
    assert report["evaluation"]["detector"]["interval_method"] == (
        "component_cluster_percentile"
    )
    assert "wilson_interval" not in report["evaluation"]["detector"]
    assert len(report["evaluation"]["detector"]["exact_component_outcomes"]) == 4
    assert {
        outcome["component_id"]
        for outcome in report["evaluation"]["detector"]["exact_component_outcomes"]
    } == {f"component-{index:02d}" for index in range(9, 13)}
    assert report["evaluation"]["structural_chance"]["balanced_accuracy"] == 0.5
    assert report["evaluation"]["permutation"]["p_value"] > 0.05
    assert report["question_key_sensitivity"]["component_count"] == 4
    assert report["decision"] == {
        "nuisance_detection_status": "pass",
        "reason_codes": [],
    }


def test_feature_records_expose_only_engineering_inputs_and_identifiers() -> None:
    diagnostics = {}
    for video_id, duration in (("target", 10.0), ("same", 10.0), ("opposite", 12.0)):
        diagnostics[video_id] = parse_ffprobe_record(
            _probe(
                duration=duration,
                video_duration=duration,
                audio_duration=duration,
                width=640,
                height=480,
                frame_rate="25/1",
                sample_rate=48_000,
                channels=2,
            ),
            size_bytes=int(duration * 1_000),
        )
    pairs = []
    for suffix, role, donor in (
        ("s", "same_answer_nuisance", "same"),
        ("o", "opposite_answer_candidate", "opposite"),
    ):
        pairs.append(
            {
                "pair_id": f"pair-{suffix}",
                "component_id": "component-a",
                "partition": "scorer_fit",
                "key_sha256": "key-a",
                "role": role,
                "anchor": {
                    "question": "Which event happened?",
                    "options": ["alpha", "beta", "gamma"],
                    "answer": "alpha",
                },
                "target": {
                    "video_id": "target",
                    "question_id": 4,
                    "answer": "alpha",
                },
                "donor": {
                    "video_id": donor,
                    "question_id": 4,
                    "answer": "alpha" if suffix == "s" else "beta",
                },
            }
        )

    records = build_feature_records(pairs, diagnostics)
    assert len(records) == 2
    assert set(records[0]) == {
        "pair_id",
        "target_id",
        "component_id",
        "partition",
        "key_sha256",
        "target_video_id",
        "donor_video_id",
        "role",
        "features",
    }
    serialized = json.dumps(records)
    assert "Which event" not in serialized
    assert "alpha" not in serialized


def test_contract_is_exact_and_bound_to_pilot_and_media() -> None:
    contract = {
        "schema": "conflictbench.perception-nuisance-detection-contract.v1",
        "status": "preregistered_not_run",
        "input_digests": {
            "configuration_sha256": "d" * 64,
            "pilot_index_sha256": "a" * 64,
            "media_receipt_sha256": "e" * 64,
            "media_set_sha256": "b" * 64,
            "implementation_bundle_sha256": "f" * 64,
        },
        "partitions": {
            "fit": "scorer_fit",
            "threshold": "threshold_calibration",
            "evaluation": "pilot_gate",
        },
        "human_evaluation": "forbidden",
        "outcome_based_sample_selection": "forbidden",
    }
    validate_nuisance_contract(
        contract,
        expected_pilot_index_sha256="a" * 64,
        expected_media_receipt_sha256="e" * 64,
        expected_media_set_sha256="b" * 64,
        expected_configuration_sha256="d" * 64,
        expected_implementation_bundle_sha256="f" * 64,
    )
    contract["input_digests"]["media_set_sha256"] = "c" * 64
    with pytest.raises(NuisanceValidationError, match="input binding"):
        validate_nuisance_contract(
            contract,
            expected_pilot_index_sha256="a" * 64,
            expected_media_receipt_sha256="e" * 64,
            expected_media_set_sha256="b" * 64,
            expected_configuration_sha256="d" * 64,
            expected_implementation_bundle_sha256="f" * 64,
        )


def test_public_contract_builder_emits_the_exact_valid_schema(
    tmp_path: pathlib.Path,
) -> None:
    output = tmp_path / "contract.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(CONTRACT_BUILDER_PATH),
            "--configuration-sha256",
            "d" * 64,
            "--pilot-index-sha256",
            "a" * 64,
            "--media-receipt-sha256",
            "e" * 64,
            "--media-set-sha256",
            "b" * 64,
            "--implementation-bundle-sha256",
            "f" * 64,
            "--output",
            str(output),
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(pathlib.Path(__file__).parents[1] / "src"),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    contract = json.loads(output.read_text(encoding="utf-8"))
    assert contract == build_nuisance_contract(
        configuration_sha256="d" * 64,
        pilot_index_sha256="a" * 64,
        media_receipt_sha256="e" * 64,
        media_set_sha256="b" * 64,
        implementation_bundle_sha256="f" * 64,
    )
    assert os.stat(output).st_mode & 0o777 == 0o400


def test_public_contract_builder_can_bind_the_v2_preregistration(
    tmp_path: pathlib.Path,
) -> None:
    output = tmp_path / "contract-v2.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(CONTRACT_BUILDER_PATH),
            "--configuration-sha256",
            "d" * 64,
            "--pilot-index-sha256",
            "a" * 64,
            "--media-receipt-sha256",
            "e" * 64,
            "--media-set-sha256",
            "b" * 64,
            "--implementation-bundle-sha256",
            "f" * 64,
            "--shortcut-preregistration-sha256",
            "c" * 64,
            "--output",
            str(output),
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(pathlib.Path(__file__).parents[1] / "src"),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    contract = json.loads(output.read_text(encoding="utf-8"))
    assert contract["schema"] == (
        "conflictbench.perception-nuisance-detection-contract.v2"
    )
    assert contract["input_digests"]["shortcut_preregistration_sha256"] == "c" * 64


def test_post_edit_diagnostic_uses_final_output_measurements_without_humans() -> None:
    records = _records(role_signal=0.0)
    source = parse_ffprobe_record(
        _probe(
            duration=10.0,
            video_duration=10.0,
            audio_duration=10.0,
            width=640,
            height=480,
            frame_rate="25/1",
            sample_rate=48_000,
            channels=2,
        ),
        size_bytes=10_000,
    )
    source_diagnostics = {}
    output_diagnostics = {}
    rows = []
    for record in records:
        source_diagnostics[record["target_video_id"]] = source
        output_id = f"edited-{record['pair_id']}"
        width = 640 if record["role"] == "same_answer_nuisance" else 1280
        output_diagnostics[output_id] = parse_ffprobe_record(
            _probe(
                duration=10.0,
                video_duration=10.0,
                audio_duration=10.0,
                width=width,
                height=480,
                frame_rate="25/1",
                sample_rate=48_000,
                channels=2,
            ),
            size_bytes=10_000,
        )
        rows.append(
            {
                "pair_id": record["pair_id"],
                "target_id": record["target_id"],
                "component_id": record["component_id"],
                "partition": record["partition"],
                "key_sha256": record["key_sha256"],
                "target_video_id": record["target_video_id"],
                "output_video_id": output_id,
                "role": record["role"],
            }
        )

    built = build_post_edit_feature_records(
        rows, source_diagnostics, output_diagnostics
    )
    report = evaluate_post_edit_diagnostics(
        rows, source_diagnostics, output_diagnostics, _config()
    )
    assert len(built) == len(rows)
    assert report["diagnostic_stage"] == "final_edited_outputs"
    assert report["human_evaluation_used"] is False
    assert report["decision"]["nuisance_detection_status"] == "fail"


def test_handoff_builder_rejects_an_unvalidated_report() -> None:
    fake_report = {
        "decision": {"nuisance_detection_status": "pass", "reason_codes": []},
        "frozen_detector": {"threshold_record_sha256": "a" * 64},
    }
    with pytest.raises(NuisanceValidationError, match="report"):
        build_handoff_result(
            report=fake_report,
            configuration=_config(),
            report_sha256="b" * 64,
            pilot_index_sha256="c" * 64,
            media_receipt_sha256="d" * 64,
            media_set_sha256="e" * 64,
            nuisance_contract_sha256="f" * 64,
            configuration_sha256="0" * 64,
            implementation_bundle_sha256="1" * 64,
        )


def _complete_report(records: list[dict]) -> tuple[dict, dict[str, str]]:
    configuration = _config()
    report = evaluate_nuisance_records(records, configuration)
    report.pop("attestation_sha256")
    source_digests = {
        "__init__.py": "1" * 64,
        "runner.py": "2" * 64,
        "perception_nuisance_gate.py": "3" * 64,
        "perception_media_pilot.py": "4" * 64,
        "perception_candidate_audit.py": "5" * 64,
        "perception_omni_gate.py": "6" * 64,
    }
    bindings = {
        "configuration_sha256": "d" * 64,
        "pilot_index_sha256": "a" * 64,
        "media_receipt_sha256": "e" * 64,
        "media_set_sha256": "b" * 64,
        "nuisance_contract_sha256": "c" * 64,
        "implementation_bundle_sha256": _canonical_digest(source_digests),
    }
    report.update(
        {
            "configuration": configuration,
            "input_digests": bindings,
            "runtime": {
                "numpy_version": np.__version__,
                "ffprobe": {},
                "implementation_source_sha256": source_digests,
            },
            "counts": {
                "media_file_count": 1,
                "feature_record_count": len(records),
                "component_count": len({record["component_id"] for record in records}),
            },
            "records": records,
        }
    )
    report["attestation_sha256"] = _canonical_digest(report)
    return report, bindings


def test_handoff_result_is_canonical_and_tamper_evident() -> None:
    report, bindings = _complete_report(
        [{**record, "features": [0.0] * len(FEATURE_NAMES)} for record in _records()]
    )
    report_bytes = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    result = build_handoff_result(
        report=report,
        configuration=_config(),
        report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        pilot_index_sha256=bindings["pilot_index_sha256"],
        media_receipt_sha256=bindings["media_receipt_sha256"],
        media_set_sha256=bindings["media_set_sha256"],
        nuisance_contract_sha256=bindings["nuisance_contract_sha256"],
        configuration_sha256=bindings["configuration_sha256"],
        implementation_bundle_sha256=bindings["implementation_bundle_sha256"],
    )
    validate_handoff_result(
        result,
        report=report,
        expected_configuration=_config(),
        expected_pilot_index_sha256=bindings["pilot_index_sha256"],
        expected_media_receipt_sha256=bindings["media_receipt_sha256"],
        expected_media_set_sha256=bindings["media_set_sha256"],
        expected_nuisance_contract_sha256=bindings["nuisance_contract_sha256"],
        expected_configuration_sha256=bindings["configuration_sha256"],
        expected_implementation_bundle_sha256=bindings["implementation_bundle_sha256"],
        expected_report_sha256=hashlib.sha256(report_bytes).hexdigest(),
    )
    result["decision"]["nuisance_detection_status"] = "fail"
    with pytest.raises(NuisanceValidationError, match="attestation"):
        validate_handoff_result(
            result,
            report=report,
            expected_configuration=_config(),
            expected_pilot_index_sha256=bindings["pilot_index_sha256"],
            expected_media_receipt_sha256=bindings["media_receipt_sha256"],
            expected_media_set_sha256=bindings["media_set_sha256"],
            expected_nuisance_contract_sha256=bindings["nuisance_contract_sha256"],
            expected_configuration_sha256=bindings["configuration_sha256"],
            expected_implementation_bundle_sha256=bindings[
                "implementation_bundle_sha256"
            ],
            expected_report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        )


def test_handoff_recomputes_a_reissued_detailed_report_decision() -> None:
    records = [
        {**record, "features": [0.0] * len(FEATURE_NAMES)} for record in _records()
    ]
    report, bindings = _complete_report(records)
    report.pop("attestation_sha256")
    report["evaluation"]["detector"]["role_recovery_point_estimate"] = 1.0
    report["attestation_sha256"] = _canonical_digest(report)
    report_bytes = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    with pytest.raises(NuisanceValidationError, match="aggregate|decision"):
        build_handoff_result(
            report=report,
            configuration=_config(),
            report_sha256=hashlib.sha256(report_bytes).hexdigest(),
            pilot_index_sha256=bindings["pilot_index_sha256"],
            media_receipt_sha256=bindings["media_receipt_sha256"],
            media_set_sha256=bindings["media_set_sha256"],
            nuisance_contract_sha256=bindings["nuisance_contract_sha256"],
            configuration_sha256=bindings["configuration_sha256"],
            implementation_bundle_sha256=bindings["implementation_bundle_sha256"],
        )


def test_handoff_rejects_a_reissued_report_with_changed_locked_configuration() -> None:
    records = [
        {**record, "features": [0.0] * len(FEATURE_NAMES)} for record in _records()
    ]
    report, bindings = _complete_report(records)
    report.pop("attestation_sha256")
    report["configuration"]["gate"]["maximum_primary_upper_confidence_bound"] = 1.0
    report["attestation_sha256"] = _canonical_digest(report)
    report_bytes = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    with pytest.raises(NuisanceValidationError, match="locked configuration"):
        build_handoff_result(
            report=report,
            configuration=_config(),
            report_sha256=hashlib.sha256(report_bytes).hexdigest(),
            pilot_index_sha256=bindings["pilot_index_sha256"],
            media_receipt_sha256=bindings["media_receipt_sha256"],
            media_set_sha256=bindings["media_set_sha256"],
            nuisance_contract_sha256=bindings["nuisance_contract_sha256"],
            configuration_sha256=bindings["configuration_sha256"],
            implementation_bundle_sha256=bindings["implementation_bundle_sha256"],
        )


def test_v2_configuration_is_separate_from_the_frozen_v1_gate() -> None:
    configuration = json.loads(V2_CONFIG_PATH.read_text(encoding="utf-8"))
    assert configuration["schema_version"] == 2
    assert configuration["runner"] == "perception_edit_role_shortcut_gate"
    assert configuration["feature_set"]["names"] == list(FEATURE_NAMES)
    assert configuration["model"]["families"] == [
        "ridge_linear",
        "univariate_linear",
        "pairwise_interaction",
    ]
    assert configuration["protocol"]["minimum_unseen_question_components"] == 15
    assert configuration["uncertainty"]["bootstrap_repetitions"] == 10_000
    assert configuration["permutation"]["repetitions"] == 10_000
    assert configuration["shortcut_protocol"]["bootstrap_repetitions"] == 10_000
    assert configuration["shortcut_protocol"]["preregistration_schema"] == (
        "conflictbench.perception-shortcut-preregistration.v2"
    )
    assert configuration["shortcut_protocol"]["output_schema"] == (
        "conflictbench.perception-shortcut-output.v2"
    )
    assert configuration["shortcut_protocol"]["replay_worker_request_schema"] == (
        "conflictbench.perception-replay-request.v2"
    )
    assert configuration["gate"] == {
        "maximum_primary_upper_confidence_bound": 0.55,
        "maximum_unseen_question_upper_confidence_bound": 0.55,
        "maximum_point_lift_over_structural_chance": 0.10,
        "maximum_structural_chance_balanced_accuracy": 0.60,
        "minimum_permutation_p_value_exclusive": 0.05,
    }
    assert configuration["shortcut_gate"] == {
        "maximum_metadata_balanced_accuracy_ucb_exclusive": 0.55,
        "maximum_question_only_balanced_accuracy_ucb_exclusive": 0.55,
        "maximum_question_blind_balanced_accuracy_ucb_exclusive": 0.55,
        "minimum_remux_accuracy_difference_exclusive": -0.02,
        "remux_p_gold_minimum": "negative_semantic_fit_epsilon",
        "equivalence_interval": [-0.05, 0.05],
        "maximum_shuffled_question_balanced_accuracy_ucb_exclusive": 0.55,
        "minimum_aligned_minus_shuffled_lcb_inclusive": 0.10,
    }


def test_historical_pilot_is_too_small_for_the_v2_metadata_equivalence_gate() -> None:
    configuration = json.loads(V2_CONFIG_PATH.read_text(encoding="utf-8"))
    configuration["uncertainty"]["bootstrap_repetitions"] = 128
    configuration["permutation"]["repetitions"] = 128
    records = _records(
        partition_sizes={
            "scorer_fit": 20,
            "threshold_calibration": 20,
            "pilot_gate": 20,
        }
    )
    for record in records:
        record["features"] = [0.0] * len(FEATURE_NAMES)

    with pytest.raises(NuisanceValidationError, match="at least 10000"):
        evaluate_nuisance_records(records, configuration)


def test_v2_cannot_enter_the_legacy_wilson_report_path() -> None:
    with pytest.raises(NuisanceValidationError, match="dedicated shortcut evaluator"):
        evaluate_nuisance_records(_v2_metadata_records(), _v2_config())


def test_v1_unseen_question_bound_preserves_its_frozen_strict_boundary() -> None:
    reasons = nuisance_gate_module._decision_reason_codes(
        detector={
            "conservative_primary_upper_confidence_bound": 0.5,
            "pair_ranking_accuracy": 0.5,
            "role_recovery_point_estimate": 0.5,
        },
        structural_chance={
            "pair_ranking_accuracy": 0.5,
            "role_recovery_point_estimate": 0.5,
            "balanced_accuracy": 0.5,
        },
        question_sensitivity={
            "status": "evaluated",
            "conservative_primary_upper_confidence_bound": 0.6,
        },
        permutation={"p_value": 1.0},
        gate={
            "maximum_primary_upper_confidence_bound": 0.6,
            "maximum_unseen_question_upper_confidence_bound": 0.6,
            "maximum_point_lift_over_structural_chance": 0.1,
            "maximum_structural_chance_balanced_accuracy": 0.6,
            "minimum_permutation_p_value_exclusive": 0.05,
        },
    )

    assert "unseen_question_upper_bound_exceeds_ceiling" not in reasons


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _json_file_digest(value: object) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def test_media_receipt_reauthenticates_membership_bytes_and_media_set(
    tmp_path: pathlib.Path,
) -> None:
    media_root = tmp_path / "selected"
    media_root.mkdir()
    files = []
    for index in (1, 2):
        path = media_root / f"video_{index:04d}.mp4"
        path.write_bytes(f"media-{index}".encode())
        path.chmod(0o440)
        files.append(
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "zip_crc32": f"{index:08x}",
                "resumed_existing": False,
                "stream_types": ["audio", "video"],
            }
        )
    normalized = [
        {
            "video_id": pathlib.Path(item["filename"]).stem,
            "filename": item["filename"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
            "stream_types": item["stream_types"],
        }
        for item in files
    ]
    receipt = {
        "schema": "conflictbench.perception-pilot-media-receipt.v1",
        "status": "selected_media_extracted",
        "input_digests": {
            "train_archive_sha256": "f" * 64,
            "pilot_index_sha256": "a" * 64,
        },
        "media_root_name": "selected",
        "media_file_count": 2,
        "media_files": files,
        "stream_probe": {
            "filename": "ffprobe",
            "sha256": "e" * 64,
            "version": "ffprobe fixture",
        },
    }
    observed = validate_media_receipt(
        receipt,
        media_root=media_root,
        expected_pilot_index_sha256="a" * 64,
        expected_media_set_sha256=_canonical_digest(normalized),
        expected_video_ids={"video_0001", "video_0002"},
    )
    assert observed == normalized

    (media_root / "video_0002.mp4").chmod(0o640)
    (media_root / "video_0002.mp4").write_bytes(b"changed")
    with pytest.raises(NuisanceValidationError, match="size|digest|writable"):
        validate_media_receipt(
            receipt,
            media_root=media_root,
            expected_pilot_index_sha256="a" * 64,
            expected_media_set_sha256=_canonical_digest(normalized),
            expected_video_ids={"video_0001", "video_0002"},
        )


def test_probe_media_file_runs_a_bounded_external_probe(tmp_path: pathlib.Path) -> None:
    probe = tmp_path / "ffprobe"
    payload = _probe(
        duration=10.0,
        video_duration=10.0,
        audio_duration=10.0,
        width=640,
        height=480,
        frame_rate="25/1",
        sample_rate=48_000,
        channels=2,
    )
    probe.write_text(
        "#!/bin/sh\nprintf '%s\\n' '" + json.dumps(payload) + "'\n",
        encoding="utf-8",
    )
    probe.chmod(0o755)
    media = tmp_path / "video_0001.mp4"
    media.write_bytes(b"media")

    observed = probe_media_file(probe, media, expected_size_bytes=5)
    assert observed["video"]["width"] == 640
    assert observed["audio"]["sample_rate"] == 48_000


def test_gate_outputs_are_write_once_and_bind_the_detailed_report(
    tmp_path: pathlib.Path,
) -> None:
    records = _records()
    for record in records:
        record["features"] = [0.0] * len(FEATURE_NAMES)
    report, bindings = _complete_report(records)
    report_bytes = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    result = build_handoff_result(
        report=report,
        configuration=_config(),
        report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        pilot_index_sha256=bindings["pilot_index_sha256"],
        media_receipt_sha256=bindings["media_receipt_sha256"],
        media_set_sha256=bindings["media_set_sha256"],
        nuisance_contract_sha256=bindings["nuisance_contract_sha256"],
        configuration_sha256=bindings["configuration_sha256"],
        implementation_bundle_sha256=bindings["implementation_bundle_sha256"],
    )
    report_path = tmp_path / "report.json"
    result_path = tmp_path / "result.json"
    write_gate_outputs(
        report_path,
        result_path,
        report=report,
        result=result,
        configuration=_config(),
    )
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert json.loads(result_path.read_text(encoding="utf-8")) == result
    assert os.stat(report_path).st_mode & 0o777 == 0o400
    with pytest.raises(NuisanceValidationError, match="already exists"):
        write_gate_outputs(
            report_path,
            result_path,
            report=report,
            result=result,
            configuration=_config(),
        )


def test_public_runner_exposes_every_hash_bound_input() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--help"],
        env={
            **os.environ,
            "PYTHONPATH": str(pathlib.Path(__file__).parents[1] / "src"),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    for option in (
        "--config-sha256",
        "--pilot-index-sha256",
        "--media-receipt-sha256",
        "--media-set-sha256",
        "--implementation-bundle-sha256",
        "--nuisance-contract-sha256",
        "--shortcut-preregistration",
        "--shortcut-preregistration-sha256",
        "--shortcut-output",
        "--shortcut-output-sha256",
        "--power-record",
        "--power-record-sha256",
        "--raw-conflict-index-verification-sha256",
        "--source-only-output-verification-sha256",
        "--semantic-gate-output-verification-sha256",
        "--derived-media-v4-index-verification-sha256",
        "--replay-worker",
        "--replay-worker-sha256",
        "--report",
        "--result",
    ):
        assert option in completed.stdout


def test_public_runner_executes_with_canonical_source_roles(
    tmp_path: pathlib.Path,
) -> None:
    fixture_path = pathlib.Path(__file__).with_name("test_perception_omni_gate.py")
    specification = importlib.util.spec_from_file_location(
        "_perception_omni_gate_cli_fixture", fixture_path
    )
    assert specification is not None and specification.loader is not None
    fixture_module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(fixture_module)
    pilot = fixture_module._pilot_index()

    def write_json(path: pathlib.Path, value: object) -> str:
        payload = (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        path.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    config = _config(minimum_unseen_components=2)
    config["uncertainty"]["bootstrap_repetitions"] = 2
    config["permutation"]["repetitions"] = 2
    config_path = tmp_path / "config.json"
    config_sha256 = write_json(config_path, config)
    pilot_path = tmp_path / "pilot-index.json"
    pilot_sha256 = write_json(pilot_path, pilot)

    video_ids = sorted(
        {
            pair[source_role]["video_id"]
            for pair in pilot["candidate_index"]
            for source_role in ("target", "donor")
        }
    )
    media_root = tmp_path / "selected"
    media_root.mkdir()
    media_files = []
    normalized = []
    for index, video_id in enumerate(video_ids):
        media_path = media_root / f"{video_id}.mp4"
        media_path.write_bytes(f"fixture:{video_id}".encode())
        media_path.chmod(0o440)
        item = {
            "filename": media_path.name,
            "size_bytes": media_path.stat().st_size,
            "sha256": hashlib.sha256(media_path.read_bytes()).hexdigest(),
            "zip_crc32": f"{index:08x}",
            "resumed_existing": False,
            "stream_types": ["audio", "video"],
        }
        media_files.append(item)
        normalized.append(
            {
                "video_id": video_id,
                "filename": media_path.name,
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
                "stream_types": item["stream_types"],
            }
        )

    ffprobe_path = tmp_path / "ffprobe"
    probe_payload = _probe(
        duration=10.0,
        video_duration=10.0,
        audio_duration=10.0,
        width=640,
        height=480,
        frame_rate="25/1",
        sample_rate=48_000,
        channels=2,
    )
    ffprobe_path.write_text(
        "#!/bin/sh\n"
        'if [ "${1:-}" = "-version" ]; then\n'
        "  echo 'ffprobe fixture'\n"
        "else\n"
        f"  printf '%s\\n' '{json.dumps(probe_payload)}'\n"
        "fi\n",
        encoding="utf-8",
    )
    ffprobe_path.chmod(0o755)

    receipt = {
        "schema": "conflictbench.perception-pilot-media-receipt.v1",
        "status": "selected_media_extracted",
        "input_digests": {
            "train_archive_sha256": "f" * 64,
            "pilot_index_sha256": pilot_sha256,
        },
        "media_root_name": media_root.name,
        "media_file_count": len(media_files),
        "media_files": media_files,
        "stream_probe": {
            "filename": ffprobe_path.name,
            "sha256": hashlib.sha256(ffprobe_path.read_bytes()).hexdigest(),
            "version": "ffprobe fixture",
        },
    }
    receipt_path = tmp_path / "media-receipt.json"
    receipt_sha256 = write_json(receipt_path, receipt)
    media_set_sha256 = _canonical_digest(normalized)

    repository_root = pathlib.Path(__file__).parents[1]
    source_paths = {
        "__init__.py": repository_root / "src/conflictbench/__init__.py",
        "runner.py": RUNNER_PATH,
        "perception_nuisance_gate.py": repository_root
        / "src/conflictbench/perception_nuisance_gate.py",
        "perception_media_pilot.py": repository_root
        / "src/conflictbench/perception_media_pilot.py",
        "perception_candidate_audit.py": repository_root
        / "src/conflictbench/perception_candidate_audit.py",
        "perception_omni_gate.py": repository_root
        / "src/conflictbench/perception_omni_gate.py",
    }
    source_digests = {
        role: hashlib.sha256(path.read_bytes()).hexdigest()
        for role, path in source_paths.items()
    }
    implementation_bundle_sha256 = _canonical_digest(
        dict(sorted(source_digests.items()))
    )
    contract = build_nuisance_contract(
        configuration_sha256=config_sha256,
        pilot_index_sha256=pilot_sha256,
        media_receipt_sha256=receipt_sha256,
        media_set_sha256=media_set_sha256,
        implementation_bundle_sha256=implementation_bundle_sha256,
    )
    contract_path = tmp_path / "nuisance-contract.json"
    contract_sha256 = write_json(contract_path, contract)
    report_path = tmp_path / "nuisance-report.json"
    result_path = tmp_path / "nuisance-result.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--config",
            str(config_path),
            "--config-sha256",
            config_sha256,
            "--pilot-index",
            str(pilot_path),
            "--pilot-index-sha256",
            pilot_sha256,
            "--media-receipt",
            str(receipt_path),
            "--media-receipt-sha256",
            receipt_sha256,
            "--media-root",
            str(media_root),
            "--media-set-sha256",
            media_set_sha256,
            "--implementation-bundle-sha256",
            implementation_bundle_sha256,
            "--nuisance-contract",
            str(contract_path),
            "--nuisance-contract-sha256",
            contract_sha256,
            "--ffprobe",
            str(ffprobe_path),
            "--report",
            str(report_path),
            "--result",
            str(result_path),
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(repository_root / "src"),
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["runtime"]["implementation_source_sha256"] == dict(
        sorted(source_digests.items())
    )
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == "complete"


def _v2_config() -> dict:
    config = _config(minimum_unseen_components=15)
    config["schema_version"] = 2
    config["runner"] = "perception_edit_role_shortcut_gate"
    config["uncertainty"]["bootstrap_repetitions"] = 10_000
    config["uncertainty"]["method"] = "component_cluster_percentile_and_max_statistic"
    config["permutation"]["repetitions"] = 10_000
    config["gate"]["maximum_primary_upper_confidence_bound"] = 0.55
    config["gate"]["maximum_unseen_question_upper_confidence_bound"] = 0.55
    config["shortcut_protocol"] = {
        "preregistration_schema": (
            "conflictbench.perception-shortcut-preregistration.v2"
        ),
        "output_schema": "conflictbench.perception-shortcut-output.v2",
        "report_schema": "conflictbench.perception-shortcut-report.v2",
        "result_schema": "conflictbench.perception-shortcut-result.v2",
        "power_schema": "conflictbench.perception-shortcut-power.v2",
        "verified_dependency_schema": (
            "conflictbench.perception-verified-dependency.v2"
        ),
        "replay_worker_request_schema": "conflictbench.perception-replay-request.v2",
        "replay_worker_response_schema": (
            "conflictbench.perception-replay-worker-response.v2"
        ),
        "question_blind_request_schema": (
            "conflictbench.perception-question-blind-av-mismatch-request.v2"
        ),
        "question_blind_output_schema": (
            "conflictbench.perception-question-blind-av-mismatch-output.v2"
        ),
        "question_blind_allowed_request_fields": ["schema", "audio", "video"],
        "question_blind_forbidden_inputs": [
            "question",
            "options",
            "answer",
            "condition",
            "orientation",
            "option_permutation",
            "component_id",
            "question_key_sha256",
        ],
        "replay_verification": "two_fresh_exact_replays",
        "question_only_replay_verification": "two_fresh_exact_replays",
        "dependency_roles": [
            "raw_conflict_index",
            "source_only",
            "semantic_gate",
            "derived_media_v4",
        ],
        "derived_media_authentication": "exact_inventory_before_and_after_replay",
        "chronology": "preregistration_bound_before_completed_output",
        "power_simulations": 10_000,
        "uncertainty_unit": "connected_analysis_component",
        "uncertainty_method": (
            "one_sided_component_cluster_percentile_bonferroni_familywise"
        ),
        "confidence_level": 0.95,
        "bootstrap_repetitions": 10_000,
        "bootstrap_seed": 271_828,
    }
    config["shortcut_gate"] = {
        "maximum_metadata_balanced_accuracy_ucb_exclusive": 0.55,
        "maximum_question_only_balanced_accuracy_ucb_exclusive": 0.55,
        "maximum_question_blind_balanced_accuracy_ucb_exclusive": 0.55,
        "minimum_remux_accuracy_difference_exclusive": -0.02,
        "remux_p_gold_minimum": "negative_semantic_fit_epsilon",
        "equivalence_interval": [-0.05, 0.05],
        "maximum_shuffled_question_balanced_accuracy_ucb_exclusive": 0.55,
        "minimum_aligned_minus_shuffled_lcb_inclusive": 0.10,
    }
    return config


def _raw_output(payload: dict) -> tuple[str, str]:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return raw, hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _mismatch_output(probability: float) -> dict:
    raw, digest = _raw_output({"mismatch_probability": probability})
    return {
        "schema": "conflictbench.perception-question-blind-av-mismatch-output.v2",
        "raw_output": raw,
        "raw_output_sha256": digest,
        "mismatch_probability": probability,
    }


def _relation_output(prediction: str) -> dict:
    raw, digest = _raw_output({"prediction": prediction})
    return {
        "schema": "conflictbench.perception-conflict-relation-output.v2",
        "raw_output": raw,
        "raw_output_sha256": digest,
        "prediction": prediction,
    }


def _choice_output(probabilities: dict[str, float]) -> dict:
    raw, digest = _raw_output({"choice_probabilities": probabilities})
    return {
        "schema": "conflictbench.perception-source-choice-output.v2",
        "raw_output": raw,
        "raw_output_sha256": digest,
        "choice_probabilities": probabilities,
    }


def _control_request(
    *,
    task: str,
    question: str,
    options: list[str],
    media_request: dict,
) -> dict:
    return {
        "schema": "conflictbench.perception-control-request.v2",
        "task": task,
        "question": question,
        "options": options,
        "audio": copy.deepcopy(media_request["audio"]),
        "video": copy.deepcopy(media_request["video"]),
    }


def _shortcut_output_v2(
    config_sha256: str, *, component_count: int = 15, component_offset: int = 8
) -> dict:
    question_only_records = []
    question_blind_records = []
    remux_records = []
    orientation_records = []
    permutation_records = []
    shuffled_records = []
    for component_index in range(component_count):
        inventory_index = component_index + component_offset
        component_id = f"component-{inventory_index:02d}"
        for label_index, (mismatch_label, gold_relation) in enumerate(
            (("matched", "AGREE"), ("mismatched", "CONFLICT"))
        ):
            record_id = f"pair-{inventory_index:02d}-{label_index}"
            request = {
                "schema": (
                    "conflictbench.perception-question-blind-av-mismatch-request.v2"
                ),
                "audio": {
                    "relative_path": f"media/{record_id}.wav",
                    "sha256": hashlib.sha256(f"audio:{record_id}".encode()).hexdigest(),
                },
                "video": {
                    "relative_path": f"media/{record_id}.mp4",
                    "sha256": hashlib.sha256(f"video:{record_id}".encode()).hexdigest(),
                },
            }
            output = _mismatch_output(0.5)
            question_only_request = {
                "schema": "conflictbench.perception-replay-request.v2",
                "task": "question_only_conflict",
                "inputs": {
                    "question": f"Question {record_id}",
                    "options": ["first", "second", "third"],
                },
            }
            question_only_output = _relation_output("AGREE")
            question_only_records.append(
                {
                    "record_id": record_id,
                    "component_id": component_id,
                    "gold_relation": gold_relation,
                    "request": question_only_request,
                    "request_sha256": _canonical_digest(question_only_request),
                    "primary_output": copy.deepcopy(question_only_output),
                    "replay_output": copy.deepcopy(question_only_output),
                }
            )
            question_blind_records.append(
                {
                    "record_id": record_id,
                    "component_id": component_id,
                    "expected_relation": mismatch_label,
                    "request": request,
                    "request_sha256": _canonical_digest(request),
                    "primary_output": copy.deepcopy(output),
                    "replay_output": copy.deepcopy(output),
                }
            )

            gold = "A" if label_index == 0 else "B"
            probabilities = (
                {"A": 0.8, "B": 0.1, "C": 0.1}
                if gold == "A"
                else {"A": 0.1, "B": 0.8, "C": 0.1}
            )
            clean_request = _control_request(
                task="source_choice",
                question=f"Question {record_id}; answer={gold}",
                options=["first", "second", "third"],
                media_request=request,
            )
            remux_request = copy.deepcopy(clean_request)
            remux_records.append(
                {
                    "record_id": record_id,
                    "component_id": component_id,
                    "gold_answer": gold,
                    "clean_request": clean_request,
                    "clean_request_sha256": _canonical_digest(clean_request),
                    "clean_output": _choice_output(probabilities),
                    "remux_request": remux_request,
                    "remux_request_sha256": _canonical_digest(remux_request),
                    "remux_output": _choice_output(probabilities),
                }
            )
            relation_output = _relation_output(gold_relation)
            audio_over_video_request = _control_request(
                task="conflict_relation",
                question=f"Question {record_id}; relation={gold_relation}",
                options=["first", "second", "third"],
                media_request=request,
            )
            video_over_audio_request = copy.deepcopy(audio_over_video_request)
            orientation_records.append(
                {
                    "record_id": record_id,
                    "component_id": component_id,
                    "gold_relation": gold_relation,
                    "audio_over_video_request": audio_over_video_request,
                    "audio_over_video_request_sha256": _canonical_digest(
                        audio_over_video_request
                    ),
                    "audio_over_video_output": copy.deepcopy(relation_output),
                    "video_over_audio_request": video_over_audio_request,
                    "video_over_audio_request_sha256": _canonical_digest(
                        video_over_audio_request
                    ),
                    "video_over_audio_output": copy.deepcopy(relation_output),
                }
            )
            permutation_requests = {
                str(permutation): _control_request(
                    task="conflict_relation",
                    question=f"Question {record_id}; relation={gold_relation}",
                    options=["first", "second", "third"][permutation:]
                    + ["first", "second", "third"][:permutation],
                    media_request=request,
                )
                for permutation in range(3)
            }
            permutation_records.append(
                {
                    "record_id": record_id,
                    "component_id": component_id,
                    "gold_relation": gold_relation,
                    "requests": permutation_requests,
                    "request_sha256s": {
                        permutation: _canonical_digest(permutation_request)
                        for permutation, permutation_request in permutation_requests.items()
                    },
                    "outputs": {
                        "0": copy.deepcopy(relation_output),
                        "1": copy.deepcopy(relation_output),
                        "2": copy.deepcopy(relation_output),
                    },
                }
            )
            aligned_request = copy.deepcopy(audio_over_video_request)
            shuffled_request = _control_request(
                task="conflict_relation",
                question=f"Shuffled question {record_id}; relation=AGREE",
                options=["first", "second", "third"],
                media_request=request,
            )
            shuffled_records.append(
                {
                    "record_id": record_id,
                    "component_id": component_id,
                    "gold_relation": gold_relation,
                    "aligned_request": aligned_request,
                    "aligned_request_sha256": _canonical_digest(aligned_request),
                    "aligned_output": copy.deepcopy(relation_output),
                    "shuffled_request": shuffled_request,
                    "shuffled_request_sha256": _canonical_digest(shuffled_request),
                    "shuffled_output": _relation_output("AGREE"),
                }
            )

    value = {
        "schema": "conflictbench.perception-shortcut-output.v2",
        "status": "complete_replay_recorded",
        "shortcut_preregistration_sha256": "e" * 64,
        "input_digests": {
            "configuration_sha256": config_sha256,
            "raw_conflict_index_sha256": "1" * 64,
            "media_set_sha256": "2" * 64,
            "primary_system_sha256": "3" * 64,
            "semantic_gate_sha256": "4" * 64,
            "semantic_fit_output_sha256": "9" * 64,
            "question_blind_model_sha256": "5" * 64,
            "question_blind_implementation_sha256": "6" * 64,
            "question_blind_configuration_sha256": "a" * 64,
            "gate_implementation_bundle_sha256": "7" * 64,
        },
        "question_only_conflict": {
            "transform": "remove_audio_and_video",
            "records": question_only_records,
        },
        "question_blind_audiovisual_mismatch": {
            "transform": "remove_question_and_options",
            "threshold": 0.5,
            "threshold_record_sha256": "8" * 64,
            "records": question_blind_records,
        },
        "remux_noninferiority": {
            "transform": "container_remux_only",
            "epsilon_p_gold": 0.03,
            "epsilon_source": {
                "quantile": 0.95,
                "semantic_fit_output_sha256": "9" * 64,
            },
            "records": remux_records,
        },
        "orientation_symmetry": {
            "transform": "reverse_modality_precedence",
            "records": orientation_records,
        },
        "option_permutation_equivalence": {
            "transform": "cyclic_option_rotation",
            "records": permutation_records,
        },
        "shuffled_question": {
            "transform": "cross_component_question_derangement",
            "records": shuffled_records,
        },
    }
    value["attestation_sha256"] = _canonical_digest(value)
    return value


def _bind_output_to_preregistration_v2(
    config_sha256: str,
    value: dict,
    *,
    metadata_records: list[dict] | None = None,
) -> tuple[dict, str]:
    question_blind_records = value["question_blind_audiovisual_mismatch"]["records"]
    records = sorted(
        [
            {
                "record_id": record["record_id"],
                "component_id": record["component_id"],
            }
            for record in question_blind_records
        ],
        key=lambda record: record["record_id"],
    )
    media_by_path = {}
    for record in question_blind_records:
        for modality in ("audio", "video"):
            media = record["request"][modality]
            media_by_path[media["relative_path"]] = {
                "relative_path": media["relative_path"],
                "size_bytes": len(f"{modality}:{record['record_id']}".encode()),
                "sha256": media["sha256"],
            }
    digests = value["input_digests"]
    if metadata_records is None:
        component_ids = sorted({record["component_id"] for record in records})
        component_offset = int(component_ids[0].rsplit("-", 1)[1])
        metadata_records = _v2_metadata_records(
            component_count=len(component_ids), component_offset=component_offset
        )
    preregistration = nuisance_gate_module.build_shortcut_preregistration_v2(
        configuration_sha256=config_sha256,
        gate_implementation_bundle_sha256=digests["gate_implementation_bundle_sha256"],
        dependency_digests={
            "raw_conflict_index_sha256": digests["raw_conflict_index_sha256"],
            "raw_conflict_index_verification_sha256": "b" * 64,
            "source_only_output_sha256": digests["primary_system_sha256"],
            "source_only_replay_verification_sha256": "c" * 64,
            "semantic_gate_output_sha256": digests["semantic_gate_sha256"],
            "semantic_gate_replay_verification_sha256": "d" * 64,
            "derived_media_v4_index_sha256": digests["media_set_sha256"],
            "derived_media_v4_verification_sha256": "e" * 64,
            "power_output_sha256": "f" * 64,
        },
        replay_worker_digests={
            "implementation_sha256": digests["question_blind_implementation_sha256"],
            "configuration_sha256": digests["question_blind_configuration_sha256"],
            "model_sha256": digests["question_blind_model_sha256"],
        },
        records=records,
        media_files=sorted(
            media_by_path.values(), key=lambda item: item["relative_path"]
        ),
        metadata_records=metadata_records,
        control_plan=nuisance_gate_module.project_shortcut_control_plan_v2(value),
    )
    preregistration_sha256 = _canonical_digest(preregistration)
    value["input_digests"]["media_set_sha256"] = preregistration["inventory"][
        "media_inventory_sha256"
    ]
    value["shortcut_preregistration_sha256"] = preregistration_sha256
    value["attestation_sha256"] = _canonical_digest(
        {key: item for key, item in value.items() if key != "attestation_sha256"}
    )
    return preregistration, preregistration_sha256


def _replay_worker(tmp_path: pathlib.Path, value: dict) -> SubprocessReplayWorkerSpec:
    worker_path = tmp_path / "replay-worker"
    worker_path.write_text(
        "#!/usr/bin/env python3\n"
        "import hashlib, json, sys\n"
        "envelope = json.loads(sys.stdin.read())\n"
        "task = envelope['request']['task']\n"
        "if task == 'question_blind_audiovisual_mismatch':\n"
        "    assert __import__('pathlib').Path('audio.bin').is_file()\n"
        "    assert __import__('pathlib').Path('video.bin').is_file()\n"
        "    payload = {'mismatch_probability': 0.5}\n"
        "    schema = 'conflictbench.perception-question-blind-av-mismatch-output.v2'\n"
        "elif task == 'question_only_conflict':\n"
        "    payload = {'prediction': 'AGREE'}\n"
        "    schema = 'conflictbench.perception-conflict-relation-output.v2'\n"
        "elif task == 'source_choice':\n"
        "    question = envelope['request']['inputs']['question']\n"
        "    answer = question.rsplit('answer=', 1)[1]\n"
        "    payload = {'choice_probabilities': {choice: "
        "(0.8 if choice == answer else 0.1) for choice in ('A', 'B', 'C')}}\n"
        "    schema = 'conflictbench.perception-source-choice-output.v2'\n"
        "elif task == 'conflict_relation':\n"
        "    question = envelope['request']['inputs']['question']\n"
        "    payload = {'prediction': question.rsplit('relation=', 1)[1]}\n"
        "    schema = 'conflictbench.perception-conflict-relation-output.v2'\n"
        "else:\n"
        "    raise SystemExit(2)\n"
        "raw = json.dumps(payload, sort_keys=True, separators=(',', ':'))\n"
        "output = {'schema': schema, 'raw_output': raw, "
        "'raw_output_sha256': hashlib.sha256(raw.encode()).hexdigest(), "
        "**payload}\n"
        "response = {'schema': "
        "'conflictbench.perception-replay-worker-response.v2', "
        "'bindings': envelope['bindings'], 'output': output}\n"
        "sys.stdout.write(json.dumps(response, sort_keys=True, "
        "separators=(',', ':')) + '\\n')\n",
        encoding="utf-8",
    )
    worker_path.chmod(0o500)
    configuration_path = tmp_path / "replay-configuration.json"
    configuration_path.write_text('{"mode":"test"}\n', encoding="utf-8")
    configuration_path.chmod(0o400)
    model_path = tmp_path / "replay-model.bin"
    model_path.write_bytes(b"deterministic-test-model")
    model_path.chmod(0o400)
    executable_sha256 = hashlib.sha256(worker_path.read_bytes()).hexdigest()
    configuration_sha256 = hashlib.sha256(configuration_path.read_bytes()).hexdigest()
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    value["input_digests"]["question_blind_implementation_sha256"] = executable_sha256
    value["input_digests"]["question_blind_configuration_sha256"] = configuration_sha256
    value["input_digests"]["question_blind_model_sha256"] = model_sha256
    value["attestation_sha256"] = _canonical_digest(
        {key: item for key, item in value.items() if key != "attestation_sha256"}
    )
    digests = value["input_digests"]
    return SubprocessReplayWorkerSpec(
        executable_path=worker_path,
        executable_sha256=executable_sha256,
        configuration_path=configuration_path,
        configuration_sha256=digests["question_blind_configuration_sha256"],
        model_path=model_path,
        model_sha256=digests["question_blind_model_sha256"],
    )


def _prepared_v2(
    tmp_path: pathlib.Path,
    config_sha256: str,
    *,
    component_count: int = 15,
    component_offset: int = 8,
) -> tuple[dict, SubprocessReplayWorkerSpec, dict, str, pathlib.Path]:
    output = _shortcut_output_v2(
        config_sha256,
        component_count=component_count,
        component_offset=component_offset,
    )
    worker = _replay_worker(tmp_path, output)
    preregistration, preregistration_sha256 = _bind_output_to_preregistration_v2(
        config_sha256, output
    )
    media_root = tmp_path / "derived-media"
    for record in output["question_blind_audiovisual_mismatch"]["records"]:
        for modality in ("audio", "video"):
            media = record["request"][modality]
            path = media_root / media["relative_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{modality}:{record['record_id']}".encode())
            path.chmod(0o400)
    return output, worker, preregistration, preregistration_sha256, media_root


def _v2_metadata_records(
    *, component_count: int = 15, component_offset: int = 8
) -> list[dict]:
    fit_components = component_offset // 2
    threshold_components = component_offset - fit_components
    records = _records(
        role_signal=0.0,
        partition_sizes={
            "scorer_fit": fit_components,
            "threshold_calibration": threshold_components,
            "pilot_gate": component_count,
        },
    )
    for record in records:
        record["features"] = [0.0] * len(FEATURE_NAMES)
    return records


def _v2_preregistration(config_sha256: str) -> dict:
    output = _shortcut_output_v2(config_sha256, component_count=2, component_offset=8)
    question_blind_records = output["question_blind_audiovisual_mismatch"]["records"]
    records = sorted(
        [
            {
                "record_id": record["record_id"],
                "component_id": record["component_id"],
            }
            for record in question_blind_records
        ],
        key=lambda item: item["record_id"],
    )
    media_by_path = {}
    for record in question_blind_records:
        for modality in ("audio", "video"):
            media = record["request"][modality]
            media_by_path[media["relative_path"]] = {
                "relative_path": media["relative_path"],
                "size_bytes": len(f"{modality}:{record['record_id']}".encode()),
                "sha256": media["sha256"],
            }
    return nuisance_gate_module.build_shortcut_preregistration_v2(
        configuration_sha256=config_sha256,
        gate_implementation_bundle_sha256="7" * 64,
        dependency_digests={
            "raw_conflict_index_sha256": "1" * 64,
            "raw_conflict_index_verification_sha256": "2" * 64,
            "source_only_output_sha256": "3" * 64,
            "source_only_replay_verification_sha256": "4" * 64,
            "semantic_gate_output_sha256": "5" * 64,
            "semantic_gate_replay_verification_sha256": "6" * 64,
            "derived_media_v4_index_sha256": "8" * 64,
            "derived_media_v4_verification_sha256": "9" * 64,
            "power_output_sha256": "a" * 64,
        },
        replay_worker_digests={
            "implementation_sha256": "b" * 64,
            "configuration_sha256": "c" * 64,
            "model_sha256": "d" * 64,
        },
        records=records,
        media_files=sorted(
            media_by_path.values(), key=lambda item: item["relative_path"]
        ),
        metadata_records=_v2_metadata_records(component_count=2),
        control_plan=nuisance_gate_module.project_shortcut_control_plan_v2(output),
    )


def _v2_power_dgp(
    *,
    sample_size_components: int = 2,
    effect_distance: float = 0.05,
    marginal_standard_deviation: float = 0.005,
    within_component_correlation: float = 0.5,
    remux_p_gold_epsilon: float = 0.03,
) -> dict:
    endpoint_count = len(nuisance_gate_module._V2_POWER_ENDPOINTS)
    rules = {
        "metadata_balanced_accuracy": {
            "type": "upper_confidence_bound_below",
            "upper_boundary": 0.55,
        },
        "question_only_balanced_accuracy": {
            "type": "upper_confidence_bound_below",
            "upper_boundary": 0.55,
        },
        "question_blind_balanced_accuracy": {
            "type": "upper_confidence_bound_below",
            "upper_boundary": 0.55,
        },
        "remux_accuracy_difference": {
            "type": "lower_confidence_bound_above",
            "lower_boundary": -0.02,
        },
        "remux_p_gold_difference": {
            "type": "lower_confidence_bound_above",
            "lower_boundary": -remux_p_gold_epsilon,
        },
        "orientation_difference": {
            "type": "confidence_interval_within",
            "lower_boundary": -0.05,
            "upper_boundary": 0.05,
        },
        "option_permutation_0_1_difference": {
            "type": "confidence_interval_within",
            "lower_boundary": -0.05,
            "upper_boundary": 0.05,
        },
        "option_permutation_0_2_difference": {
            "type": "confidence_interval_within",
            "lower_boundary": -0.05,
            "upper_boundary": 0.05,
        },
        "option_permutation_1_2_difference": {
            "type": "confidence_interval_within",
            "lower_boundary": -0.05,
            "upper_boundary": 0.05,
        },
        "shuffled_question_balanced_accuracy": {
            "type": "upper_confidence_bound_below",
            "upper_boundary": 0.55,
        },
        "aligned_minus_shuffled_difference": {
            "type": "lower_confidence_bound_above",
            "lower_boundary": 0.10,
        },
    }

    def alternative(rule: dict) -> float:
        if rule["type"] == "upper_confidence_bound_below":
            return rule["upper_boundary"] - effect_distance
        if rule["type"] == "lower_confidence_bound_above":
            return rule["lower_boundary"] + effect_distance
        return (rule["lower_boundary"] + rule["upper_boundary"]) / 2.0

    endpoint_designs = {}
    for endpoint in nuisance_gate_module._V2_POWER_ENDPOINTS:
        within_endpoint_count = 6 if endpoint == "metadata_balanced_accuracy" else 1
        endpoint_designs[endpoint] = {
            "sample_size_components": sample_size_components,
            "alternative_value": alternative(rules[endpoint]),
            "marginal_standard_deviation": marginal_standard_deviation,
            "within_component_correlation": within_component_correlation,
            "within_endpoint_statistic_count": within_endpoint_count,
            "within_endpoint_statistic_correlation": (
                0.5 if within_endpoint_count > 1 else 0.0
            ),
            "critical_value": statistics.NormalDist().inv_cdf(
                1.0 - 0.05 / (endpoint_count * within_endpoint_count)
            ),
            "decision_rule": rules[endpoint],
        }
    return {
        "schema": "conflictbench.perception-shortcut-power-dgp.v2",
        "analysis_unit": "endpoint_level_simulation_draw",
        "endpoint_designs": endpoint_designs,
        "decision_statistic": {
            "name": "endpoint_specific_gaussian_component_mean_bounds",
            "sampling_model": "paired_gaussian_known_variance",
            "familywise_alpha": 0.05,
            "multiplicity_correction": "bonferroni",
        },
        "source": "frozen_training_only_component_gaussian_design_v1",
    }


def _v2_power_simulator(
    tmp_path: pathlib.Path,
) -> nuisance_gate_module.SubprocessPowerSimulatorSpec:
    executable = tmp_path / "power-simulator"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import hashlib, json, math, sys\n"
        "request = json.loads(sys.stdin.read())\n"
        "bindings = request['bindings']\n"
        "designs = request['dgp']['endpoint_designs']\n"
        "denominator = float(1 << 64)\n"
        "draws = {}\n"
        "statistics_by_endpoint = {}\n"
        "for endpoint, design in designs.items():\n"
        "    standard_error = design['marginal_standard_deviation'] * "
        "math.sqrt(2.0 * "
        "(1.0 - design['within_component_correlation']) / "
        "design['sample_size_components'])\n"
        "    values = []\n"
        "    endpoint_statistics = []\n"
        "    for draw_index in range(bindings['repetitions']):\n"
        "        prefix = (f\"{bindings['algorithm']}\\0{bindings['dgp_sha256']}\\0\"\n"
        "                  f\"{bindings['seed']}\\0{endpoint}\\0{draw_index}\")\n"
        "        first = hashlib.sha256("
        'f"{prefix}\\0common_normal_1".encode()).digest()\n'
        "        second = hashlib.sha256("
        'f"{prefix}\\0common_normal_2".encode()).digest()\n'
        "        uniform_1 = (int.from_bytes(first[:8], 'big') + 0.5) / denominator\n"
        "        uniform_2 = (int.from_bytes(second[:8], 'big') + 0.5) / denominator\n"
        "        common = math.sqrt(-2.0 * math.log(uniform_1)) * "
        "math.cos(2.0 * math.pi * uniform_2)\n"
        "        estimates = []\n"
        "        for statistic_index in range("
        "design['within_endpoint_statistic_count']):\n"
        "            first = hashlib.sha256("
        'f"{prefix}\\0statistic_{statistic_index}_normal_1".encode()).digest()\n'
        "            second = hashlib.sha256("
        'f"{prefix}\\0statistic_{statistic_index}_normal_2".encode()).digest()\n'
        "            uniform_1 = "
        "(int.from_bytes(first[:8], 'big') + 0.5) / denominator\n"
        "            uniform_2 = "
        "(int.from_bytes(second[:8], 'big') + 0.5) / denominator\n"
        "            independent = math.sqrt(-2.0 * math.log(uniform_1)) * "
        "math.cos(2.0 * math.pi * uniform_2)\n"
        "            correlation = "
        "design['within_endpoint_statistic_correlation']\n"
        "            normal = math.sqrt(correlation) * common + "
        "math.sqrt(1.0 - correlation) * independent\n"
        "            estimates.append("
        "design['alternative_value'] + standard_error * normal)\n"
        "        critical = design['critical_value']\n"
        "        rule = design['decision_rule']\n"
        "        if rule['type'] == 'upper_confidence_bound_below':\n"
        "            estimate = max(estimates)\n"
        "            upper = estimate + critical * standard_error\n"
        "            detected = upper < rule['upper_boundary']\n"
        "        elif rule['type'] == 'lower_confidence_bound_above':\n"
        "            estimate = min(estimates)\n"
        "            lower = estimate - critical * standard_error\n"
        "            detected = lower > rule['lower_boundary']\n"
        "        else:\n"
        "            midpoint = "
        "(rule['lower_boundary'] + rule['upper_boundary']) / 2.0\n"
        "            estimate = max(estimates, key=lambda item: "
        "abs(item - midpoint))\n"
        "            lower = estimate - critical * standard_error\n"
        "            upper = estimate + critical * standard_error\n"
        "            detected = (lower >= rule['lower_boundary'] and "
        "upper <= rule['upper_boundary'])\n"
        "        endpoint_statistics.append(float(estimate))\n"
        "        values.append(int(detected))\n"
        "    draws[endpoint] = values\n"
        "    statistics_by_endpoint[endpoint] = endpoint_statistics\n"
        "response = {'schema': "
        "'conflictbench.perception-power-simulator-response.v2', "
        "'bindings': bindings, 'endpoint_statistics': statistics_by_endpoint, "
        "'endpoint_draws': draws}\n"
        "sys.stdout.write(json.dumps(response, sort_keys=True, "
        "separators=(',', ':')) + '\\n')\n",
        encoding="utf-8",
    )
    executable.chmod(0o500)
    configuration = tmp_path / "power-simulator-configuration.json"
    configuration.write_text(
        '{"algorithm":"sha256_counter_endpoint_gaussian_bounds_v1"}\n'
    )
    configuration.chmod(0o400)
    return nuisance_gate_module.SubprocessPowerSimulatorSpec(
        executable_path=executable,
        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        configuration_path=configuration,
        configuration_sha256=hashlib.sha256(configuration.read_bytes()).hexdigest(),
    )


def _v2_dependency_verifier(
    tmp_path: pathlib.Path,
) -> nuisance_gate_module.SubprocessDependencyVerifierSpec:
    executable = tmp_path / "dependency-verifier"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "request = json.loads(sys.stdin.read())\n"
        "projection = request['subject_projection']\n"
        "events = request['transcript']['events']\n"
        "result = {'decision_status': events[-1]['decision_status'], "
        "'reconstructed_subject_sha256': projection['subject_sha256']}\n"
        "response = {'schema': "
        "'conflictbench.perception-dependency-verifier-response.v2', "
        "'bindings': request['bindings'], 'result': result}\n"
        "sys.stdout.write(json.dumps(response, sort_keys=True, "
        "separators=(',', ':')) + '\\n')\n",
        encoding="utf-8",
    )
    executable.chmod(0o500)
    configuration = tmp_path / "dependency-verifier-configuration.json"
    configuration.write_text('{"mode":"strict-replay"}\n')
    configuration.chmod(0o400)
    return nuisance_gate_module.SubprocessDependencyVerifierSpec(
        executable_path=executable,
        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        configuration_path=configuration,
        configuration_sha256=hashlib.sha256(configuration.read_bytes()).hexdigest(),
    )


def _v2_dependency_projection(role: str, subject_sha256: str) -> dict:
    return {
        "schema": "conflictbench.perception-dependency-subject-projection.v2",
        "role": role,
        "subject_sha256": subject_sha256,
        "projection": {"status": "complete", "record_count": 30},
    }


def _v2_dependency_transcript(role: str, decision_status: str = "pass") -> dict:
    return {
        "schema": "conflictbench.perception-dependency-verification-transcript.v2",
        "role": role,
        "events": [
            {"step": "load"},
            {"step": "verify", "decision_status": decision_status},
        ],
    }


def test_v2_preregistration_contains_only_pre_run_inputs() -> None:
    config = _v2_config()
    config_sha256 = _canonical_digest(config)
    preregistration = _v2_preregistration(config_sha256)

    nuisance_gate_module.validate_shortcut_preregistration_v2(
        preregistration,
        expected_configuration_sha256=config_sha256,
        expected_gate_implementation_bundle_sha256="7" * 64,
    )
    serialized = json.dumps(preregistration, sort_keys=True)
    assert preregistration["status"] == "preregistered_not_run"
    assert "primary_output" not in serialized
    assert "replay_output" not in serialized
    assert "shortcut_input_sha256" not in serialized
    assert preregistration["inventory"]["record_count"] == 4
    assert preregistration["inventory"]["component_count"] == 2


def test_v2_power_record_requires_10000_pre_run_simulations(
    tmp_path: pathlib.Path,
) -> None:
    preregistration = _v2_preregistration(_canonical_digest(_v2_config()))
    simulator = _v2_power_simulator(tmp_path)
    record = nuisance_gate_module.build_shortcut_power_record_v2(
        inventory_sha256=preregistration["inventory"]["inventory_sha256"],
        component_count=preregistration["inventory"]["component_count"],
        remux_p_gold_epsilon=0.03,
        simulation_repetitions=10_000,
        dgp=_v2_power_dgp(),
        simulator=simulator,
    )
    record["simulation"]["repetitions"] = 9_999
    record["attestation_sha256"] = _canonical_digest(
        {key: item for key, item in record.items() if key != "attestation_sha256"}
    )

    with pytest.raises(NuisanceValidationError, match="exactly 10000"):
        nuisance_gate_module.validate_shortcut_power_record_v2(
            record,
            expected_inventory_sha256=preregistration["inventory"]["inventory_sha256"],
            expected_component_count=preregistration["inventory"]["component_count"],
            expected_remux_p_gold_epsilon=0.03,
            simulator=simulator,
        )


def test_v2_power_dgp_uses_an_explicit_component_design_and_decision_statistic() -> (
    None
):
    """Catch circular power evidence supplied as endpoint detection probabilities."""

    dgp = _v2_power_dgp(
        sample_size_components=15,
        effect_distance=0.08,
        marginal_standard_deviation=0.10,
        within_component_correlation=0.4,
    )
    normalized = nuisance_gate_module._validate_shortcut_power_dgp_v2(dgp)
    assert "endpoint_detection_probabilities" not in normalized
    metadata_design = normalized["endpoint_designs"]["metadata_balanced_accuracy"]
    assert metadata_design["alternative_value"] == pytest.approx(0.47)
    assert {
        key: item for key, item in metadata_design.items() if key != "alternative_value"
    } == {
        "sample_size_components": 15,
        "marginal_standard_deviation": 0.10,
        "within_component_correlation": 0.4,
        "within_endpoint_statistic_count": 6,
        "within_endpoint_statistic_correlation": 0.5,
        "critical_value": statistics.NormalDist().inv_cdf(1.0 - 0.05 / 66),
        "decision_rule": {
            "type": "upper_confidence_bound_below",
            "upper_boundary": 0.55,
        },
    }
    assert normalized["decision_statistic"]["name"] == (
        "endpoint_specific_gaussian_component_mean_bounds"
    )
    assert normalized["endpoint_designs"]["orientation_difference"][
        "decision_rule"
    ] == {
        "type": "confidence_interval_within",
        "lower_boundary": -0.05,
        "upper_boundary": 0.05,
    }
    assert (
        normalized["endpoint_designs"]["metadata_balanced_accuracy"]["critical_value"]
        > normalized["endpoint_designs"]["question_only_balanced_accuracy"][
            "critical_value"
        ]
    )

    circular = {
        "schema": "conflictbench.perception-shortcut-power-dgp.v2",
        "analysis_unit": "endpoint_level_simulation_draw",
        "endpoint_detection_probabilities": {
            endpoint: 0.9 for endpoint in nuisance_gate_module._V2_POWER_ENDPOINTS
        },
        "source": "caller_authored_probabilities",
    }
    with pytest.raises(NuisanceValidationError, match="DGP fields"):
        nuisance_gate_module._validate_shortcut_power_dgp_v2(circular)

    misaligned = copy.deepcopy(dgp)
    misaligned["endpoint_designs"]["orientation_difference"]["decision_rule"] = {
        "type": "upper_confidence_bound_below",
        "upper_boundary": 0.55,
    }
    with pytest.raises(NuisanceValidationError, match="decision rule"):
        nuisance_gate_module._validate_shortcut_power_dgp_v2(misaligned)

    uncorrected_metadata = copy.deepcopy(dgp)
    metadata_design = uncorrected_metadata["endpoint_designs"][
        "metadata_balanced_accuracy"
    ]
    metadata_design["within_endpoint_statistic_count"] = 1
    metadata_design["within_endpoint_statistic_correlation"] = 0.0
    metadata_design["critical_value"] = statistics.NormalDist().inv_cdf(
        1.0 - 0.05 / len(nuisance_gate_module._V2_POWER_ENDPOINTS)
    )
    with pytest.raises(NuisanceValidationError, match="within-endpoint count differs"):
        nuisance_gate_module._validate_shortcut_power_dgp_v2(uncorrected_metadata)

    weaker = _v2_power_dgp(
        sample_size_components=2,
        effect_distance=0.01,
        marginal_standard_deviation=0.20,
        within_component_correlation=0.4,
    )
    stronger = _v2_power_dgp(
        sample_size_components=40,
        effect_distance=0.08,
        marginal_standard_deviation=0.05,
        within_component_correlation=0.4,
    )
    weak_draws = nuisance_gate_module._expected_shortcut_power_draws_v2(
        weaker, seed=17, repetitions=1_000
    )
    strong_draws = nuisance_gate_module._expected_shortcut_power_draws_v2(
        stronger, seed=17, repetitions=1_000
    )
    assert sum(map(sum, strong_draws.values())) > sum(map(sum, weak_draws.values()))


def test_v2_power_dgp_sample_size_must_match_the_frozen_inventory(
    tmp_path: pathlib.Path,
) -> None:
    preregistration = _v2_preregistration(_canonical_digest(_v2_config()))
    simulator = _v2_power_simulator(tmp_path)

    with pytest.raises(NuisanceValidationError, match="sample size differs"):
        nuisance_gate_module.build_shortcut_power_record_v2(
            inventory_sha256=preregistration["inventory"]["inventory_sha256"],
            component_count=preregistration["inventory"]["component_count"],
            remux_p_gold_epsilon=0.03,
            simulation_repetitions=10_000,
            dgp=_v2_power_dgp(sample_size_components=3),
            simulator=simulator,
        )


def test_v2_power_dgp_remux_probability_margin_must_match_the_control_plan(
    tmp_path: pathlib.Path,
) -> None:
    preregistration = _v2_preregistration(_canonical_digest(_v2_config()))
    simulator = _v2_power_simulator(tmp_path)

    with pytest.raises(NuisanceValidationError, match="remux p-gold boundary differs"):
        nuisance_gate_module.build_shortcut_power_record_v2(
            inventory_sha256=preregistration["inventory"]["inventory_sha256"],
            component_count=preregistration["inventory"]["component_count"],
            remux_p_gold_epsilon=0.04,
            simulation_repetitions=10_000,
            dgp=_v2_power_dgp(remux_p_gold_epsilon=0.03),
            simulator=simulator,
        )


def test_v2_power_builder_rejects_caller_supplied_detection_counts() -> None:
    """Catch replacement of simulated endpoint draws with caller assertions."""

    preregistration = _v2_preregistration(_canonical_digest(_v2_config()))
    counts = {
        endpoint: 9_000
        for endpoint in (
            "metadata_balanced_accuracy",
            "question_only_balanced_accuracy",
            "question_blind_balanced_accuracy",
            "remux_accuracy_difference",
            "remux_p_gold_difference",
            "orientation_difference",
            "option_permutation_0_1_difference",
            "option_permutation_0_2_difference",
            "option_permutation_1_2_difference",
            "shuffled_question_balanced_accuracy",
            "aligned_minus_shuffled_difference",
        )
    }

    with pytest.raises(NuisanceValidationError, match="caller-supplied"):
        nuisance_gate_module.build_shortcut_power_record_v2(
            inventory_sha256=preregistration["inventory"]["inventory_sha256"],
            component_count=preregistration["inventory"]["component_count"],
            remux_p_gold_epsilon=0.03,
            simulation_repetitions=10_000,
            endpoint_detection_counts=counts,
        )


def test_v2_power_record_recomputes_each_endpoint_lower_bound(
    tmp_path: pathlib.Path,
) -> None:
    preregistration = _v2_preregistration(_canonical_digest(_v2_config()))
    simulator = _v2_power_simulator(tmp_path)
    record = nuisance_gate_module.build_shortcut_power_record_v2(
        inventory_sha256=preregistration["inventory"]["inventory_sha256"],
        component_count=preregistration["inventory"]["component_count"],
        remux_p_gold_epsilon=0.03,
        simulation_repetitions=10_000,
        dgp=_v2_power_dgp(),
        simulator=simulator,
    )
    record["simulation"]["endpoints"]["metadata_balanced_accuracy"][
        "power_lower_bound"
    ] = 0.99
    record["attestation_sha256"] = _canonical_digest(
        {key: item for key, item in record.items() if key != "attestation_sha256"}
    )

    with pytest.raises(NuisanceValidationError, match="power summary differs"):
        nuisance_gate_module.validate_shortcut_power_record_v2(
            record,
            expected_inventory_sha256=preregistration["inventory"]["inventory_sha256"],
            expected_component_count=preregistration["inventory"]["component_count"],
            expected_remux_p_gold_epsilon=0.03,
            simulator=simulator,
        )


def test_v2_power_record_rejects_a_re_attested_seed_change(
    tmp_path: pathlib.Path,
) -> None:
    """Catch a simulator transcript whose recorded seed did not generate its draws."""

    preregistration = _v2_preregistration(_canonical_digest(_v2_config()))
    simulator = _v2_power_simulator(tmp_path)
    record = nuisance_gate_module.build_shortcut_power_record_v2(
        inventory_sha256=preregistration["inventory"]["inventory_sha256"],
        component_count=preregistration["inventory"]["component_count"],
        remux_p_gold_epsilon=0.03,
        simulation_repetitions=10_000,
        dgp=_v2_power_dgp(),
        simulator=simulator,
    )
    record["simulation"]["seed"] += 1
    record["attestation_sha256"] = _canonical_digest(
        {key: item for key, item in record.items() if key != "attestation_sha256"}
    )

    with pytest.raises(NuisanceValidationError, match="draws differ"):
        nuisance_gate_module.validate_shortcut_power_record_v2(
            record,
            expected_inventory_sha256=preregistration["inventory"]["inventory_sha256"],
            expected_component_count=preregistration["inventory"]["component_count"],
            expected_remux_p_gold_epsilon=0.03,
            simulator=simulator,
        )


def test_v2_dependency_record_requires_verified_replay_and_pass_status(
    tmp_path: pathlib.Path,
) -> None:
    subject_sha256 = "1" * 64
    verifier = _v2_dependency_verifier(tmp_path)
    projection = _v2_dependency_projection("semantic_gate", subject_sha256)
    transcript = _v2_dependency_transcript("semantic_gate", "fail")
    record = nuisance_gate_module.build_verified_dependency_record_v2(
        role="semantic_gate",
        subject_sha256=subject_sha256,
        subject_projection=projection,
        transcript=transcript,
        verifier=verifier,
    )

    with pytest.raises(NuisanceValidationError, match="did not pass"):
        nuisance_gate_module.validate_verified_dependency_record_v2(
            record,
            expected_role="semantic_gate",
            expected_subject_sha256=subject_sha256,
            subject_projection=projection,
            transcript=transcript,
            verifier=verifier,
        )


def test_v2_dependency_validation_rejects_hash_only_self_attestation() -> None:
    """Catch accepting a receipt without authenticating and running its verifier."""

    subject_sha256 = "1" * 64
    with pytest.raises(NuisanceValidationError, match="authenticated"):
        nuisance_gate_module.build_verified_dependency_record_v2(
            role="semantic_gate",
            subject_sha256=subject_sha256,
            decision_status="pass",
            verifier_sha256="2" * 64,
            transcript_sha256="3" * 64,
        )


def test_v2_dependency_validation_replays_the_authenticated_transcript(
    tmp_path: pathlib.Path,
) -> None:
    """Catch substitution of a new transcript behind a still-attested receipt."""

    role = "semantic_gate"
    subject_sha256 = "1" * 64
    verifier = _v2_dependency_verifier(tmp_path)
    projection = _v2_dependency_projection(role, subject_sha256)
    transcript = _v2_dependency_transcript(role)
    record = nuisance_gate_module.build_verified_dependency_record_v2(
        role=role,
        subject_sha256=subject_sha256,
        subject_projection=projection,
        transcript=transcript,
        verifier=verifier,
    )
    substituted = _v2_dependency_transcript(role, "complete")

    with pytest.raises(NuisanceValidationError, match="transcript differs"):
        nuisance_gate_module.validate_verified_dependency_record_v2(
            record,
            expected_role=role,
            expected_subject_sha256=subject_sha256,
            subject_projection=projection,
            transcript=substituted,
            verifier=verifier,
        )


def test_v2_dependency_chain_requires_all_four_replay_verified_inputs(
    tmp_path: pathlib.Path,
) -> None:
    config_sha256 = _canonical_digest(_v2_config())
    preregistration = _v2_preregistration(config_sha256)
    role_keys = {
        "raw_conflict_index": (
            "raw_conflict_index_sha256",
            "raw_conflict_index_verification_sha256",
        ),
        "source_only": (
            "source_only_output_sha256",
            "source_only_replay_verification_sha256",
        ),
        "semantic_gate": (
            "semantic_gate_output_sha256",
            "semantic_gate_replay_verification_sha256",
        ),
        "derived_media_v4": (
            "derived_media_v4_index_sha256",
            "derived_media_v4_verification_sha256",
        ),
    }
    dependencies = {}
    dependency_sha256s = {}
    projections = {}
    transcripts = {}
    verifiers = {}
    for index, (role, (subject_key, verification_key)) in enumerate(role_keys.items()):
        role_root = tmp_path / role
        role_root.mkdir()
        subject_sha256 = preregistration["input_digests"][subject_key]
        projection = _v2_dependency_projection(role, subject_sha256)
        transcript = _v2_dependency_transcript(role)
        verifier = _v2_dependency_verifier(role_root)
        record = nuisance_gate_module.build_verified_dependency_record_v2(
            role=role,
            subject_sha256=subject_sha256,
            subject_projection=projection,
            transcript=transcript,
            verifier=verifier,
        )
        dependencies[role] = record
        projections[role] = projection
        transcripts[role] = transcript
        verifiers[role] = verifier
        digest = _canonical_digest(record)
        dependency_sha256s[role] = digest
        preregistration["input_digests"][verification_key] = digest

    simulator_root = tmp_path / "power"
    simulator_root.mkdir()
    simulator = _v2_power_simulator(simulator_root)
    power = nuisance_gate_module.build_shortcut_power_record_v2(
        inventory_sha256=preregistration["inventory"]["inventory_sha256"],
        component_count=preregistration["inventory"]["component_count"],
        remux_p_gold_epsilon=0.03,
        simulation_repetitions=10_000,
        dgp=_v2_power_dgp(),
        simulator=simulator,
    )
    power_sha256 = _canonical_digest(power)
    preregistration["input_digests"]["power_output_sha256"] = power_sha256
    preregistration["attestation_sha256"] = _canonical_digest(
        {
            key: item
            for key, item in preregistration.items()
            if key != "attestation_sha256"
        }
    )

    nuisance_gate_module.validate_v2_dependency_chain(
        preregistration,
        power_record=power,
        power_record_sha256=power_sha256,
        power_simulator=simulator,
        dependency_records=dependencies,
        dependency_record_sha256s=dependency_sha256s,
        dependency_subject_projections=projections,
        dependency_transcripts=transcripts,
        dependency_verifiers=verifiers,
    )

    changed_margin = copy.deepcopy(preregistration)
    changed_margin["control_plan"]["remux_noninferiority"]["epsilon_p_gold"] = 0.04
    changed_margin["control_plan_sha256"] = _canonical_digest(
        changed_margin["control_plan"]
    )
    changed_margin["attestation_sha256"] = _canonical_digest(
        {
            key: item
            for key, item in changed_margin.items()
            if key != "attestation_sha256"
        }
    )
    with pytest.raises(NuisanceValidationError, match="remux p-gold boundary differs"):
        nuisance_gate_module.validate_v2_dependency_chain(
            changed_margin,
            power_record=power,
            power_record_sha256=power_sha256,
            power_simulator=simulator,
            dependency_records=dependencies,
            dependency_record_sha256s=dependency_sha256s,
            dependency_subject_projections=projections,
            dependency_transcripts=transcripts,
            dependency_verifiers=verifiers,
        )

    original_record = dependencies["semantic_gate"]
    original_transcript = transcripts["semantic_gate"]
    substituted_transcript = _v2_dependency_transcript("semantic_gate", "complete")
    substituted_record = nuisance_gate_module.build_verified_dependency_record_v2(
        role="semantic_gate",
        subject_sha256=preregistration["input_digests"]["semantic_gate_output_sha256"],
        subject_projection=projections["semantic_gate"],
        transcript=substituted_transcript,
        verifier=verifiers["semantic_gate"],
    )
    dependencies["semantic_gate"] = substituted_record
    transcripts["semantic_gate"] = substituted_transcript
    with pytest.raises(NuisanceValidationError, match="content digest differs"):
        nuisance_gate_module.validate_v2_dependency_chain(
            preregistration,
            power_record=power,
            power_record_sha256=power_sha256,
            power_simulator=simulator,
            dependency_records=dependencies,
            dependency_record_sha256s=dependency_sha256s,
            dependency_subject_projections=projections,
            dependency_transcripts=transcripts,
            dependency_verifiers=verifiers,
        )
    dependencies["semantic_gate"] = original_record
    transcripts["semantic_gate"] = original_transcript

    substituted_power = nuisance_gate_module.build_shortcut_power_record_v2(
        inventory_sha256=preregistration["inventory"]["inventory_sha256"],
        component_count=preregistration["inventory"]["component_count"],
        remux_p_gold_epsilon=0.03,
        simulation_repetitions=10_000,
        dgp=_v2_power_dgp(effect_distance=0.06),
        simulator=simulator,
        seed=20_270_919,
    )
    with pytest.raises(NuisanceValidationError, match="content digest differs"):
        nuisance_gate_module.validate_v2_dependency_chain(
            preregistration,
            power_record=substituted_power,
            power_record_sha256=power_sha256,
            power_simulator=simulator,
            dependency_records=dependencies,
            dependency_record_sha256s=dependency_sha256s,
            dependency_subject_projections=projections,
            dependency_transcripts=transcripts,
            dependency_verifiers=verifiers,
        )

    dependencies.pop("semantic_gate")
    with pytest.raises(NuisanceValidationError, match="dependency role set"):
        nuisance_gate_module.validate_v2_dependency_chain(
            preregistration,
            power_record=power,
            power_record_sha256=power_sha256,
            power_simulator=simulator,
            dependency_records=dependencies,
            dependency_record_sha256s=dependency_sha256s,
            dependency_subject_projections=projections,
            dependency_transcripts=transcripts,
            dependency_verifiers=verifiers,
        )


def test_v2_derived_media_inventory_is_authenticated_before_and_after_replay(
    tmp_path: pathlib.Path,
) -> None:
    preregistration = _v2_preregistration(_canonical_digest(_v2_config()))
    media_root = tmp_path / "derived-media"
    media_root.mkdir()
    for media in preregistration["inventory"]["media_files"]:
        path = media_root / media["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        modality = "audio" if path.suffix == ".wav" else "video"
        path.write_bytes(f"{modality}:{path.stem}".encode())
        path.chmod(0o400)

    before = nuisance_gate_module.authenticate_v2_media_inventory(
        media_root, preregistration["inventory"]["media_files"]
    )
    assert (
        before["media_inventory_sha256"]
        == preregistration["inventory"]["media_inventory_sha256"]
    )

    changed = (
        media_root / preregistration["inventory"]["media_files"][2]["relative_path"]
    )
    changed.chmod(0o600)
    changed.write_bytes(b"changed")
    changed.chmod(0o400)
    with pytest.raises(NuisanceValidationError, match="size|digest"):
        nuisance_gate_module.authenticate_v2_media_inventory(
            media_root, preregistration["inventory"]["media_files"]
        )


def test_v2_shortcut_output_is_exact_replay_bound_and_question_blind(
    tmp_path: pathlib.Path,
) -> None:
    config = _v2_config()
    config_sha256 = _canonical_digest(config)
    (
        value,
        replay_worker,
        preregistration,
        preregistration_sha256,
        media_root,
    ) = _prepared_v2(tmp_path, config_sha256)

    validated = validate_shortcut_output_v2(
        value,
        expected_configuration_sha256=config_sha256,
        expected_media_set_sha256=preregistration["inventory"][
            "media_inventory_sha256"
        ],
        expected_gate_implementation_bundle_sha256="7" * 64,
        expected_shortcut_preregistration_sha256=preregistration_sha256,
    )
    request = validated["question_blind_audiovisual_mismatch"]["records"][0]["request"]
    assert set(request) == {"schema", "audio", "video"}
    assert not {
        "question",
        "options",
        "answer",
        "condition",
        "orientation",
        "option_permutation",
    }.intersection(request)

    report = evaluate_shortcut_controls_v2(
        value,
        preregistration,
        _v2_metadata_records(),
        config,
        expected_configuration_sha256=config_sha256,
        expected_gate_implementation_bundle_sha256="7" * 64,
        expected_shortcut_preregistration_sha256=preregistration_sha256,
        expected_shortcut_output_sha256=_json_file_digest(value),
        replay_worker=replay_worker,
        derived_media_root=media_root,
    )

    assert report["schema"] == "conflictbench.perception-shortcut-report.v2"
    assert report["question_blind_audiovisual_mismatch"]["balanced_accuracy"] == 0.5
    assert report["question_only_conflict"]["balanced_accuracy"] == 0.5
    assert report["question_only_conflict"]["fresh_replays_per_record"] == 2
    assert (
        report["derived_media_authentication"]["before"]
        == report["derived_media_authentication"]["after"]
    )
    assert report["question_blind_audiovisual_mismatch"]["method"] == (
        "analysis_component_cluster_percentile"
    )
    assert (
        "equal_component_wilson_interval"
        not in report["question_blind_audiovisual_mismatch"]
    )
    assert report["decision"] == {"status": "pass", "reason_codes": []}


def test_v2_shortcut_gate_refuses_self_reported_question_blind_scores(
    tmp_path: pathlib.Path,
) -> None:
    config = _v2_config()
    config_sha256 = _canonical_digest(config)
    value, _, preregistration, preregistration_sha256, media_root = _prepared_v2(
        tmp_path, config_sha256
    )

    with pytest.raises(NuisanceValidationError, match="subprocess replay worker"):
        evaluate_shortcut_controls_v2(
            value,
            preregistration,
            _v2_metadata_records(),
            config,
            expected_configuration_sha256=config_sha256,
            expected_gate_implementation_bundle_sha256="7" * 64,
            expected_shortcut_preregistration_sha256=preregistration_sha256,
            expected_shortcut_output_sha256=_json_file_digest(value),
            replay_worker=None,
            derived_media_root=media_root,
        )


def test_v2_shortcut_gate_rejects_a_nonproduction_bootstrap_count() -> None:
    config = _v2_config()
    config["shortcut_protocol"]["bootstrap_repetitions"] = 9_999
    config_sha256 = _canonical_digest(config)
    value = _shortcut_output_v2(config_sha256)

    with pytest.raises(NuisanceValidationError, match="at least 10000"):
        evaluate_shortcut_controls_v2(
            value,
            {},
            _v2_metadata_records(),
            config,
            expected_configuration_sha256=config_sha256,
            expected_gate_implementation_bundle_sha256="7" * 64,
            expected_shortcut_preregistration_sha256="e" * 64,
            expected_shortcut_output_sha256="f" * 64,
            replay_worker=None,
            derived_media_root=None,
        )


@pytest.mark.parametrize(
    ("section", "field"),
    [("uncertainty", "bootstrap_repetitions"), ("permutation", "repetitions")],
)
def test_v2_rejects_nonproduction_resampling_counts(section: str, field: str) -> None:
    config = _v2_config()
    config[section][field] = 9_999
    config_sha256 = _canonical_digest(config)

    with pytest.raises(NuisanceValidationError, match="at least 10000"):
        evaluate_shortcut_controls_v2(
            _shortcut_output_v2(config_sha256),
            {},
            _v2_metadata_records(component_count=2),
            config,
            expected_configuration_sha256=config_sha256,
            expected_gate_implementation_bundle_sha256="7" * 64,
            expected_shortcut_preregistration_sha256="e" * 64,
            expected_shortcut_output_sha256="f" * 64,
            replay_worker=None,
            derived_media_root=None,
        )


def test_v2_metadata_component_summary_uses_balanced_accuracy() -> None:
    outcomes = [
        {
            "component_id": "component-a",
            "role": "same_answer_nuisance",
            "predicted_role": "same_answer_nuisance",
        },
        {
            "component_id": "component-a",
            "role": "same_answer_nuisance",
            "predicted_role": "same_answer_nuisance",
        },
        {
            "component_id": "component-a",
            "role": "same_answer_nuisance",
            "predicted_role": "same_answer_nuisance",
        },
        {
            "component_id": "component-a",
            "role": "opposite_answer_candidate",
            "predicted_role": "same_answer_nuisance",
        },
    ]

    assert nuisance_gate_module._classification_component_balanced_accuracies(
        outcomes
    ) == [("component-a", 0.5)]


def test_v2_shortcut_gate_does_not_treat_two_constant_components_as_equivalence(
    tmp_path: pathlib.Path,
) -> None:
    config = _v2_config()
    config_sha256 = _canonical_digest(config)
    (
        value,
        replay_worker,
        preregistration,
        preregistration_sha256,
        media_root,
    ) = _prepared_v2(tmp_path, config_sha256, component_count=2, component_offset=4)
    records = _records(
        partition_sizes={
            "scorer_fit": 2,
            "threshold_calibration": 2,
            "pilot_gate": 2,
        }
    )
    for record in records:
        record["features"] = [0.0] * len(FEATURE_NAMES)

    with pytest.raises(NuisanceValidationError, match="at least 15 components"):
        evaluate_shortcut_controls_v2(
            value,
            preregistration,
            records,
            config,
            expected_configuration_sha256=config_sha256,
            expected_gate_implementation_bundle_sha256="7" * 64,
            expected_shortcut_preregistration_sha256=preregistration_sha256,
            expected_shortcut_output_sha256=_json_file_digest(value),
            replay_worker=replay_worker,
            derived_media_root=media_root,
        )


def test_v2_shortcut_gate_rejects_an_unbound_replay_verifier(
    tmp_path: pathlib.Path,
) -> None:
    config = _v2_config()
    config_sha256 = _canonical_digest(config)
    value, valid, preregistration, preregistration_sha256, media_root = _prepared_v2(
        tmp_path, config_sha256
    )
    changed = SubprocessReplayWorkerSpec(
        executable_path=valid.executable_path,
        executable_sha256=valid.executable_sha256,
        configuration_path=valid.configuration_path,
        configuration_sha256=valid.configuration_sha256,
        model_path=valid.model_path,
        model_sha256="f" * 64,
    )

    with pytest.raises(NuisanceValidationError, match="verifier binding differs"):
        evaluate_shortcut_controls_v2(
            value,
            preregistration,
            _v2_metadata_records(),
            config,
            expected_configuration_sha256=config_sha256,
            expected_gate_implementation_bundle_sha256="7" * 64,
            expected_shortcut_preregistration_sha256=preregistration_sha256,
            expected_shortcut_output_sha256=_json_file_digest(value),
            replay_worker=changed,
            derived_media_root=media_root,
        )


def test_v2_replay_worker_is_a_hash_pinned_process_not_a_callable(
    tmp_path: pathlib.Path,
) -> None:
    worker_path = tmp_path / "replay-worker"
    response_output = _mismatch_output(0.5)
    worker_path.write_text(
        "#!/usr/bin/env python3\n"
        "import hashlib, json, sys\n"
        "request = json.loads(sys.stdin.read())\n"
        "bindings = request['bindings']\n"
        f"output = {response_output!r}\n"
        "response = {"
        "'schema': 'conflictbench.perception-replay-worker-response.v2', "
        "'bindings': bindings, 'output': output}\n"
        "sys.stdout.write(json.dumps(response, sort_keys=True, "
        "separators=(',', ':')) + '\\n')\n",
        encoding="utf-8",
    )
    worker_path.chmod(0o500)
    configuration_path = tmp_path / "configuration.json"
    configuration_path.write_text("{}\n", encoding="utf-8")
    configuration_path.chmod(0o400)
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"model")
    model_path.chmod(0o400)
    worker_sha256 = hashlib.sha256(worker_path.read_bytes()).hexdigest()
    spec = nuisance_gate_module.SubprocessReplayWorkerSpec(
        executable_path=worker_path,
        executable_sha256=worker_sha256,
        configuration_path=configuration_path,
        configuration_sha256=hashlib.sha256(
            configuration_path.read_bytes()
        ).hexdigest(),
        model_path=model_path,
        model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
    )
    assert "replay" not in vars(spec)

    output = nuisance_gate_module.run_replay_worker(
        spec,
        {
            "schema": "conflictbench.perception-replay-request.v2",
            "task": "question_blind_audiovisual_mismatch",
            "inputs": {
                "audio_path": "audio.bin",
                "video_path": "video.bin",
            },
        },
    )
    assert output == response_output


def test_v2_replay_worker_rejects_an_executable_hash_mismatch(
    tmp_path: pathlib.Path,
) -> None:
    worker_path = tmp_path / "replay-worker"
    worker_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    worker_path.chmod(0o500)
    configuration_path = tmp_path / "configuration.json"
    configuration_path.write_text("{}\n", encoding="utf-8")
    configuration_path.chmod(0o400)
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"model")
    model_path.chmod(0o400)
    spec = nuisance_gate_module.SubprocessReplayWorkerSpec(
        executable_path=worker_path,
        executable_sha256="f" * 64,
        configuration_path=configuration_path,
        configuration_sha256=hashlib.sha256(
            configuration_path.read_bytes()
        ).hexdigest(),
        model_path=model_path,
        model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(NuisanceValidationError, match="executable digest differs"):
        nuisance_gate_module.run_replay_worker(
            spec,
            {
                "schema": "conflictbench.perception-replay-request.v2",
                "task": "question_blind_audiovisual_mismatch",
                "inputs": {
                    "audio_path": "audio.bin",
                    "video_path": "video.bin",
                },
            },
        )


def test_v2_shortcut_output_rejects_question_access_and_replay_drift() -> None:
    config = _v2_config()
    config_sha256 = _canonical_digest(config)
    value = _shortcut_output_v2(config_sha256)
    record = value["question_blind_audiovisual_mismatch"]["records"][0]
    record["request"]["question"] = "Which event occurs?"
    value["attestation_sha256"] = _canonical_digest(
        {key: item for key, item in value.items() if key != "attestation_sha256"}
    )
    with pytest.raises(NuisanceValidationError, match="request fields are unexpected"):
        validate_shortcut_output_v2(
            value,
            expected_configuration_sha256=config_sha256,
            expected_media_set_sha256="2" * 64,
            expected_gate_implementation_bundle_sha256="7" * 64,
            expected_shortcut_preregistration_sha256="e" * 64,
        )


def test_v2_shortcut_output_requires_the_exact_shared_record_inventory() -> None:
    config = _v2_config()
    config_sha256 = _canonical_digest(config)
    value = _shortcut_output_v2(config_sha256)
    value["remux_noninferiority"]["records"][0]["record_id"] = "different-record"
    value["attestation_sha256"] = _canonical_digest(
        {key: item for key, item in value.items() if key != "attestation_sha256"}
    )

    with pytest.raises(NuisanceValidationError, match="record inventory differs"):
        validate_shortcut_output_v2(
            value,
            expected_configuration_sha256=config_sha256,
            expected_media_set_sha256="2" * 64,
            expected_gate_implementation_bundle_sha256="7" * 64,
            expected_shortcut_preregistration_sha256="e" * 64,
        )


def test_v2_rejects_nonreplayed_control_outputs() -> None:
    config = _v2_config()
    config_sha256 = _canonical_digest(config)
    value = _shortcut_output_v2(config_sha256)
    del value["remux_noninferiority"]["records"][0]["clean_request"]
    value["attestation_sha256"] = _canonical_digest(
        {key: item for key, item in value.items() if key != "attestation_sha256"}
    )

    with pytest.raises(NuisanceValidationError, match="remux record 0 fields"):
        validate_shortcut_output_v2(
            value,
            expected_configuration_sha256=config_sha256,
            expected_media_set_sha256="2" * 64,
            expected_gate_implementation_bundle_sha256="7" * 64,
            expected_shortcut_preregistration_sha256="e" * 64,
        )


def test_v2_metadata_records_must_match_the_frozen_record_inventory(
    tmp_path: pathlib.Path,
) -> None:
    config = _v2_config()
    config_sha256 = _canonical_digest(config)
    (
        value,
        replay_worker,
        preregistration,
        preregistration_sha256,
        media_root,
    ) = _prepared_v2(tmp_path, config_sha256)
    metadata_records = _v2_metadata_records(component_count=2)
    evaluation_record = next(
        record for record in metadata_records if record["partition"] == "pilot_gate"
    )
    evaluation_record["pair_id"] = "substituted-record"

    with pytest.raises(NuisanceValidationError, match="metadata record inventory"):
        evaluate_shortcut_controls_v2(
            value,
            preregistration,
            metadata_records,
            config,
            expected_configuration_sha256=config_sha256,
            expected_gate_implementation_bundle_sha256="7" * 64,
            expected_shortcut_preregistration_sha256=preregistration_sha256,
            expected_shortcut_output_sha256=_json_file_digest(value),
            replay_worker=replay_worker,
            derived_media_root=media_root,
        )

    value = _shortcut_output_v2(config_sha256)
    value["question_blind_audiovisual_mismatch"]["records"][0]["replay_output"] = (
        _mismatch_output(0.6)
    )
    value["attestation_sha256"] = _canonical_digest(
        {key: item for key, item in value.items() if key != "attestation_sha256"}
    )
    with pytest.raises(NuisanceValidationError, match="recorded replay differs"):
        validate_shortcut_output_v2(
            value,
            expected_configuration_sha256=config_sha256,
            expected_media_set_sha256="2" * 64,
            expected_gate_implementation_bundle_sha256="7" * 64,
            expected_shortcut_preregistration_sha256="e" * 64,
        )


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    [
        ("remux", "remux_accuracy_noninferiority_failed"),
        ("orientation", "orientation_equivalence_failed"),
        ("option_permutation", "option_permutation_equivalence_failed"),
        ("shuffled", "shuffled_question_ucb_not_below_0_55"),
    ],
)
def test_v2_shortcut_gate_enforces_every_preregistered_margin(
    mutation: str, reason_code: str, tmp_path: pathlib.Path
) -> None:
    config = _v2_config()
    config_sha256 = _canonical_digest(config)
    (
        value,
        replay_worker,
        preregistration,
        preregistration_sha256,
        media_root,
    ) = _prepared_v2(tmp_path, config_sha256)
    if mutation == "remux":
        for record in value["remux_noninferiority"]["records"]:
            record["remux_request"]["question"] = (
                f"Question {record['record_id']}; answer=C"
            )
            record["remux_request_sha256"] = _canonical_digest(record["remux_request"])
            record["remux_output"] = _choice_output({"A": 0.1, "B": 0.1, "C": 0.8})
    elif mutation == "orientation":
        for record in value["orientation_symmetry"]["records"]:
            wrong = "CONFLICT" if record["gold_relation"] == "AGREE" else "AGREE"
            record["video_over_audio_request"]["question"] = (
                f"Question {record['record_id']}; relation={wrong}"
            )
            record["video_over_audio_request_sha256"] = _canonical_digest(
                record["video_over_audio_request"]
            )
            record["video_over_audio_output"] = _relation_output(wrong)
    elif mutation == "option_permutation":
        for record in value["option_permutation_equivalence"]["records"]:
            wrong = "CONFLICT" if record["gold_relation"] == "AGREE" else "AGREE"
            record["requests"]["2"]["question"] = (
                f"Question {record['record_id']}; relation={wrong}"
            )
            record["request_sha256s"]["2"] = _canonical_digest(record["requests"]["2"])
            record["outputs"]["2"] = _relation_output(wrong)
    elif mutation == "shuffled":
        for record in value["shuffled_question"]["records"]:
            record["shuffled_request"]["question"] = (
                f"Shuffled question {record['record_id']}; "
                f"relation={record['gold_relation']}"
            )
            record["shuffled_request_sha256"] = _canonical_digest(
                record["shuffled_request"]
            )
            record["shuffled_output"] = copy.deepcopy(record["aligned_output"])
    value["attestation_sha256"] = _canonical_digest(
        {key: item for key, item in value.items() if key != "attestation_sha256"}
    )
    preregistration, preregistration_sha256 = _bind_output_to_preregistration_v2(
        config_sha256, value
    )

    report = evaluate_shortcut_controls_v2(
        value,
        preregistration,
        _v2_metadata_records(),
        config,
        expected_configuration_sha256=config_sha256,
        expected_gate_implementation_bundle_sha256="7" * 64,
        expected_shortcut_preregistration_sha256=preregistration_sha256,
        expected_shortcut_output_sha256=_json_file_digest(value),
        replay_worker=replay_worker,
        derived_media_root=media_root,
    )

    assert report["decision"]["status"] == "fail"
    assert reason_code in report["decision"]["reason_codes"]


@pytest.mark.parametrize(
    ("summary_name", "reason_code"),
    [
        ("question_only_summary", "question_only_ucb_not_below_0_55"),
        ("question_blind_summary", "question_blind_ucb_not_below_0_55"),
    ],
)
def test_v2_shortcut_gate_applies_question_ablation_margins(
    summary_name: str, reason_code: str
) -> None:
    summaries = {
        "metadata_summary": {"one_sided_upper_confidence_bound": 0.50},
        "question_only_summary": {"one_sided_upper_confidence_bound": 0.50},
        "question_blind_summary": {"one_sided_upper_confidence_bound": 0.50},
        "remux_summary": {
            "accuracy_difference": {"one_sided_lower_confidence_bound": 0.0},
            "p_gold_difference": {"one_sided_lower_confidence_bound": 0.0},
            "epsilon_p_gold": 0.03,
        },
        "orientation_summary": {
            "audio_over_video_minus_video_over_audio": {
                "one_sided_lower_confidence_bound": 0.0,
                "one_sided_upper_confidence_bound": 0.0,
            }
        },
        "permutation_summary": {
            name: {
                "one_sided_lower_confidence_bound": 0.0,
                "one_sided_upper_confidence_bound": 0.0,
            }
            for name in ("0_minus_1", "0_minus_2", "1_minus_2")
        },
        "shuffled_summary": {
            "shuffled": {"one_sided_upper_confidence_bound": 0.50},
            "aligned_minus_shuffled": {"one_sided_lower_confidence_bound": 0.20},
        },
    }
    summaries[summary_name]["one_sided_upper_confidence_bound"] = 0.55

    reasons = nuisance_gate_module._shortcut_reason_codes_v2(
        **summaries,
        gate=_v2_config()["shortcut_gate"],
    )

    assert reason_code in reasons


def test_v2_contract_binds_the_preregistration_without_any_completed_output() -> None:
    contract = build_nuisance_contract(
        configuration_sha256="d" * 64,
        pilot_index_sha256="a" * 64,
        media_receipt_sha256="e" * 64,
        media_set_sha256="b" * 64,
        implementation_bundle_sha256="f" * 64,
        shortcut_preregistration_sha256="c" * 64,
    )
    assert contract["schema"] == (
        "conflictbench.perception-nuisance-detection-contract.v2"
    )
    assert contract["input_digests"]["shortcut_preregistration_sha256"] == "c" * 64
    assert "shortcut_input_sha256" not in contract["input_digests"]
    validate_nuisance_contract(
        contract,
        expected_pilot_index_sha256="a" * 64,
        expected_media_receipt_sha256="e" * 64,
        expected_media_set_sha256="b" * 64,
        expected_configuration_sha256="d" * 64,
        expected_implementation_bundle_sha256="f" * 64,
        expected_shortcut_preregistration_sha256="c" * 64,
    )


def test_public_v2_runner_stops_before_media_without_the_complete_replay_chain(
    tmp_path: pathlib.Path,
) -> None:
    config = _v2_config()
    config_path = tmp_path / "config.json"
    config_bytes = (json.dumps(config, indent=2, sort_keys=True) + "\n").encode()
    config_path.write_bytes(config_bytes)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    shortcut = _shortcut_output_v2(config_sha256)
    shortcut_path = tmp_path / "shortcut.json"
    shortcut_bytes = (json.dumps(shortcut, indent=2, sort_keys=True) + "\n").encode()
    shortcut_path.write_bytes(shortcut_bytes)
    shortcut_sha256 = hashlib.sha256(shortcut_bytes).hexdigest()
    missing = tmp_path / "not-opened"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            "--config",
            str(config_path),
            "--config-sha256",
            config_sha256,
            "--pilot-index",
            str(missing),
            "--pilot-index-sha256",
            "1" * 64,
            "--media-receipt",
            str(missing),
            "--media-receipt-sha256",
            "1" * 64,
            "--media-root",
            str(missing),
            "--media-set-sha256",
            "2" * 64,
            "--implementation-bundle-sha256",
            "7" * 64,
            "--nuisance-contract",
            str(missing),
            "--nuisance-contract-sha256",
            "1" * 64,
            "--shortcut-output",
            str(shortcut_path),
            "--shortcut-output-sha256",
            shortcut_sha256,
            "--ffprobe",
            str(missing),
            "--report",
            str(tmp_path / "report.json"),
            "--result",
            str(tmp_path / "result.json"),
        ],
        env={
            **os.environ,
            "PYTHONPATH": str(pathlib.Path(__file__).parents[1] / "src"),
        },
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "configuration v2 is missing required inputs" in completed.stderr
    assert "replay worker" in completed.stderr
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "result.json").exists()


def test_v2_production_execution_fails_closed_until_external_replayers_are_frozen(
    tmp_path: pathlib.Path,
) -> None:
    config = _v2_config()
    config_path = tmp_path / "config.json"
    config_bytes = (json.dumps(config, indent=2, sort_keys=True) + "\n").encode()
    config_path.write_bytes(config_bytes)
    missing = tmp_path / "not-opened"
    roles = {
        "raw_conflict_index",
        "source_only",
        "semantic_gate",
        "derived_media_v4",
    }

    with pytest.raises(
        NuisanceValidationError,
        match="frozen dependency replayers and power simulator",
    ):
        nuisance_gate_module.run_nuisance_gate(
            config_path=config_path,
            expected_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            pilot_index_path=missing,
            expected_pilot_index_sha256="1" * 64,
            media_receipt_path=missing,
            expected_media_receipt_sha256="1" * 64,
            media_root=missing,
            expected_media_set_sha256="2" * 64,
            expected_implementation_bundle_sha256="7" * 64,
            nuisance_contract_path=missing,
            expected_nuisance_contract_sha256="1" * 64,
            ffprobe_path=missing,
            report_path=tmp_path / "report.json",
            result_path=tmp_path / "result.json",
            source_paths={
                role: missing
                for role in nuisance_gate_module.IMPLEMENTATION_SOURCE_ROLES
            },
            derived_media_root=missing,
            shortcut_preregistration_path=missing,
            expected_shortcut_preregistration_sha256="1" * 64,
            shortcut_output_path=missing,
            expected_shortcut_output_sha256="1" * 64,
            power_record_path=missing,
            expected_power_record_sha256="1" * 64,
            dependency_subject_paths={role: missing for role in roles},
            dependency_subject_sha256s={role: "1" * 64 for role in roles},
            dependency_record_paths={role: missing for role in roles},
            dependency_record_sha256s={role: "1" * 64 for role in roles},
            replay_worker_path=missing,
            expected_replay_worker_sha256="1" * 64,
            replay_worker_configuration_path=missing,
            expected_replay_worker_configuration_sha256="1" * 64,
            replay_worker_model_path=missing,
            expected_replay_worker_model_sha256="1" * 64,
        )


def test_v2_completed_report_and_result_are_validated_and_written_once(
    tmp_path: pathlib.Path,
) -> None:
    config = _v2_config()
    config_sha256 = _canonical_digest(config)
    (
        output,
        replay_worker,
        preregistration,
        preregistration_sha256,
        media_root,
    ) = _prepared_v2(tmp_path, config_sha256)
    report = evaluate_shortcut_controls_v2(
        output,
        preregistration,
        _v2_metadata_records(),
        config,
        expected_configuration_sha256=config_sha256,
        expected_gate_implementation_bundle_sha256="7" * 64,
        expected_shortcut_preregistration_sha256=preregistration_sha256,
        expected_shortcut_output_sha256=_json_file_digest(output),
        replay_worker=replay_worker,
        derived_media_root=media_root,
    )
    report_sha256 = hashlib.sha256(
        (
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode()
    ).hexdigest()
    result = nuisance_gate_module.build_shortcut_result_v2(
        report=report,
        report_sha256=report_sha256,
    )
    report_path = tmp_path / "shortcut-report.json"
    result_path = tmp_path / "shortcut-result.json"
    nuisance_gate_module.write_shortcut_gate_outputs_v2(
        report_path,
        result_path,
        report=report,
        result=result,
    )

    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert json.loads(result_path.read_text(encoding="utf-8")) == result
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o400
    assert stat.S_IMODE(result_path.stat().st_mode) == 0o400
    with pytest.raises(NuisanceValidationError, match="output already exists"):
        nuisance_gate_module.write_shortcut_gate_outputs_v2(
            report_path,
            result_path,
            report=report,
            result=result,
        )


def test_v2_report_rejects_re_attested_nested_statistics(
    tmp_path: pathlib.Path,
) -> None:
    """Catch validation that trusts self-reported summaries after re-attestation."""

    config = _v2_config()
    config_sha256 = _canonical_digest(config)
    (
        output,
        replay_worker,
        preregistration,
        preregistration_sha256,
        media_root,
    ) = _prepared_v2(tmp_path, config_sha256)
    report = evaluate_shortcut_controls_v2(
        output,
        preregistration,
        _v2_metadata_records(),
        config,
        expected_configuration_sha256=config_sha256,
        expected_gate_implementation_bundle_sha256="7" * 64,
        expected_shortcut_preregistration_sha256=preregistration_sha256,
        expected_shortcut_output_sha256=_json_file_digest(output),
        replay_worker=replay_worker,
        derived_media_root=media_root,
    )
    first_family = config["model"]["families"][0]
    mutations = (
        (
            "metadata",
            lambda value: value["metadata_only"]["models"][first_family][
                "pair_ranking"
            ].__setitem__("point_estimate", 0.123456),
        ),
        (
            "question_only",
            lambda value: value["question_only_conflict"].__setitem__(
                "balanced_accuracy", 0.25
            ),
        ),
        (
            "question_blind",
            lambda value: value["question_blind_audiovisual_mismatch"].__setitem__(
                "balanced_accuracy", 0.25
            ),
        ),
        (
            "remux",
            lambda value: value["remux_noninferiority"][
                "accuracy_difference"
            ].__setitem__("point_estimate", 0.25),
        ),
        (
            "orientation",
            lambda value: value["orientation_symmetry"][
                "audio_over_video_minus_video_over_audio"
            ].__setitem__("point_estimate", 0.25),
        ),
        (
            "permutation",
            lambda value: value["option_permutation_equivalence"][
                "0_minus_1"
            ].__setitem__("point_estimate", 0.25),
        ),
        (
            "shuffled",
            lambda value: value["shuffled_question"]["aligned"].__setitem__(
                "point_estimate", 0.25
            ),
        ),
    )
    for label, mutate in mutations:
        changed = copy.deepcopy(report)
        mutate(changed)
        changed["attestation_sha256"] = _canonical_digest(
            {key: item for key, item in changed.items() if key != "attestation_sha256"}
        )
        with pytest.raises(
            NuisanceValidationError,
            match="authenticated records",
        ) as error:
            nuisance_gate_module.validate_shortcut_report_v2(changed)
        assert label in str(error.value) or "summary differs" in str(error.value)

    changed = copy.deepcopy(report)
    authenticated = changed["authenticated_records"]
    authenticated["question_only_records"][0]["verified_prediction"] = "CONFLICT"
    authenticated["records_sha256"] = _canonical_digest(
        {
            key: item
            for key, item in authenticated.items()
            if key not in {"schema", "records_sha256"}
        }
    )
    changed["attestation_sha256"] = _canonical_digest(
        {key: item for key, item in changed.items() if key != "attestation_sha256"}
    )
    with pytest.raises(NuisanceValidationError, match="authenticated records differ"):
        nuisance_gate_module.validate_shortcut_report_v2(changed)


def test_v2_report_rejects_re_attested_metadata_feature_substitution(
    tmp_path: pathlib.Path,
) -> None:
    """Catch recomputing every derived value around substituted metadata features."""

    config = _v2_config()
    config_sha256 = _canonical_digest(config)
    (
        output,
        replay_worker,
        preregistration,
        preregistration_sha256,
        media_root,
    ) = _prepared_v2(tmp_path, config_sha256)
    report = evaluate_shortcut_controls_v2(
        output,
        preregistration,
        _v2_metadata_records(),
        config,
        expected_configuration_sha256=config_sha256,
        expected_gate_implementation_bundle_sha256="7" * 64,
        expected_shortcut_preregistration_sha256=preregistration_sha256,
        expected_shortcut_output_sha256=_json_file_digest(output),
        replay_worker=replay_worker,
        derived_media_root=media_root,
    )

    changed = copy.deepcopy(report)
    authenticated = changed["authenticated_records"]
    authenticated["metadata_records"][0]["features"][0] = 1_000.0
    authenticated["records_sha256"] = _canonical_digest(
        {
            key: item
            for key, item in authenticated.items()
            if key not in {"schema", "records_sha256"}
        }
    )
    recomputed = nuisance_gate_module._summarize_authenticated_shortcut_records_v2(
        authenticated,
        config=config,
        worker=changed["replay_worker"],
        endpoint_confidence=changed["inference"]["per_endpoint_confidence_level"],
    )
    changed.update(recomputed)
    reasons = nuisance_gate_module._shortcut_reason_codes_v2(
        metadata_summary=recomputed["metadata_only"],
        question_only_summary=recomputed["question_only_conflict"],
        question_blind_summary=recomputed["question_blind_audiovisual_mismatch"],
        remux_summary=recomputed["remux_noninferiority"],
        orientation_summary=recomputed["orientation_symmetry"],
        permutation_summary=recomputed["option_permutation_equivalence"],
        shuffled_summary=recomputed["shuffled_question"],
        gate=config["shortcut_gate"],
    )
    changed["decision"] = {
        "status": "pass" if not reasons else "fail",
        "reason_codes": reasons,
    }
    changed["attestation_sha256"] = _canonical_digest(
        {key: item for key, item in changed.items() if key != "attestation_sha256"}
    )

    with pytest.raises(
        NuisanceValidationError, match="metadata-record binding differs"
    ):
        nuisance_gate_module.validate_shortcut_report_v2(changed)
