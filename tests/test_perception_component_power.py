from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import platform
import subprocess
import sys

import numpy as np
import pytest

from conflictbench import perception_component_power as power


def test_component_power_module_is_available() -> None:
    assert (
        importlib.util.find_spec("conflictbench.perception_component_power") is not None
    )


def _configuration() -> dict:
    return {
        "schema": "conflictbench.perception-component-power-config.v2",
        "analysis": "paired_component_sign_flip_power_approximation",
        "scope": {
            "dataset": "Perception Test",
            "derivation_partition": "training",
            "derivation_split_id": "perception_test_train_v1",
            "evaluation_outcomes_access": "forbidden",
            "policy_id": "value_of_evidence_policy_v1",
            "acquisition_budget": 0.50,
        },
        "simulation": {
            "repetitions": 10000,
            "randomization_draws": 999,
            "seed": 20270918,
            "minimum_detectable_error_reduction": 0.01,
            "two_sided_alpha": 0.05,
            "target_power": 0.80,
            "decision_confidence_level": 0.95,
            "null_calibration_max_rejection_rate": 0.06,
        },
        "paired_analysis": {
            "estimand": "event_weighted_absolute_error_reduction",
            "cluster_unit": "target_donor_connected_component",
            "test": "monte_carlo_component_sign_flip",
            "reference_distribution": "component_rademacher_sign_flips",
            "outcome_model": (
                "rademacher_orient_whole_observed_component_scores_to_global_mde"
            ),
            "component_size_model": "fixed_observed",
            "within_component_dependence": (
                "preserve_whole_observed_component_aggregates"
            ),
            "finite_sample_requirement": "minimum_20_components",
            "final_analysis_alignment": "pre_study_approximation_only",
        },
    }


def _write_json(path: pathlib.Path, value: dict) -> str:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _component_summary(
    *, event_count_per_component: int = 100, component_count: int = 20
) -> dict:
    if event_count_per_component < 100 or event_count_per_component % 100:
        raise ValueError("test component size must be a positive multiple of 100")
    component_ids = [f"{index:024x}" for index in range(component_count)]
    components = []
    source_ids: list[str] = []
    score_cycle = (-2, 0, 2, 0)
    scale = event_count_per_component // 100
    for index, component_id in enumerate(component_ids):
        component_sources = [f"source-{index:03d}-a", f"source-{index:03d}-b"]
        source_ids.extend(component_sources)
        discordant = 20 * scale
        score = score_cycle[index % len(score_cycle)] * scale
        baseline_only = (discordant + score) // 2
        proposed_only = (discordant - score) // 2
        both_error = 10 * scale
        both_correct = (
            event_count_per_component - baseline_only - proposed_only - both_error
        )
        components.append(
            {
                "component_id": component_id,
                "split_id": "perception_test_train_v1",
                "member_source_ids": component_sources,
                "event_counts": {
                    "both_correct": both_correct,
                    "baseline_error_only": baseline_only,
                    "proposed_error_only": proposed_only,
                    "both_error": both_error,
                },
            }
        )
    total = event_count_per_component * len(components)
    exclusions = [
        {
            "record_id": "excluded-record-001",
            "source_id": "excluded-source-001",
            "split_id": "perception_test_train_v1",
            "reason": "missing paired policy outcome",
        },
        {
            "record_id": "excluded-record-002",
            "source_id": "excluded-source-002",
            "split_id": "perception_test_train_v1",
            "reason": "source failed deterministic integrity check",
        },
    ]
    return {
        "schema": "conflictbench.perception-component-event-summary.v2",
        "derivation": {
            "dataset": "Perception Test",
            "partition": "training",
            "split_id": "perception_test_train_v1",
            "frozen_before_evaluation": True,
            "evaluation_outcomes_access": "forbidden",
            "event_unit": "target_question",
            "paired_outcomes": ["baseline_error", "proposed_error"],
            "policy_id": "value_of_evidence_policy_v1",
            "acquisition_budget": 0.50,
            "source_data_sha256": "1" * 64,
            "policy_configuration_sha256": "2" * 64,
            "summary_generator_sha256": "3" * 64,
        },
        "counts": {
            "eligible_target_count": total,
            "unique_source_count": len(source_ids),
            "connected_component_count": component_count,
            "exclusion_count": len(exclusions),
        },
        "exclusions": exclusions,
        "components": components,
    }


def test_configuration_locks_finite_sample_sign_flip_protocol(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "power.json"
    digest = _write_json(path, _configuration())

    loaded = power.load_power_configuration(path, digest)

    assert loaded == _configuration()
    assert loaded["paired_analysis"]["test"] == "monte_carlo_component_sign_flip"
    with pytest.raises(power.PowerValidationError, match="configuration SHA-256"):
        power.load_power_configuration(path, "0" * 64)


def test_component_summary_binds_training_provenance_and_exact_exclusions(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "summary.json"
    expected = _component_summary()
    digest = _write_json(path, expected)

    loaded = power.load_component_event_summary(path, digest)

    assert loaded == expected
    assert loaded["derivation"]["acquisition_budget"] == 0.50
    assert loaded["components"][0]["member_source_ids"] == [
        "source-000-a",
        "source-000-b",
    ]
    assert loaded["exclusions"][0]["reason"] == "missing paired policy outcome"

    invalid = copy.deepcopy(expected)
    invalid["derivation"]["partition"] = "validation"
    invalid_digest = _write_json(path, invalid)
    with pytest.raises(power.PowerValidationError, match="training"):
        power.load_component_event_summary(path, invalid_digest)


@pytest.mark.parametrize("numeric_true", [1, 1.0])
def test_component_summary_requires_boolean_frozen_flag(
    tmp_path: pathlib.Path, numeric_true: float
) -> None:
    path = tmp_path / "summary.json"
    invalid = _component_summary()
    invalid["derivation"]["frozen_before_evaluation"] = numeric_true
    digest = _write_json(path, invalid)

    with pytest.raises(power.PowerValidationError, match="frozen"):
        power.load_component_event_summary(path, digest)


def test_component_summary_requires_twenty_components_and_consistent_membership(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "summary.json"
    too_few = _component_summary(component_count=19)
    digest = _write_json(path, too_few)
    with pytest.raises(power.PowerValidationError, match="at least 20"):
        power.load_component_event_summary(path, digest)

    duplicate_member = _component_summary()
    duplicate_member["components"][1]["member_source_ids"][0] = duplicate_member[
        "components"
    ][0]["member_source_ids"][0]
    digest = _write_json(path, duplicate_member)
    with pytest.raises(power.PowerValidationError, match="one component"):
        power.load_component_event_summary(path, digest)


def test_component_summary_rejects_inconsistent_counts_and_exclusion_records(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "summary.json"
    invalid = _component_summary()
    invalid["counts"]["eligible_target_count"] += 1
    digest = _write_json(path, invalid)
    with pytest.raises(power.PowerValidationError, match="eligible target count"):
        power.load_component_event_summary(path, digest)

    invalid = _component_summary()
    invalid["exclusions"][1]["record_id"] = invalid["exclusions"][0]["record_id"]
    digest = _write_json(path, invalid)
    with pytest.raises(power.PowerValidationError, match="exclusion record IDs"):
        power.load_component_event_summary(path, digest)


def test_power_design_preserves_whole_component_aggregates_and_heterogeneity() -> None:
    summary = _component_summary()

    design = power.build_power_design(summary, minimum_detectable_effect=0.01)

    assert design["component_count"] == 20
    assert design["event_count"] == 2000
    assert design["expected_error_reduction"] == pytest.approx(0.01)
    assert design["observed_absolute_score_rate"] == pytest.approx(0.01)
    assert design["directional_orientation_bias"] == pytest.approx(1.0)
    assert design["alternative_positive_orientation_probability"] == pytest.approx(1.0)
    observed_effects = {
        item["observed_error_reduction"] for item in design["components"]
    }
    assert len(observed_effects) == 3
    assert len(summary["components"]) == len(design["components"])
    for source, component in zip(summary["components"], design["components"]):
        assert component["observed_event_counts"] == source["event_counts"]
        assert component["member_source_ids"] == source["member_source_ids"]
        observed_score = (
            source["event_counts"]["baseline_error_only"]
            - source["event_counts"]["proposed_error_only"]
        )
        assert component["absolute_score"] == abs(observed_score)
        assert "simulated_event_probabilities" not in component


def test_component_sign_flip_test_uses_whole_block_scores() -> None:
    component_scores = np.asarray(
        [[2.0] * 20, [2.0] * 10 + [-2.0] * 10], dtype=np.float64
    )
    component_sizes = np.asarray([10] * 20)
    reference_signs = np.asarray(
        [[1.0] * 10 + [-1.0] * 10, [-1.0] * 10 + [1.0] * 10, [-1.0] * 20],
        dtype=np.float64,
    )

    estimates, p_values, rejected = power.paired_component_sign_flip_test(
        component_scores,
        component_sizes,
        reference_signs=reference_signs,
        two_sided_alpha=0.05,
    )

    assert estimates.tolist() == pytest.approx([0.2, 0.0])
    assert p_values.tolist() == pytest.approx([0.5, 1.0])
    assert rejected.tolist() == [False, False]


def test_component_sign_flip_test_rejects_fewer_than_twenty_blocks() -> None:
    with pytest.raises(power.PowerValidationError, match="at least 20"):
        power.paired_component_sign_flip_test(
            np.ones((1, 19)),
            np.ones(19),
            reference_signs=np.ones((9, 19)),
            two_sided_alpha=0.05,
        )


def test_power_and_calibration_decisions_use_wilson_bounds() -> None:
    power_decision = power.wilson_decision(
        successes=80,
        repetitions=100,
        confidence_level=0.95,
        threshold=0.80,
        direction="lower_at_least",
    )
    calibration_decision = power.wilson_decision(
        successes=600,
        repetitions=10000,
        confidence_level=0.95,
        threshold=0.06,
        direction="upper_at_most",
    )

    assert power_decision["estimate"] == 0.80
    assert power_decision["passes"] is False
    assert power_decision["interval"]["lower"] < 0.80
    assert calibration_decision["estimate"] == 0.06
    assert calibration_decision["passes"] is False
    assert calibration_decision["interval"]["upper"] > 0.06


def test_simulation_is_deterministic_calibrated_and_hash_bound(
    tmp_path: pathlib.Path,
) -> None:
    config_path = tmp_path / "config.json"
    summary_path = tmp_path / "summary.json"
    config_sha256 = _write_json(config_path, _configuration())
    summary = _component_summary(event_count_per_component=5000, component_count=40)
    summary_sha256 = _write_json(summary_path, summary)
    source_paths = {
        "perception_component_power.py": pathlib.Path(power.__file__).resolve(),
    }

    first = power.run_power_simulation(
        config_path=config_path,
        expected_config_sha256=config_sha256,
        summary_path=summary_path,
        expected_summary_sha256=summary_sha256,
        source_paths=source_paths,
    )
    second = power.run_power_simulation(
        config_path=config_path,
        expected_config_sha256=config_sha256,
        summary_path=summary_path,
        expected_summary_sha256=summary_sha256,
        source_paths=source_paths,
    )

    assert first == second
    assert first["schema"] == "conflictbench.perception-component-power-report.v2"
    assert first["analysis"]["role"] == "pre_study_power_approximation"
    assert first["design"]["component_count"] == 40
    assert first["design"]["event_count"] == 200000
    assert first["design"]["expected_error_reduction"] == pytest.approx(0.01)
    assert first["simulation"]["repetitions"] == 10000
    assert first["simulation"]["null_calibration"]["passes"] is True
    assert first["simulation"]["power"]["interval"]["lower"] >= 0.80
    assert first["simulation"]["power"]["meets_target_power"] is True
    assert first["runtime"] == {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    assert first["input_digests"]["source_data_sha256"] == "1" * 64
    assert first["input_digests"]["policy_configuration_sha256"] == "2" * 64
    assert first["input_digests"]["summary_generator_sha256"] == "3" * 64
    assert len(first["input_digests"]["component_membership_sha256"]) == 64
    assert len(first["input_digests"]["exclusions_sha256"]) == 64
    assert len(first["payload_sha256"]) == 64
    power.validate_power_output(
        first,
        expected_config_sha256=config_sha256,
        expected_summary_sha256=summary_sha256,
        expected_implementation_bundle_sha256=first["input_digests"][
            "implementation_bundle_sha256"
        ],
    )

    tampered = copy.deepcopy(first)
    tampered["simulation"]["power"]["rejection_count"] -= 1
    with pytest.raises(power.PowerValidationError, match="payload SHA-256"):
        power.validate_power_output(tampered)

    tampered["payload_sha256"] = _canonical_digest(
        {key: tampered[key] for key in tampered if key != "payload_sha256"}
    )
    with pytest.raises(power.PowerValidationError, match="deterministic replay"):
        power.validate_power_output(tampered)


def test_power_output_is_write_once(tmp_path: pathlib.Path) -> None:
    config_path = tmp_path / "config.json"
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "power-output.json"
    config_sha256 = _write_json(config_path, _configuration())
    summary_sha256 = _write_json(summary_path, _component_summary())
    result = power.run_power_simulation(
        config_path=config_path,
        expected_config_sha256=config_sha256,
        summary_path=summary_path,
        expected_summary_sha256=summary_sha256,
        source_paths={
            "perception_component_power.py": pathlib.Path(power.__file__).resolve()
        },
    )

    power.write_power_output(output_path, result)

    assert json.loads(output_path.read_text(encoding="utf-8")) == result
    with pytest.raises(power.PowerValidationError, match="already exists"):
        power.write_power_output(output_path, result)


def test_public_configuration_matches_the_finite_sample_protocol() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "perception_component_power.json"
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()

    assert power.load_power_configuration(config_path, digest) == _configuration()


def test_public_runner_writes_a_valid_hash_bound_report(
    tmp_path: pathlib.Path,
) -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "run_perception_component_power.py"
    config_path = root / "configs" / "perception_component_power.json"
    summary_path = tmp_path / "summary.json"
    output_path = tmp_path / "report.json"
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    summary_sha256 = _write_json(summary_path, _component_summary())
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--config",
            str(config_path),
            "--config-sha256",
            config_sha256,
            "--component-event-summary",
            str(summary_path),
            "--component-event-summary-sha256",
            summary_sha256,
            "--output",
            str(output_path),
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    power.validate_power_output(
        report,
        expected_config_sha256=config_sha256,
        expected_summary_sha256=summary_sha256,
    )
    assert [item["role"] for item in report["implementation_files"]] == [
        "perception_component_power.py",
        "run_perception_component_power.py",
    ]
