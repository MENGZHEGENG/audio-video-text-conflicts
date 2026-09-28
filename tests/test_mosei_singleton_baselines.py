import copy
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import conflictbench.mosei_singleton_baselines as singleton
import numpy as np
import pytest
import sklearn

RUN_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_mosei_singleton_baselines.py"
VERIFY_SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "verify_mosei_singleton_baselines.py"
)
AGGREGATE_SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "aggregate_mosei_singleton_baselines.py"
)
VERIFY_AGGREGATE_SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "verify_mosei_singleton_aggregate.py"
)


def _write_cache(path, *, invert_valid=False):
    rng = np.random.default_rng(19)
    train_groups = [f"train-{index:02d}" for index in range(18)]
    valid_groups = [f"valid-{index:02d}" for index in range(6)]
    train_ids = np.asarray(
        [f"{group}[{segment}]" for group in train_groups for segment in range(2)],
        dtype="U",
    )
    valid_ids = np.asarray(
        [f"{group}[{segment}]" for group in valid_groups for segment in range(2)],
        dtype="U",
    )
    train_y = np.asarray(
        [(index // 2) % 2 for index in range(len(train_ids))], dtype=np.int64
    )
    valid_y = np.asarray(
        [(index // 2) % 2 for index in range(len(valid_ids))], dtype=np.int64
    )
    train_signal = 2 * train_y - 1
    valid_signal = 2 * valid_y - 1
    train_x = train_signal[:, None] + rng.normal(0.0, 1.0, size=(len(train_ids), 3))
    valid_x = valid_signal[:, None] + rng.normal(0.0, 1.0, size=(len(valid_ids), 3))
    if invert_valid:
        valid_x = -valid_x
        valid_y = 1 - valid_y
    inaccessible_test = np.asarray([{"do_not_read": True}], dtype=object)
    metadata = {
        "alignment": "positive_interval_overlap_mean",
        "loaded_split_counts": {
            "train": len(train_ids),
            "valid": len(valid_ids),
            "test": 1,
        },
    }
    np.savez_compressed(
        path,
        cache_schema=np.asarray("conflictbench.mosei-cache.v2"),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        train_x=train_x.astype(np.float32),
        train_y=train_y,
        train_sample_ids=train_ids,
        valid_x=valid_x.astype(np.float32),
        valid_y=valid_y,
        valid_sample_ids=valid_ids,
        test_x=inaccessible_test,
        test_y=inaccessible_test,
        test_sample_ids=inaccessible_test,
    )


def _config():
    return {
        "schema": "conflictbench.mosei-singleton-baselines.v3",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "dataset": "CMU-MOSEI",
        "expected_split_counts": {"train": 36, "valid": 12, "test": 1},
        "train_group_fractions": [0.5, 0.25, 0.25],
        "seeds": [11],
        "budgets": [0.0, 0.5, 1.0],
        "primary_budget": 0.5,
        "task_heads": {
            "ridge": 0.01,
            "modality_dropout_draws_per_example": 3,
            "minimum_validation_accuracy_per_mask": 0.0,
            "families": [
                "separate_mask_ridge",
                "shared_mask_aware_ridge",
                "modality_dropout_ridge",
            ],
        },
        "action_classifier": {
            "risk_ridge": 0.1,
            "calibration_ridge": 0.1,
            "cost_units": "normalized_zero_one_task_loss",
            "acquisition_cost_semantics": "fixed analysis sensitivity, not measured latency",
            "action_order": [
                "answer",
                "abstain",
                "acquire_audio",
                "acquire_video",
                "acquire_text",
            ],
            "cost_matrix": {
                "correct_answer": 0.0,
                "incorrect_answer": 1.0,
                "abstain": 0.25,
                "acquisition": {"audio": 0.05, "video": 0.05, "text": 0.05},
            },
        },
        "task_head_sensitivity": {
            "maximum_primary_cost_reduction_range": 0.05,
            "require_matching_sign": True,
        },
        "uncertainty": {
            "unit": "validation_video_group",
            "bootstrap_repetitions": 40,
            "bootstrap_seed": 20270917,
        },
        "published_comparator": {
            "status": "excluded_incompatible",
            "benchmark_only": True,
            "selected": None,
            "audit_path": "docs/mosei_published_comparator_compatibility.md",
            "sources": [
                "https://arxiv.org/abs/2211.05039",
                "https://proceedings.mlr.press/v235/valancius24a.html",
                "https://github.com/lupalab/aaco",
            ],
        },
    }


def test_task_and_action_features_ignore_every_hidden_value():
    scores = np.asarray(
        [[0.25, 3.0, -7.0], [-0.75, 5.0, 11.0]],
        dtype=np.float64,
    )
    changed = scores.copy()
    changed[:, 1:] = np.asarray([[1.0e12, np.nan], [-1.0e12, np.nan]])
    observed = np.asarray([True, False, False])
    labels = np.asarray([1, 0], dtype=np.int64)

    for family in singleton.TASK_HEAD_FAMILIES:
        task_head, _ = singleton._fit_task_head(
            family,
            scores,
            labels,
            ridge=0.01,
            seed=11,
            dropout_draws=3,
        )
        base_score = singleton._task_scores(scores, observed, task_head)
        changed_base_score = singleton._task_scores(changed, observed, task_head)
        np.testing.assert_allclose(base_score, changed_base_score)

        np.testing.assert_allclose(
            singleton._observed_action_features(scores, observed, base_score),
            singleton._observed_action_features(changed, observed, changed_base_score),
        )


def test_exact_budget_selection_is_deterministic_and_has_the_requested_count():
    priority = np.asarray([0.4, 0.4, 0.2, -0.1, -0.2], dtype=np.float64)

    first = singleton._select_exact_budget(priority, 0.4, seed=17)
    repeated = singleton._select_exact_budget(priority, 0.4, seed=17)

    assert int(first.sum()) == 2
    np.testing.assert_array_equal(first, repeated)


def test_nonintegral_exact_budget_validates_against_the_rounded_count(tmp_path):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    config = _config()
    config["budgets"] = [0.0, 0.1, 1.0]
    config["primary_budget"] = 0.1

    result = singleton.run_singleton_baselines(cache, config, seed=11)

    assert (
        singleton.validate_singleton_baseline_result(result, expected_config=config)
        == []
    )
    for family in singleton.TASK_HEAD_FAMILIES:
        metrics = result["task_heads"][family]["exact_budget"]["0.100000"]
        assert metrics["query_count"] == 4
        assert metrics["query_rate"] == pytest.approx(4 / 36)


def test_hidden_values_cannot_change_query_priority_or_source_choice():
    scores = np.asarray([[0.25, 3.0, -7.0], [-0.75, 5.0, 11.0]], dtype=np.float64)
    changed = scores.copy()
    changed[:, 1:] = np.asarray([[1.0e12, -1.0e12], [-1.0e12, 1.0e12]])
    labels = np.asarray([1, 0], dtype=np.int64)
    groups = np.asarray(["a", "b"])
    task_head, _ = singleton._fit_task_head(
        "separate_mask_ridge",
        scores,
        labels,
        ridge=0.01,
        seed=11,
        dropout_draws=3,
    )

    def first_context(values):
        full = singleton._make_action_decisions(values, labels, groups, task_head)
        size = len(values)
        return singleton._ActionDecisions(
            labels=full.labels[:size],
            state_features=full.state_features[:size],
            base_loss=full.base_loss[:size],
            base_confidence=full.base_confidence[:size],
            candidate_loss=full.candidate_loss[:size],
            candidate_confidence=full.candidate_confidence[:size],
            candidate_modality=full.candidate_modality[:size],
            context_index=full.context_index[:size],
            groups=full.groups[:size],
        )

    original = first_context(scores)
    mutated = first_context(changed)
    np.testing.assert_allclose(original.state_features, mutated.state_features)
    width = original.state_features.shape[1]
    weights = np.linspace(-0.2, 0.2, 1 + width + 4 + 4 * width)
    model = {
        "mean": np.zeros(width),
        "scale": np.ones(width),
        "weights": weights,
        "maximum_cost": 1.05,
    }
    original_risk = singleton._predict_raw_risk(original, model)
    mutated_risk = singleton._predict_raw_risk(mutated, model)
    np.testing.assert_allclose(original_risk, mutated_risk, equal_nan=True)
    calibrators = {(0, action): np.asarray([0.0, 1.0]) for action in (0, 2, 3)}
    original_cost = singleton._predict_calibrated_risk(
        original, original_risk, calibrators, maximum_cost=1.05
    )
    mutated_cost = singleton._predict_calibrated_risk(
        mutated, mutated_risk, calibrators, maximum_cost=1.05
    )
    base_error = np.asarray([0.2, 0.3])
    original_components = singleton._policy_components(
        original,
        original_cost,
        (base_error, np.asarray([[0.1, 0.9], [0.8, 0.2]])),
        _config(),
    )
    mutated_components = singleton._policy_components(
        mutated,
        mutated_cost,
        (base_error, np.asarray([[0.9, 0.1], [0.2, 0.8]])),
        _config(),
    )
    np.testing.assert_array_equal(
        original_components["choice_modality"], mutated_components["choice_modality"]
    )
    np.testing.assert_allclose(
        original_components["priority"], mutated_components["priority"]
    )


def test_run_is_deterministic_group_isolated_and_exact_budgeted(tmp_path):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    config = _config()

    first = singleton.run_singleton_baselines(cache, config, seed=11)
    repeated = singleton.run_singleton_baselines(cache, config, seed=11)

    assert first == repeated
    assert (
        singleton.validate_singleton_baseline_result(first, expected_config=config)
        == []
    )
    assert first["data_access"]["loaded_splits"] == ["train", "valid"]
    assert first["data_access"]["test_opened"] is False
    assert first["group_partitions"]["pairwise_disjoint"] is True
    assert first["action_classifier"]["fit_split"] == "router_fit"
    assert first["action_classifier"]["calibration_split"] == "calibration"
    assert first["action_classifier"]["decision_contract"]["no_query_cost"].startswith(
        "calibrated action-risk"
    )
    assert first["published_comparator"]["benchmark_only"] is True
    assert first["published_comparator"]["selected"] is None
    expected_count = {"0.000000": 0, "0.500000": 18, "1.000000": 36}
    for family in config["task_heads"]["families"]:
        result = first["task_heads"][family]
        assert set(result["validation_accuracy_by_mask"]) == {
            "audio",
            "video",
            "text",
            "audio+video",
            "audio+text",
            "video+text",
            "audio+video+text",
        }
        for budget, count in expected_count.items():
            metrics = result["exact_budget"][budget]
            assert metrics["query_count"] == count
            assert sum(metrics["initial_action_counts"].values()) == 36
        assert result["validation_accuracy_gate"]["passed"] is True
    assert first["task_head_sensitivity"]["primary_budget"] == 0.5
    assert first["task_head_sensitivity"]["task_head_validation_passed"] is True


def test_validation_values_cannot_change_any_fitted_training_record(tmp_path):
    first_cache = tmp_path / "first.npz"
    second_cache = tmp_path / "second.npz"
    _write_cache(first_cache)
    _write_cache(second_cache, invert_valid=True)

    first = singleton.run_singleton_baselines(first_cache, _config(), seed=11)
    second = singleton.run_singleton_baselines(second_cache, _config(), seed=11)

    assert first["training_attestation_sha256"] == second["training_attestation_sha256"]
    assert first["action_classifier"] == second["action_classifier"]


def test_validator_rejects_budget_and_training_provenance_tampering(tmp_path):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    config = _config()
    result = singleton.run_singleton_baselines(cache, config, seed=11)

    changed = copy.deepcopy(result)
    changed["task_heads"]["separate_mask_ridge"]["exact_budget"]["0.500000"][
        "query_count"
    ] -= 1
    assert (
        "exact query budget is invalid"
        in singleton.validate_singleton_baseline_result(changed, expected_config=config)
    )

    changed = copy.deepcopy(result)
    changed["action_classifier"]["training_risk_sha256"] = "0" * 64
    assert (
        "training provenance is invalid"
        in singleton.validate_singleton_baseline_result(changed, expected_config=config)
    )

    changed = copy.deepcopy(result)
    changed["implementation_sha256"] = "0" * 64
    assert (
        "implementation digest is invalid"
        in singleton.validate_singleton_baseline_result(changed, expected_config=config)
    )


def test_validator_reconstructs_group_and_task_head_sensitivity_gates(tmp_path):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    config = _config()
    result = singleton.run_singleton_baselines(cache, config, seed=11)

    changed = copy.deepcopy(result)
    changed["group_partitions"]["router_fit"]["group_hashes"][0] = changed[
        "group_partitions"
    ]["task_fit"]["group_hashes"][0]
    assert (
        "group partitions are invalid"
        in singleton.validate_singleton_baseline_result(changed, expected_config=config)
    )

    changed = copy.deepcopy(result)
    changed["task_head_sensitivity"]["observed_range"] += 0.1
    assert "task-head sensitivity result is invalid" in (
        singleton.validate_singleton_baseline_result(changed, expected_config=config)
    )


def test_task_head_accuracy_floor_is_a_locked_interpretation_gate(tmp_path):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    config = _config()
    config["task_heads"]["minimum_validation_accuracy_per_mask"] = 1.0

    result = singleton.run_singleton_baselines(cache, config, seed=11)

    assert any(
        not result["task_heads"][family]["validation_accuracy_gate"]["passed"]
        for family in singleton.TASK_HEAD_FAMILIES
    )
    assert result["task_head_sensitivity"]["task_head_validation_passed"] is False
    assert result["task_head_sensitivity"]["passed"] is False
    assert (
        singleton.validate_singleton_baseline_result(result, expected_config=config)
        == []
    )


def test_config_requires_a_benchmark_only_comparator_decision():
    config = _config()
    config["published_comparator"]["benchmark_only"] = False
    config["published_comparator"]["selected"] = "A2MT"

    with pytest.raises(ValueError, match="benchmark-only"):
        singleton._validate_config(config)


def test_validation_is_opened_only_after_all_training_and_calibration(
    tmp_path, monkeypatch
):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    events = []
    real_load = singleton._load_split
    real_fit = singleton._fit_one_family

    def tracked_load(*args, **kwargs):
        split_name = args[2]
        events.append(f"load:{split_name}")
        return real_load(*args, **kwargs)

    def tracked_fit(*args, **kwargs):
        events.append(f"fit:{args[0]}")
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(singleton, "_load_split", tracked_load)
    monkeypatch.setattr(singleton, "_fit_one_family", tracked_fit)
    singleton.run_singleton_baselines(cache, _config(), seed=11)

    assert events == [
        "load:train",
        "fit:separate_mask_ridge",
        "fit:shared_mask_aware_ridge",
        "fit:modality_dropout_ridge",
        "load:valid",
    ]


def test_cost_matrix_can_choose_answer_abstain_and_a_valid_acquisition():
    decisions = singleton._ActionDecisions(
        labels=np.asarray([1, 0]),
        state_features=np.zeros((2, 10)),
        base_loss=np.asarray([0.0, 1.0]),
        base_confidence=np.asarray([0.5, 0.5]),
        candidate_loss=np.asarray([[0.0, 1.0], [1.0, 1.0]]),
        candidate_confidence=np.asarray([[0.5, 0.5], [0.5, 0.5]]),
        candidate_modality=np.asarray([[1, 2], [0, 2]]),
        context_index=np.asarray([0, 1]),
        groups=np.asarray(["a", "b"]),
    )
    # Columns are answer, acquire-audio, acquire-video, acquire-text.
    calibrated_risk = np.asarray(
        [[0.10, np.nan, 0.00, 0.90], [0.90, 0.95, np.nan, 0.95]],
        dtype=np.float64,
    )

    terminal_error = (
        np.asarray([0.10, 0.90]),
        np.asarray([[0.00, 0.90], [0.90, 0.90]]),
    )
    components = singleton._policy_components(
        decisions,
        calibrated_risk,
        terminal_error,
        _config(),
    )

    np.testing.assert_array_equal(components["terminal_answer"], [True, False])
    np.testing.assert_array_equal(components["choice_modality"], [1, 0])
    np.testing.assert_array_equal(components["candidate_answer"], [True, False])
    assert components["priority"][0] > 0.0
    assert components["priority"][1] < 0.0


def test_command_line_verifier_replays_the_cache_and_rejects_tampering(tmp_path):
    cache = tmp_path / "scores.npz"
    config_path = tmp_path / "config.json"
    output = tmp_path / "result.json"
    _write_cache(cache)
    config_path.write_text(json.dumps(_config(), sort_keys=True), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--config",
            str(config_path),
            "--cache",
            str(cache),
            "--seed",
            "11",
            "--output",
            str(output),
        ],
        check=True,
    )
    verified = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT),
            str(output),
            "--config",
            str(config_path),
            "--cache",
            str(cache),
            "--seed",
            "11",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert verified.returncode == 0, verified.stderr

    changed = json.loads(output.read_text(encoding="utf-8"))
    changed["task_heads"]["separate_mask_ridge"]["exact_budget"]["0.500000"][
        "query_count"
    ] -= 1
    output.write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
    rejected = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT),
            str(output),
            "--config",
            str(config_path),
            "--cache",
            str(cache),
            "--seed",
            "11",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "exact query budget is invalid" in rejected.stderr


def test_complete_seed_set_aggregate_is_recomputed_exactly(tmp_path):
    cache = tmp_path / "scores.npz"
    config_path = tmp_path / "config.json"
    result_path = tmp_path / "singleton-11.json"
    aggregate_path = tmp_path / "aggregate.json"
    _write_cache(cache)
    config = _config()
    config["expected_cache_sha256"] = hashlib.sha256(cache.read_bytes()).hexdigest()
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    result = singleton.run_singleton_baselines(cache, config, seed=11)
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(AGGREGATE_SCRIPT),
            "--config",
            str(config_path),
            "--input",
            str(result_path),
            "--output",
            str(aggregate_path),
        ],
        check=True,
    )
    verified = subprocess.run(
        [
            sys.executable,
            str(VERIFY_AGGREGATE_SCRIPT),
            str(aggregate_path),
            "--config",
            str(config_path),
            "--input",
            str(result_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["seeds"] == [11]
    assert aggregate["implementation_sha256"] == singleton.implementation_sha256()
    assert (
        aggregate["decision"]["all_task_head_sensitivity_passed"]
        == result["task_head_sensitivity"]["passed"]
    )
    assert aggregate["decision"]["status"] in {
        "positive_acquisition_supported",
        "negative_acquisition_supported",
        "inconclusive",
    }

    changed = copy.deepcopy(result)
    changed["implementation_sha256"] = "0" * 64
    changed_path = tmp_path / "singleton-11-wrong-implementation.json"
    changed_path.write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
    rejected = subprocess.run(
        [
            sys.executable,
            str(AGGREGATE_SCRIPT),
            "--config",
            str(config_path),
            "--input",
            str(changed_path),
            "--output",
            str(tmp_path / "aggregate-wrong-implementation.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "implementation digest is invalid" in rejected.stderr


def test_task_head_sensitivity_failure_forces_inconclusive_aggregate(tmp_path):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    config = _config()
    config["expected_cache_sha256"] = hashlib.sha256(cache.read_bytes()).hexdigest()
    config["task_head_sensitivity"]["maximum_primary_cost_reduction_range"] = 0.0

    result = singleton.run_singleton_baselines(cache, config, seed=11)
    assert result["task_head_sensitivity"]["passed"] is False
    assert result["acquisition_evidence"]["status"] == "inconclusive"

    aggregate = singleton.aggregate_singleton_baseline_results([result], config)
    assert aggregate["decision"]["task_head_sensitivity_passed_by_seed"] == {
        "11": False
    }
    assert aggregate["decision"]["all_task_head_sensitivity_passed"] is False
    assert aggregate["decision"]["positive_acquisition_supported"] is False
    assert aggregate["decision"]["negative_acquisition_supported"] is False
    assert aggregate["decision"]["status"] == "inconclusive"


def test_negative_intervals_cannot_bypass_task_head_sensitivity_gate(tmp_path):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    config = _config()
    config["expected_cache_sha256"] = hashlib.sha256(cache.read_bytes()).hexdigest()
    config["task_head_sensitivity"]["maximum_primary_cost_reduction_range"] = 0.0
    config["action_classifier"]["cost_matrix"]["acquisition"] = {
        modality: 2.0 for modality in singleton.MODALITIES
    }

    result = singleton.run_singleton_baselines(cache, config, seed=11)

    assert all(
        result["acquisition_evidence"]["interval_negative_by_task_head"].values()
    )
    assert result["task_head_sensitivity"]["agreement_passed"] is False
    assert result["task_head_sensitivity"]["passed"] is False
    assert result["acquisition_evidence"]["negative_acquisition_supported"] is False
    assert result["acquisition_evidence"]["status"] == "inconclusive"
    assert (
        singleton.validate_singleton_baseline_result(
            result,
            expected_seed=11,
            expected_cache_sha256=config["expected_cache_sha256"],
            expected_config=config,
        )
        == []
    )

    aggregate = singleton.aggregate_singleton_baseline_results([result], config)
    assert aggregate["decision"]["all_task_head_sensitivity_passed"] is False
    assert aggregate["decision"]["negative_acquisition_supported"] is False
    assert aggregate["decision"]["status"] == "inconclusive"
