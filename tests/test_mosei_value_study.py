import copy
import importlib.util
import json
import sys
from pathlib import Path

import conflictbench.mosei_value_study as study
import numpy as np
import pytest
from conflictbench.mosei_value_study import (
    configuration_sha256,
    load_training_and_validation,
    run_value_study,
    validate_value_result,
)

AGGREGATE_SCRIPT = Path(__file__).parents[1] / "scripts" / "aggregate_mosei_value.py"
VERIFY_SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_mosei_value.py"


def _write_cache(path):
    rng = np.random.default_rng(7)
    train_groups = [f"train-{index:02d}" for index in range(12)]
    valid_groups = [f"valid-{index:02d}" for index in range(6)]

    train_ids = np.asarray(
        [f"{group}[{segment}]" for group in train_groups for segment in range(2)],
        dtype="U",
    )
    valid_ids = np.asarray(
        [f"{group}[{segment}]" for group in valid_groups for segment in range(2)],
        dtype="U",
    )
    train_y = np.asarray([(index // 2) % 2 for index in range(len(train_ids))], dtype=np.int64)
    valid_y = np.asarray([(index // 2) % 2 for index in range(len(valid_ids))], dtype=np.int64)
    train_signal = 2 * train_y - 1
    valid_signal = 2 * valid_y - 1
    train_x = train_signal[:, None] + rng.normal(0.0, 0.8, size=(len(train_ids), 3))
    valid_x = valid_signal[:, None] + rng.normal(0.0, 0.8, size=(len(valid_ids), 3))

    # Object-valued test arrays raise if a loader tries to read them with
    # allow_pickle=False.  The pilot must leave them unopened.
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
        "schema": "conflictbench.mosei-value-study.v2",
        "expected_split_counts": {"train": 24, "valid": 12, "test": 1},
        "train_group_fractions": [0.5, 0.25, 0.25],
        "budgets": [0.0, 0.5, 1.0],
        "coverage_targets": [1.0, 0.9, 0.8],
        "ridge": 0.01,
        "benefit_ridge": 0.1,
        "confidence_ridge": 0.1,
        "bootstrap_repetitions": 40,
        "bootstrap_seed": 20270917,
        "headroom_budget": 0.5,
        "primary_budget": 0.5,
        "minimum_oracle_error_reduction": 0.0,
        "minimum_primary_error_reduction_gain": 0.0,
        "minimum_video_macro_gain_fraction": 0.0,
        "minimum_nonnegative_start_contexts": 0,
        "maximum_start_context_harm": 1.0,
        "task_head_reference_name": "test reference",
        "reference_full_avt_validation_accuracy": 0.5,
        "task_head_reference_sha256": "0" * 64,
        "maximum_task_head_accuracy_gap": 0.5,
        "minimum_mask_validation_accuracy": 0.0,
        "router_ladder": {
            "inner_group_folds": 3,
            "selection_budget": 0.5,
            "selection_metric": "mean_held_out_error_reduction",
            "family_tie_order": [
                "linear_value",
                "hist_gbt_value",
                "shallow_mlp_value",
            ],
            "scikit_learn_version": "1.9.0",
            "hist_gbt_grid": [
                {
                    "max_leaf_nodes": 7,
                    "learning_rate": 0.05,
                    "max_iter": 8,
                    "min_samples_leaf": 2,
                    "l2_regularization": 1.0,
                    "max_bins": 31,
                    "early_stopping": False,
                }
            ],
            "shallow_mlp_grid": [
                {
                    "hidden_layer_sizes": [4],
                    "activation": "relu",
                    "solver": "adam",
                    "alpha": 0.001,
                    "learning_rate_init": 0.001,
                    "batch_size": 16,
                    "max_iter": 20,
                    "early_stopping": False,
                }
            ],
        },
    }


def test_router_ladder_selects_on_calibration_and_exactly_aliases_the_winner(tmp_path):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    result = run_value_study(cache, _config(), mode="singleton", seed=11)

    ladder = result["router_ladder"]
    selected = ladder["selection"]["selected_family"]
    assert selected in {"linear_value", "hist_gbt_value", "shallow_mlp_value"}
    assert ladder["selection"]["split"] == "calibration"
    assert ladder["selection"]["budget"] == 0.5
    for budget in result["evaluation"]["policies"]["selected_router"]:
        assert (
            result["evaluation"]["policies"]["selected_router"][budget]
            == result["evaluation"]["policies"][selected][budget]
        )
        assert (
            result["bootstrap"]["policy_error_reduction_draws"]["selected_router"][budget]
            == result["bootstrap"]["policy_error_reduction_draws"][selected][budget]
        )


def test_router_inner_folds_are_group_disjoint_and_complete(tmp_path):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    result = run_value_study(cache, _config(), mode="singleton", seed=23)

    folds = result["router_ladder"]["inner_cv"]["folds"]
    assert len(folds) == 3
    assert all(fold["group_disjoint"] for fold in folds)
    heldout = [digest for fold in folds for digest in fold["heldout_group_hashes"]]
    assert sorted(heldout) == sorted(result["group_partitions"]["router_fit"]["group_hashes"])


def test_validation_changes_cannot_change_router_training_or_selection(tmp_path):
    first_cache = tmp_path / "first.npz"
    second_cache = tmp_path / "second.npz"
    _write_cache(first_cache)
    _write_cache(second_cache)
    with np.load(second_cache, allow_pickle=True) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["valid_x"] = -np.asarray(payload["valid_x"])
    payload["valid_y"] = 1 - np.asarray(payload["valid_y"])
    np.savez_compressed(second_cache, **payload)

    first = run_value_study(first_cache, _config(), mode="singleton", seed=37)
    second = run_value_study(second_cache, _config(), mode="singleton", seed=37)
    assert (
        first["router_ladder"]["training_attestation_sha256"]
        == second["router_ladder"]["training_attestation_sha256"]
    )
    assert first["router_ladder"]["selection"] == second["router_ladder"]["selection"]


def test_official_validation_is_loaded_only_after_router_selection(tmp_path, monkeypatch):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    events = []
    real_load = study._load_score_split
    real_fit = study._fit_router_ladder

    def tracked_load(path, config, split_name):
        events.append(f"load:{split_name}")
        return real_load(path, config, split_name)

    def tracked_fit(*args, **kwargs):
        events.append("fit_router_ladder")
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(study, "_load_score_split", tracked_load)
    monkeypatch.setattr(study, "_fit_router_ladder", tracked_fit)
    run_value_study(cache, _config(), mode="singleton", seed=11)

    assert events.index("fit_router_ladder") < events.index("load:valid")


def test_router_family_ties_follow_the_locked_order():
    scores = {
        "linear_value": 0.125,
        "hist_gbt_value": 0.125,
        "shallow_mlp_value": 0.125,
    }
    assert study._select_router_family(
        scores,
        ("linear_value", "hist_gbt_value", "shallow_mlp_value"),
    ) == "linear_value"


def test_validator_rejects_tampered_router_cv_and_prediction_provenance(tmp_path):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    config = _config()
    result = run_value_study(cache, config, mode="singleton", seed=11)
    assert validate_value_result(result, expected_config=config) == []

    changed = copy.deepcopy(result)
    changed["router_ladder"]["families"]["hist_gbt_value"]["cv_candidates"][0][
        "mean_held_out_error_reduction"
    ] += 0.25
    assert "router-ladder provenance is invalid" in validate_value_result(
        changed,
        expected_config=config,
    )

    changed = copy.deepcopy(result)
    changed["router_ladder"]["official_validation_prediction_sha256"]["linear_value"] = "0" * 64
    assert "router-ladder provenance is invalid" in validate_value_result(
        changed,
        expected_config=config,
    )


def test_config_rejects_internal_router_early_stopping(tmp_path):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    config = _config()
    config["router_ladder"]["shallow_mlp_grid"][0]["early_stopping"] = True
    with pytest.raises(ValueError, match="disable internal early stopping"):
        run_value_study(cache, config, mode="singleton", seed=11)


def test_config_requires_three_router_group_folds(tmp_path):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    config = _config()
    config["router_ladder"]["inner_group_folds"] = 2
    with pytest.raises(ValueError, match="must equal three"):
        run_value_study(cache, config, mode="singleton", seed=11)


def test_loader_never_opens_test_arrays(tmp_path):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    loaded = load_training_and_validation(cache, _config())
    assert tuple(loaded) == ("train", "valid")
    assert loaded["train"].x.shape == (24, 3)
    assert loaded["valid"].x.shape == (12, 3)


def test_value_study_is_deterministic_and_group_disjoint(tmp_path):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    first = run_value_study(cache, _config(), mode="singleton", seed=11)
    second = run_value_study(cache, _config(), mode="singleton", seed=11)

    assert first == second
    assert first["validation"] == {"ok": True, "errors": []}
    assert first["data_access"]["loaded_splits"] == ["train", "valid"]
    assert first["data_access"]["test_opened"] is False
    assert first["group_partitions"]["pairwise_disjoint"] is True
    assert first["evaluation"]["split"] == "official_validation"
    assert first["evaluation"]["decision_instances"] == 36
    assert set(first["evaluation"]["policies"]) == {
        "confidence",
        "confidence_query_selected_router_choice",
        "hist_gbt_value",
        "linear_value",
        "online_selected_router",
        "oracle",
        "random_matched",
        "selected_baseline",
        "selected_router",
        "selected_router_query_source_prior_choice",
        "shallow_mlp_value",
        "source_prior",
    }
    for policy in first["evaluation"]["policies"].values():
        assert set(policy) == {"0.000000", "0.500000", "1.000000"}


def test_pair_mode_has_one_decision_for_each_missing_modality(tmp_path):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    result = run_value_study(cache, _config(), mode="pair", seed=19)
    assert result["mode"] == "pair"
    assert result["evaluation"]["decision_instances"] == 36
    assert result["evaluation"]["candidate_instances"] == 36
    for policy in result["evaluation"]["policies"].values():
        for metrics in policy.values():
            assert metrics["candidate_ranking_evaluable"] == 0
            assert metrics["candidate_ranking_accuracy"] is None
            assert metrics["queried_candidate_selection_accuracy"] is None
            assert metrics["candidate_value_tie_rate"] is None


def test_prequery_features_ignore_hidden_modality_values():
    scores = np.asarray([[0.2, 7.0, -3.0], [-0.8, 4.0, 9.0]], dtype=np.float64)
    observed = np.asarray([True, False, False])
    changed = scores.copy()
    changed[:, 1:] = [[1e12, np.nan], [-1e12, np.nan]]
    base_score = np.asarray([0.3, -0.4])

    np.testing.assert_allclose(
        study._task_features(scores, observed),
        study._task_features(changed, observed),
    )
    np.testing.assert_allclose(
        study._router_features(scores, observed, 1, base_score),
        study._router_features(changed, observed, 1, base_score),
    )


def test_exact_score_ties_use_a_reproducible_seeded_order():
    scores = np.ones(8, dtype=np.float64)
    first = study._seeded_descending_order(scores, seed=17)
    repeated = study._seeded_descending_order(scores, seed=17)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first[:3], np.arange(3))


def test_online_query_threshold_uses_only_calibration_scores_and_has_explicit_ties():
    calibration_priority = np.asarray([0.9, 0.8, 0.8, 0.1], dtype=np.float64)

    fitted = study._fit_online_query_threshold(calibration_priority, 0.5)

    assert fitted == {
        "target_query_rate": 0.5,
        "target_query_count": 2,
        "calibration_decisions": 4,
        "calibration_query_count": 1,
        "calibration_realized_query_rate": 0.25,
        "threshold": 0.8,
        "query_on_equal": False,
        "tie_convention": "priority_strictly_greater_than_threshold",
    }
    evaluation_priority = np.asarray([0.85, 0.8, 0.79], dtype=np.float64)
    original = study._apply_online_query_threshold(evaluation_priority, fitted)
    extended = study._apply_online_query_threshold(
        np.concatenate((evaluation_priority, np.asarray([100.0]))), fitted
    )
    np.testing.assert_array_equal(original, np.asarray([True, False, False]))
    np.testing.assert_array_equal(extended[: len(original)], original)


def test_online_query_threshold_edge_targets_are_never_and_always_query():
    calibration_priority = np.asarray([-2.0, -1.0, 1.0, 2.0], dtype=np.float64)
    shifted_evaluation_priority = np.asarray(
        [-1.0e300, -3.0, 0.0, 3.0, 1.0e300], dtype=np.float64
    )

    never = study._fit_online_query_threshold(calibration_priority, 0.0)
    always = study._fit_online_query_threshold(calibration_priority, 1.0)

    assert never["calibration_query_count"] == 0
    assert always["calibration_query_count"] == len(calibration_priority)
    assert not np.any(
        study._apply_online_query_threshold(shifted_evaluation_priority, never)
    )
    assert np.all(
        study._apply_online_query_threshold(shifted_evaluation_priority, always)
    )


def test_fixed_threshold_bootstrap_never_reselects_queries():
    decisions = study._DecisionSet(
        labels=np.asarray([0, 1, 0]),
        base_loss=np.asarray([1.0, 1.0, 1.0]),
        base_confidence=np.ones(3),
        candidate_loss=np.asarray([[0.0], [0.0], [0.0]]),
        candidate_confidence=np.ones((3, 1)),
        value=np.ones((3, 1)),
        router_features=np.zeros((3, 1, 1)),
        context_index=np.zeros(3, dtype=np.int64),
        candidate_modality=np.zeros((3, 1), dtype=np.int64),
        groups=np.asarray(["a", "b", "b"]),
    )
    # Only the first decision is queried. Resampling group b must not transfer
    # its unused quota to either unqueried decision.
    draws = study._fixed_threshold_cluster_bootstrap(
        decisions,
        np.asarray([True, False, False]),
        np.zeros(3, dtype=np.int64),
        repetitions=40,
        bootstrap_seed=29,
    )

    assert set(draws).issubset({0.0, 1.0 / 3.0, 1.0})
    assert 0.0 in draws
    assert 1.0 in draws


def test_online_policy_arrays_never_rank_the_evaluation_cohort(monkeypatch):
    decisions = study._DecisionSet(
        labels=np.asarray([0, 1, 0]),
        base_loss=np.asarray([1.0, 0.0, 1.0]),
        base_confidence=np.ones(3),
        candidate_loss=np.asarray([[0.0], [1.0], [0.0]]),
        candidate_confidence=np.ones((3, 1)),
        value=np.asarray([[1.0], [-1.0], [1.0]]),
        router_features=np.zeros((3, 1, 1)),
        context_index=np.zeros(3, dtype=np.int64),
        candidate_modality=np.zeros((3, 1), dtype=np.int64),
        groups=np.asarray(["a", "b", "c"]),
    )
    monkeypatch.setattr(
        study,
        "_policy_priority_and_choice",
        lambda *_: (np.asarray([0.9, 0.5, 0.1]), np.zeros(3, dtype=np.int64)),
    )
    monkeypatch.setattr(
        study,
        "_calibrated_confidence",
        lambda *_: (np.ones(3), np.ones((3, 1))),
    )

    def reject_ranking(*_args, **_kwargs):
        raise AssertionError("evaluation-batch ranking is forbidden for the online policy")

    monkeypatch.setattr(study, "_seeded_descending_order", reject_ranking)
    query, choice, _, _ = study._policy_arrays(
        decisions,
        {
            "online_query_thresholds": {
                "0.500000": {"threshold": 0.5, "query_on_equal": True}
            }
        },
        "online_selected_router",
        0.5,
        11,
    )

    np.testing.assert_array_equal(query, np.asarray([True, True, False]))
    np.testing.assert_array_equal(choice, np.zeros(3, dtype=np.int64))


def test_router_features_allow_candidate_preference_to_flip_with_observations():
    z = np.asarray([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]], dtype=np.float64)
    observed = np.asarray([True, False, False])
    base_score = np.zeros(2, dtype=np.float64)
    candidate_one = study._router_features(z, observed, 1, base_score)
    candidate_two = study._router_features(z, observed, 2, base_score)

    observable_width = 10
    interaction_start = observable_width + 3
    weights = np.zeros(candidate_one.shape[1], dtype=np.float64)
    # Modality 0 is the first observable feature. Opposite candidate-specific
    # slopes make the preferred missing modality depend on the current sample.
    weights[interaction_start + observable_width] = 1.0
    weights[interaction_start + 2 * observable_width] = -1.0
    prediction_one = candidate_one @ weights
    prediction_two = candidate_two @ weights

    assert prediction_one[0] > prediction_two[0]
    assert prediction_one[1] < prediction_two[1]


def test_risk_coverage_auc_is_invariant_to_tied_confidence_order():
    losses = np.asarray([1.0, 0.0, 0.0, 1.0])
    confidence = np.ones(4)
    assert study._risk_coverage_auc(losses, confidence) == 0.5
    assert study._risk_coverage_auc(losses[::-1], confidence) == 0.5


def test_full_coverage_endpoint_accepts_every_evaluation_decision(tmp_path):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    result = run_value_study(cache, _config(), mode="singleton", seed=11)
    for policy, by_budget in result["evaluation"]["selective_curves"].items():
        for budget, by_coverage in by_budget.items():
            full = by_coverage["1.000000"]
            assert full["coverage"] == 1.0, (policy, budget)
            assert full["abstention_rate"] == 0.0, (policy, budget)
            assert full["selective_risk"] == result["evaluation"]["policies"][policy][budget]["final_error"]


def test_exact_matched_coverage_contrast_is_null_for_identical_tied_policies():
    state = {
        "evaluation_confidence": np.ones(20, dtype=np.float64),
        "evaluation_loss": np.asarray([0.0, 1.0] * 10),
        "query_priority": np.arange(20, dtype=np.float64),
        "query_order_seed": 17,
        "query_budget": 0.5,
        "base_loss": np.asarray([0.0, 1.0] * 10),
        "queried_loss": np.asarray([0.0, 1.0] * 10),
        "base_confidence": np.ones(20, dtype=np.float64),
        "queried_confidence": np.ones(20, dtype=np.float64),
    }
    groups = np.asarray([f"video-{index // 2}" for index in range(20)])
    contrast = study._paired_selective_risk_contrast(
        state,
        state,
        groups,
        repetitions=40,
        target_coverage=0.5,
        tie_seed=101,
        bootstrap_seed=202,
    )
    assert contrast["accepted_count"] == 10
    assert contrast["left_coverage"] == contrast["right_coverage"] == 0.5
    assert contrast["selective_risk_difference"] == 0.0
    assert contrast["cluster_ci95"] == [0.0, 0.0]


def test_nested_selective_bootstrap_reselects_queries_after_group_duplication():
    state = {
        "query_priority": np.asarray([3.0, 2.0, 1.0]),
        "query_order_seed": 19,
        "query_budget": 0.5,
        "base_loss": np.asarray([1.0, 0.0, 0.0]),
        "queried_loss": np.asarray([0.0, 1.0, 1.0]),
        "base_confidence": np.ones(3),
        "queried_confidence": np.ones(3),
    }
    weights = np.asarray([[1, 1, 1], [2, 1, 0]], dtype=np.int64)
    draws = study._nested_selective_risk_draws(
        state, weights, target_coverage=1.0, tie_seed=23
    )

    assert draws[0] == pytest.approx(1.0 / 3.0)
    assert draws[1] == 0.0


def test_nested_selective_bootstrap_preserves_null_under_different_query_orders():
    count = 12
    loss = np.asarray([0.0, 1.0] * 6)
    common = {
        "evaluation_confidence": np.ones(count),
        "evaluation_loss": loss,
        "query_order_seed": 17,
        "query_budget": 0.5,
        "base_loss": loss,
        "queried_loss": loss,
        "base_confidence": np.ones(count),
        "queried_confidence": np.ones(count),
    }
    left = {**common, "query_priority": np.arange(count, dtype=np.float64)}
    right = {**common, "query_priority": np.arange(count, dtype=np.float64)[::-1]}
    groups = np.asarray([f"group-{index // 2}" for index in range(count)])

    contrast = study._paired_selective_risk_contrast(
        left,
        right,
        groups,
        repetitions=200,
        target_coverage=0.5,
        tie_seed=101,
        bootstrap_seed=202,
    )

    assert contrast["selective_risk_difference"] == 0.0
    assert contrast["cluster_ci95"] == [0.0, 0.0]
    assert set(contrast["bootstrap_draws"]) == {0.0}


def test_weighted_topk_matches_explicit_cluster_expansion():
    weights = np.asarray([[2, 0, 1, 3], [0, 3, 2, 1]], dtype=np.int64)
    order = np.asarray([2, 0, 3, 1], dtype=np.int64)
    values = np.asarray([0.5, -1.0, 1.0, -0.25], dtype=np.float64)
    fractions = (0.0, 0.5, 1.0)
    sums, counts = study._weighted_topk_sums(weights, order, values, fractions)

    for row, row_weights in enumerate(weights):
        expanded = np.concatenate(
            [np.repeat(index, row_weights[index]) for index in order]
        )
        for column, fraction in enumerate(fractions):
            expected_count = int(np.rint(fraction * len(expanded)))
            assert counts[row, column] == expected_count
            assert sums[row, column] == pytest.approx(values[expanded[:expected_count]].sum())


def test_policy_bootstraps_reuse_group_draws_across_policies(monkeypatch):
    decisions = study._DecisionSet(
        labels=np.asarray([0, 1, 0, 1]),
        base_loss=np.asarray([1.0, 0.0, 1.0, 0.0]),
        base_confidence=np.ones(4),
        candidate_loss=np.asarray([[0.0], [1.0], [0.0], [1.0]]),
        candidate_confidence=np.ones((4, 1)),
        value=np.asarray([[1.0], [-1.0], [1.0], [-1.0]]),
        router_features=np.zeros((4, 1, 1)),
        context_index=np.zeros(4, dtype=np.int64),
        candidate_modality=np.zeros((4, 1), dtype=np.int64),
        groups=np.asarray(["a", "a", "b", "c"]),
    )
    captured: list[np.ndarray] = []
    original = study._weighted_topk_sums

    def capture(weights, order, values, fractions):
        captured.append(np.asarray(weights).copy())
        return original(weights, order, values, fractions)

    def priority(_decisions, _router, policy, _seed):
        values = np.asarray([4.0, 3.0, 2.0, 1.0])
        if policy == "linear_value":
            values = values[::-1]
        return values, np.zeros(4, dtype=np.int64)

    monkeypatch.setattr(study, "_weighted_topk_sums", capture)
    monkeypatch.setattr(study, "_policy_priority_and_choice", priority)
    first = study._policy_reselected_cluster_bootstrap(
        decisions, {}, "oracle", (0.0, 0.5, 1.0), 7, 11, 29
    )
    second = study._policy_reselected_cluster_bootstrap(
        decisions, {}, "linear_value", (0.0, 0.5, 1.0), 7, 17, 29
    )
    assert first.shape == second.shape == (7, 3)
    np.testing.assert_array_equal(captured[0], captured[1])


def test_hybrid_policies_isolate_query_timing_and_candidate_choice(monkeypatch):
    decisions = study._DecisionSet(
        labels=np.asarray([0, 1]),
        base_loss=np.zeros(2),
        base_confidence=np.ones(2),
        candidate_loss=np.zeros((2, 2)),
        candidate_confidence=np.ones((2, 2)),
        value=np.zeros((2, 2)),
        router_features=np.zeros((2, 2, 1)),
        context_index=np.zeros(2, dtype=np.int64),
        candidate_modality=np.asarray([[1, 2], [1, 2]]),
        groups=np.asarray(["a", "b"]),
    )
    learned_value = np.asarray([[3.0, 1.0], [1.0, 4.0]])
    source_prior = np.asarray([[0.1, 0.2], [0.1, 0.2]])
    monkeypatch.setattr(
        study,
        "_router_predictions",
        lambda *_: (learned_value, np.zeros_like(learned_value)),
    )
    monkeypatch.setattr(study, "_source_prior_arrays", lambda *_: source_prior)
    monkeypatch.setattr(
        study,
        "_calibrated_confidence",
        lambda *_: (np.asarray([0.2, 0.9]), np.ones((2, 2))),
    )

    learned_priority, prior_choice = study._policy_priority_and_choice(
        decisions, {}, "selected_router_query_source_prior_choice", 11
    )
    confidence_priority, learned_choice = study._policy_priority_and_choice(
        decisions, {}, "confidence_query_selected_router_choice", 11
    )

    np.testing.assert_array_equal(learned_priority, np.asarray([3.0, 4.0]))
    np.testing.assert_array_equal(prior_choice, np.asarray([1, 1]))
    np.testing.assert_allclose(confidence_priority, np.asarray([-0.2, -0.9]))
    np.testing.assert_array_equal(learned_choice, np.asarray([0, 1]))
    linear_query = study._policy_arrays(decisions, {}, "linear_value", 0.5, 11)[0]
    learned_timing_query = study._policy_arrays(
        decisions, {}, "selected_router_query_source_prior_choice", 0.5, 11
    )[0]
    confidence_query = study._policy_arrays(decisions, {}, "confidence", 0.5, 11)[0]
    confidence_timing_query = study._policy_arrays(
        decisions, {}, "confidence_query_selected_router_choice", 0.5, 11
    )[0]
    np.testing.assert_array_equal(linear_query, learned_timing_query)
    np.testing.assert_array_equal(confidence_query, confidence_timing_query)


def test_run_reports_exact_matched_coverage_and_reselected_bootstrap(tmp_path):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    config = _config()
    result = run_value_study(cache, config, mode="singleton", seed=37)
    expected_coverage = np.rint(0.9 * result["evaluation"]["decision_instances"]) / result["evaluation"][
        "decision_instances"
    ]
    for contrast in result["evaluation"]["paired_contrasts"].values():
        selective = contrast["coverage_0.90"]
        assert selective["left_coverage"] == selective["right_coverage"] == expected_coverage
        assert selective["coverage_definition"] == "exact_evaluation_count"
        assert len(contrast["bootstrap_draws"]) == config["bootstrap_repetitions"]
        assert len(selective["bootstrap_draws"]) == config["bootstrap_repetitions"]


def test_run_reports_a_frozen_online_query_policy_and_fixed_threshold_intervals(
    tmp_path,
):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    config = _config()
    result = run_value_study(cache, config, mode="singleton", seed=37)

    training = result["router_ladder"]["online_query_policy"]
    evaluation = result["evaluation"]["online_query_policy"]
    assert training["fit_split"] == "calibration"
    assert training["priority"] == "maximum_selected_router_predicted_value"
    assert training["selected_family"] == result["router_ladder"]["selection"][
        "selected_family"
    ]
    assert evaluation["evaluation_batch_access"] == "independent_per_example"
    assert evaluation["priority_sha256"]
    for budget in ("0.000000", "0.500000", "1.000000"):
        fitted = training["by_target_budget"][budget]
        observed = evaluation["by_target_budget"][budget]
        metrics = result["evaluation"]["policies"]["online_selected_router"][budget]
        assert observed["threshold"] == fitted["threshold"]
        assert observed["query_on_equal"] == fitted["query_on_equal"]
        assert observed["tie_convention"] == fitted["tie_convention"]
        assert observed["evaluation_realized_query_rate"] == metrics["query_rate"]
        assert observed["evaluation_query_count"] == metrics["query_count"]
        assert metrics["bootstrap_method"] == (
            "video_cluster_resample_with_fixed_calibration_threshold"
        )
        assert len(
            result["bootstrap"]["policy_error_reduction_draws"][
                "online_selected_router"
            ][budget]
        ) == config["bootstrap_repetitions"]
    assert evaluation["by_target_budget"]["0.000000"]["evaluation_query_count"] == 0
    assert evaluation["by_target_budget"]["1.000000"][
        "evaluation_query_count"
    ] == result["evaluation"]["decision_instances"]
    assert validate_value_result(result, expected_config=config) == []


def test_validator_rejects_tampered_online_threshold_provenance(tmp_path):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    config = _config()
    result = run_value_study(cache, config, mode="singleton", seed=11)

    changed = copy.deepcopy(result)
    changed["evaluation"]["online_query_policy"]["by_target_budget"]["0.500000"][
        "evaluation_realized_query_rate"
    ] += 0.1
    assert validate_value_result(changed, expected_config=config)

    changed = copy.deepcopy(result)
    changed["router_ladder"]["online_query_policy"]["by_target_budget"]["0.500000"][
        "threshold"
    ] += 0.1
    assert validate_value_result(changed, expected_config=config)

    changed = copy.deepcopy(result)
    changed["evaluation"]["online_query_policy"]["priority_sha256"] = "0" * 64
    assert validate_value_result(changed, expected_config=config)

    changed = copy.deepcopy(result)
    evaluation_policy = changed["evaluation"]["online_query_policy"]
    evaluation_policy["by_target_budget"]["0.500000"]["threshold"] = np.nextafter(
        evaluation_policy["by_target_budget"]["0.500000"]["threshold"],
        np.inf,
    )
    attested = copy.deepcopy(evaluation_policy)
    attested.pop("attestation_sha256")
    evaluation_policy["attestation_sha256"] = study._json_sha256(attested)
    assert validate_value_result(changed, expected_config=config)


def test_aggregate_reports_online_policy_thresholds_and_pooled_fixed_rule_interval(
    tmp_path,
):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    config = _config()
    results = [
        run_value_study(cache, config, mode="singleton", seed=seed)
        for seed in (11, 23)
    ]
    spec = importlib.util.spec_from_file_location(
        "aggregate_mosei_value", AGGREGATE_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    summary = module._online_query_policy_summary(
        [
            (Path(f"seed-{result['seed']}.json"), result)
            for result in results
        ],
        config["bootstrap_repetitions"],
    )

    assert summary["fit_split"] == "calibration"
    assert summary["evaluation_batch_access"] == "independent_per_example"
    assert set(summary["evaluation_attestation_sha256_by_seed"]) == {"11", "23"}
    middle = summary["by_target_budget"]["0.500000"]
    assert middle["target_query_rate"] == 0.5
    assert set(middle["fixed_threshold_by_seed"]) == {"11", "23"}
    assert middle["evaluation_realized_query_rate"]["runs"] == 2
    aligned_draws = np.asarray(
        [
            result["bootstrap"]["policy_error_reduction_draws"]
            ["online_selected_router"]["0.500000"]
            for result in results
        ]
    )
    np.testing.assert_allclose(
        middle["error_reduction_cluster_ci95"],
        np.quantile(np.mean(aligned_draws, axis=0), (0.025, 0.975)),
    )
    assert middle["bootstrap_method"] == (
        "aligned_video_cluster_resample_with_fixed_calibration_threshold_then_seed_mean"
    )


def test_result_validation_rejects_a_different_configuration(tmp_path):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    config = _config()
    result = run_value_study(cache, config, mode="singleton", seed=11)
    changed = dict(config)
    changed["ridge"] = 0.2

    errors = validate_value_result(
        result,
        expected_configuration_sha256=configuration_sha256(changed),
    )
    assert "configuration SHA256 does not match the expected value" in errors


def test_verifier_replays_cache_and_attestation_detects_later_change(tmp_path, monkeypatch):
    cache = tmp_path / "scores.npz"
    config_path = tmp_path / "config.json"
    result_path = tmp_path / "result.json"
    attestation_path = tmp_path / "result.attestation.json"
    _write_cache(cache)
    config = _config()
    config_path.write_text(json.dumps(config), encoding="utf-8")
    result = run_value_study(cache, config, mode="singleton", seed=11)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("verify_mosei_value", VERIFY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(VERIFY_SCRIPT),
            str(result_path),
            "--mode",
            "singleton",
            "--seed",
            "11",
            "--config",
            str(config_path),
            "--cache",
            str(cache),
            "--write-attestation",
            str(attestation_path),
        ],
    )
    assert module.main() == 0
    result["unvalidated_extra_field"] = "changed after replay"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(VERIFY_SCRIPT),
            str(result_path),
            "--mode",
            "singleton",
            "--seed",
            "11",
            "--config",
            str(config_path),
            "--check-attestation",
            str(attestation_path),
        ],
    )
    assert module.main() == 2


def test_aggregate_requires_the_exact_seed_set():
    spec = importlib.util.spec_from_file_location("aggregate_mosei_value", AGGREGATE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = [(Path("seed-11.json"), {"seed": 11})]

    with pytest.raises(ValueError, match="seed set mismatch"):
        module._require_seed_set(records, {11, 23}, "singleton")


def test_negative_scientific_gates_do_not_invalidate_a_structural_aggregate():
    spec = importlib.util.spec_from_file_location("aggregate_mosei_value", AGGREGATE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def record(headroom_passed):
        return {
            "headroom_gate": {"passed": headroom_passed},
            "task_head_gate": {"passed": True},
            "evaluation": {
                "policies": {
                    "selected_router": {"0.500000": {"harmful_query_rate": 0.1}},
                    "selected_baseline": {"0.500000": {"harmful_query_rate": 0.2}},
                },
                "paired_contrasts": {"0.500000": {"video_macro_difference": 0.02}},
                "contexts": {
                    name: {
                        "policy_outcomes": {
                            "selected_router": {"0.500000": {"error_reduction": 0.02}},
                            "selected_baseline": {"0.500000": {"error_reduction": 0.0}},
                        }
                    }
                    for name in ("audio", "video", "text")
                },
            },
        }

    pooled = {
        "0.500000": {
            "error_reduction_difference": 0.02,
            "error_reduction_difference_cluster_ci95": [0.01, 0.03],
        }
    }
    passed = [(Path("seed-11.json"), record(True))]
    failed = passed + [
        (Path("seed-23.json"), record(False))
    ]

    gate_config = {
        "primary_budget": 0.5,
        "minimum_primary_error_reduction_gain": 0.005,
        "minimum_video_macro_gain_fraction": 0.5,
        "minimum_nonnegative_start_contexts": 2,
        "maximum_start_context_harm": 0.005,
    }
    validation, gates = module._scientific_validation(passed, pooled, gate_config)
    assert validation == {"ok": True, "errors": []}
    assert gates["selected_router_selection"]["passed"]
    validation, failed_gates = module._scientific_validation(failed, pooled, gate_config)
    assert validation == {"ok": True, "errors": []}
    assert not failed_gates["singleton_oracle_headroom"]["passed"]


def test_aggregate_records_a_negative_selected_router_result_without_invalidating_it():
    spec = importlib.util.spec_from_file_location("aggregate_mosei_value", AGGREGATE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = {
        "headroom_gate": {"passed": True},
        "task_head_gate": {"passed": True},
        "evaluation": {
            "policies": {
                "selected_router": {"0.500000": {"harmful_query_rate": 0.2}},
                "selected_baseline": {"0.500000": {"harmful_query_rate": 0.1}},
            },
            "paired_contrasts": {"0.500000": {"video_macro_difference": 0.01}},
            "contexts": {
                name: {
                    "policy_outcomes": {
                        "selected_router": {"0.500000": {"error_reduction": 0.01}},
                        "selected_baseline": {"0.500000": {"error_reduction": 0.0}},
                    }
                }
                for name in ("audio", "video", "text")
            },
        },
    }
    pooled = {
        "0.500000": {
            "error_reduction_difference": 0.01,
            "error_reduction_difference_cluster_ci95": [-0.01, 0.03],
        }
    }

    validation, gates = module._scientific_validation(
        [(Path("seed-11.json"), record)],
        pooled,
        {
            "primary_budget": 0.5,
            "minimum_primary_error_reduction_gain": 0.005,
            "minimum_video_macro_gain_fraction": 0.5,
            "minimum_nonnegative_start_contexts": 2,
            "maximum_start_context_harm": 0.005,
        },
    )
    assert validation == {"ok": True, "errors": []}
    assert not gates["selected_router_selection"]["passed"]


@pytest.mark.parametrize("seed", [8, 9, 15, 23, 25, 26])
def test_paired_contrast_reuses_the_saved_policy_decisions(tmp_path, seed):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    result = run_value_study(cache, _config(), mode="singleton", seed=seed)
    assert validate_value_result(result) == []
    for budget, selection in result["evaluation"]["baseline_selection"].items():
        chosen = selection["policy"]
        selected_metrics = result["evaluation"]["policies"]["selected_baseline"][budget]
        chosen_metrics = result["evaluation"]["policies"][chosen][budget]
        for metric in ("final_error", "error_reduction", "query_rate"):
            assert selected_metrics[metric] == chosen_metrics[metric]


def test_validator_fails_closed_on_missing_or_tampered_scientific_fields(tmp_path):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    config = _config()
    result = run_value_study(cache, config, mode="singleton", seed=11)
    assert validate_value_result(result, expected_config=config) == []

    for field in ("selective_curves", "contexts"):
        changed = copy.deepcopy(result)
        del changed["evaluation"][field]
        assert validate_value_result(changed, expected_config=config)
    changed = copy.deepcopy(result)
    del changed["headroom_gate"]
    assert validate_value_result(changed, expected_config=config)
    changed = copy.deepcopy(result)
    changed["evaluation"]["policies"]["linear_value"]["0.500000"]["final_error"] = 99.0
    assert validate_value_result(changed, expected_config=config)
    changed = copy.deepcopy(result)
    changed["evaluation"]["policies"]["linear_value"]["0.500000"]["risk_coverage_auc"] = -123.0
    assert validate_value_result(changed, expected_config=config)
    changed = copy.deepcopy(result)
    changed["group_partitions"]["task_fit"]["utterances"] += 1
    assert validate_value_result(changed, expected_config=config)
    changed = copy.deepcopy(result)
    changed["group_partitions"]["task_fit"]["group_hashes"][0] = "f" * 64
    assert validate_value_result(changed, expected_config=config)
    changed = copy.deepcopy(result)
    changed["task_head_gate"]["validation_accuracy_by_mask"]["audio"] = 99.0
    assert validate_value_result(changed, expected_config=config)
    changed = copy.deepcopy(result)
    changed["evaluation"]["contexts"]["audio"]["policy_outcomes"]["linear_value"][
        "0.500000"
    ]["query_target_outcomes"]["video"]["mean_realized_value"] += 0.1
    assert validate_value_result(changed, expected_config=config)
    changed = copy.deepcopy(result)
    changed["task_head_gate"]["validation_accuracy_by_mask"]["audio"] = 0.5
    assert validate_value_result(changed, expected_config=config)
    changed = copy.deepcopy(result)
    changed["evaluation"]["paired_contrasts"]["0.500000"]["video_macro_difference"] += 0.1
    assert validate_value_result(changed, expected_config=config)
    changed = copy.deepcopy(result)
    selected_family = changed["router_ladder"]["selection"]["selected_family"]
    for policy in (selected_family, "selected_router"):
        changed["evaluation"]["policies"][policy]["0.500000"][
            "harmful_query_rate"
        ] = 0.999
    errors = validate_value_result(changed, expected_config=config)
    assert any("context outcomes do not reconstruct" in error for error in errors)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("contexts", "audio", "policy_outcomes", "linear_value", "0.500000", "error_reduction"), 999.0),
        (("contexts", "audio", "policy_outcomes", "linear_value", "0.500000", "harmful_query_rate"), 999.0),
        (
            ("contexts", "audio", "policy_outcomes", "linear_value", "0.500000", "query_target_counts", "video"),
            999,
        ),
        (("paired_contrasts", "0.500000", "coverage_0.90", "selective_risk_difference"), 999.0),
        (
            (
                "paired_contrasts",
                "0.500000",
                "coverage_0.90",
                "video_group_statistics",
                0,
                "left_accepted_error_sum",
            ),
            999.0,
        ),
        (("contexts", "audio", "candidates", "video", "mean_realized_value"), 999.0),
        (("contexts", "audio", "base_error"), 0.123456),
        (("policies", "linear_value", "0.500000", "error_reduction_video_macro"), 999.0),
        (("selective_curves", "linear_value", "1.000000", "selective_risk"), 0.123456),
        (("paired_contrasts", "0.500000", "left_policy"), "oracle"),
        (("paired_contrasts", "0.500000", "video_macro_difference"), 999.0),
    ],
)
def test_validator_rejects_tampered_context_and_selective_statistics(tmp_path, path, value):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    config = _config()
    result = run_value_study(cache, config, mode="singleton", seed=11)
    assert validate_value_result(result, expected_config=config) == []
    changed = copy.deepcopy(result)
    target = changed["evaluation"]
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    assert validate_value_result(changed, expected_config=config)


def test_validator_rejects_pair_only_candidate_ranking_fields(tmp_path):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    config = _config()
    result = run_value_study(cache, config, mode="pair", seed=11)
    assert validate_value_result(result, expected_config=config) == []
    for field in ("candidate_value_tie_rate", "queried_candidate_selection_accuracy"):
        changed = copy.deepcopy(result)
        changed["evaluation"]["policies"]["linear_value"]["0.500000"][field] = 0.5
        assert validate_value_result(changed, expected_config=config)


def test_validator_rejects_tampered_bootstrap_interval_and_implementation_digest(tmp_path):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    config = _config()
    result = run_value_study(cache, config, mode="singleton", seed=11)
    assert validate_value_result(result, expected_config=config) == []

    changed = copy.deepcopy(result)
    changed["evaluation"]["policies"]["oracle"]["0.500000"]["error_reduction_cluster_ci95"] = [
        0.123456,
        0.234567,
    ]
    assert validate_value_result(changed, expected_config=config)
    changed = copy.deepcopy(result)
    changed["implementation_sha256"] = "0" * 64
    assert validate_value_result(changed, expected_config=config)


@pytest.mark.parametrize(
    "path",
    [
        ("policies", "selected_baseline", "0.500000", "risk_coverage_auc"),
        ("policies", "selected_baseline", "0.500000", "final_macro_f1"),
        ("selective_curves", "selected_baseline", "0.500000", "0.900000", "selective_risk"),
        ("contexts", "audio", "policy_outcomes", "selected_baseline", "0.500000", "base_macro_f1"),
    ],
)
def test_validator_requires_selected_baseline_to_be_an_exact_alias(tmp_path, path):
    cache = tmp_path / "scores.npz"
    _write_cache(cache)
    config = _config()
    result = run_value_study(cache, config, mode="singleton", seed=11)
    assert validate_value_result(result, expected_config=config) == []
    changed = copy.deepcopy(result)
    target = changed["evaluation"]
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = 0.123456
    assert validate_value_result(changed, expected_config=config)
