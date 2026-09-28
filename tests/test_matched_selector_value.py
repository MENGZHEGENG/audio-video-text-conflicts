from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from conflictbench.matched_selector_value import (
    SelectorRunConfig,
    _pair_features,
    _sum_rule,
    _top_k,
    aggregate_selector_runs,
    run_selector_value,
)


def test_shared_sum_rule_allows_the_requested_score_to_change_the_action() -> None:
    scores = np.asarray([[1.0, 1.0, -3.0], [1.0, 1.0, -1.0]])
    before = _sum_rule(scores[:, :2], threshold=0.35)
    after = _sum_rule(scores, threshold=0.35)
    assert before.tolist() == [1, 1]
    assert after.tolist() == [0, 1]


def test_pair_features_do_not_depend_on_the_unavailable_score() -> None:
    pair = np.asarray([[0.3, -0.2], [1.0, 1.0]])
    full = np.column_stack((pair, [99.0, -99.0]))
    assert _pair_features(pair).shape == (2, 7)
    np.testing.assert_array_equal(_pair_features(pair), _pair_features(full[:, :2]))


@pytest.mark.parametrize("budget,count", [(0.0, 0), (0.1, 2), (0.25, 5), (1.0, 20)])
def test_top_k_uses_exact_budget_and_stable_tie_breaks(budget: float, count: int) -> None:
    scores = np.ones(20)
    first = _top_k(scores, budget)
    second = _top_k(scores, budget)
    assert int(first.sum()) == count
    np.testing.assert_array_equal(first, second)


def test_run_uses_the_fixed_training_and_test_mechanism_contract() -> None:
    result = run_selector_value(
        SelectorRunConfig(seed=11, train_size=200, eval_size=200, budgets=(0.1, 0.5))
    )
    assert result["schema"] == "conflictbench.matched-selector-value.v1"
    assert result["contract"]["train_mechanisms"] == ["clean", "invert", "swap", "dropout"]
    assert result["contract"]["evaluation_mechanisms"] == ["clean", "mixed", "burst", "ambiguity"]
    for budget in ("0.100", "0.500"):
        counts = {
            metrics["query_count"]
            for name, metrics in result["results"][budget].items()
            if name != "no_query"
        }
        assert len(counts) == 1


def test_aggregate_fails_closed_on_missing_or_duplicate_seeds() -> None:
    record = run_selector_value(
        SelectorRunConfig(seed=11, train_size=200, eval_size=200, budgets=(0.1,))
    )
    with pytest.raises(ValueError, match="coverage mismatch"):
        aggregate_selector_runs([record], expected_seeds=[11, 23], bootstrap_samples=10)
    with pytest.raises(ValueError, match="duplicate seed"):
        aggregate_selector_runs([record, record], expected_seeds=[11], bootstrap_samples=10)


def test_aggregate_accepts_seed_specific_random_streams_and_checks_exact_budgets() -> None:
    records = [
        run_selector_value(
            SelectorRunConfig(seed=seed, train_size=200, eval_size=200, budgets=(0.1,))
        )
        for seed in (11, 23)
    ]
    aggregate = aggregate_selector_runs(
        records, expected_seeds=(11, 23), bootstrap_seed=17, bootstrap_samples=20
    )
    assert aggregate["seed_count"] == 2
    assert aggregate["schema"] == "conflictbench.matched-selector-value.aggregate.v2"
    assert aggregate["bootstrap"] == {
        "method": "paired_seed_percentile",
        "samples": 20,
        "seed": 17,
    }
    for method in ("train_only_value", "pair_uncertainty", "matched_random", "oracle_upper_bound"):
        assert aggregate["summary"]["0.100"][method]["query_rate"]["mean"] == pytest.approx(0.1)
    comparisons = aggregate["summary"]["0.100"]["paired_comparisons"]
    assert comparisons["train_only_value"]["no_query"]["utility"]["paired_mean_difference"] == pytest.approx(
        aggregate["summary"]["0.100"]["train_only_value"]["utility"]["mean"]
        - aggregate["summary"]["0.100"]["no_query"]["utility"]["mean"]
    )
    assert comparisons["train_only_value"]["pair_uncertainty"]["utility"][
        "paired_seed_percentile_95_ci"
    ][0] <= comparisons["train_only_value"]["pair_uncertainty"]["utility"][
        "paired_seed_percentile_95_ci"
    ][1]


@pytest.mark.parametrize("mutation", ["extra_outcome", "missing_outcome", "extra_method", "wrong_count"])
def test_aggregate_rejects_unreviewed_raw_schema_and_counts(mutation: str) -> None:
    record = run_selector_value(
        SelectorRunConfig(seed=11, train_size=200, eval_size=200, budgets=(0.1,))
    )
    changed = deepcopy(record)
    cell = changed["results"]["0.100"]["train_only_value"]
    if mutation == "extra_outcome":
        cell["new_diagnostic"] = 0.5
    elif mutation == "missing_outcome":
        del cell["queried_corrected_error_rate"]
    elif mutation == "extra_method":
        changed["results"]["0.100"]["new_selector"] = deepcopy(cell)
    else:
        cell["n"] -= 1
    with pytest.raises(ValueError, match="schema changed|count changed"):
        aggregate_selector_runs([changed], expected_seeds=[11], bootstrap_samples=10)
