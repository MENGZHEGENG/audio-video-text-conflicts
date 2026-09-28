"""Matched-budget selector test with one shared pair/full terminal rule.

This experiment asks whether a pre-request score from the observed pair can
rank examples for which acquiring the third score changes reward. Labels and
the unavailable score are used only to construct the training target or the
explicit oracle reference; inference-time selectors receive the initial pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .core import Dataset, generate_dataset


SCHEMA = "conflictbench.matched-selector-value.v1"
DEFAULT_BUDGETS = (0.10, 0.25, 0.50, 0.637)
DEFAULT_TRAIN_MECHANISMS = ("clean", "invert", "swap", "dropout")
DEFAULT_TEST_MECHANISMS = ("clean", "mixed", "burst", "ambiguity")


def _sum_rule(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Answer by the sign of the observed-score sum; abstain near zero."""

    if scores.ndim != 2 or scores.shape[1] not in (2, 3):
        raise ValueError("scores must contain two or three channels")
    total = scores.sum(axis=1)
    action = np.full(len(total), 2, dtype=np.int64)
    confident = np.abs(total) >= threshold
    action[confident] = (total[confident] >= 0.0).astype(np.int64)
    return action


def _reward(action: np.ndarray, data: Dataset, abstain_cost: float) -> np.ndarray:
    target = np.where(data.ambiguous, 2, data.y)
    value = np.where(action == target, 1.0, -1.0).astype(np.float64)
    value -= abstain_cost * ((action == 2) & ~data.ambiguous)
    return value


def _pair_features(scores: np.ndarray) -> np.ndarray:
    """Features available before the third source is requested."""

    if scores.ndim != 2 or scores.shape[1] != 2:
        raise ValueError("selector features require exactly the observed pair")
    first, second = scores[:, 0], scores[:, 1]
    total = first + second
    return np.column_stack(
        (first, second, np.abs(first), np.abs(second), total, np.abs(total), first * second)
    ).astype(np.float64, copy=False)


def _fit_value_ridge(features: np.ndarray, gain: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a standardized ridge regression using only the declared fit split."""

    if features.ndim != 2 or gain.shape != (len(features),):
        raise ValueError("training feature and target shapes do not match")
    if alpha <= 0 or not np.isfinite(features).all() or not np.isfinite(gain).all():
        raise ValueError("invalid ridge inputs")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (features - mean) / scale
    design = np.column_stack((np.ones(len(features)), standardized))
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ gain)
    return coefficients, mean, scale


def _predict_value(
    features: np.ndarray, fitted: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    coefficients, mean, scale = fitted
    standardized = (features - mean) / scale
    return np.column_stack((np.ones(len(features)), standardized)) @ coefficients


def _top_k(scores: np.ndarray, budget: float) -> np.ndarray:
    if not 0.0 <= budget <= 1.0:
        raise ValueError("budget must lie in [0, 1]")
    n = len(scores)
    count = int(round(n * budget))
    if count == 0:
        return np.zeros(n, dtype=bool)
    if count == n:
        return np.ones(n, dtype=bool)
    order = np.argsort(-scores, kind="mergesort")
    mask = np.zeros(n, dtype=bool)
    mask[order[:count]] = True
    return mask


def _evaluate_selector(
    data: Dataset,
    initial_action: np.ndarray,
    queried_action: np.ndarray,
    query_mask: np.ndarray,
    *,
    query_cost: float,
    abstain_cost: float,
) -> dict[str, Any]:
    action = initial_action.copy()
    action[query_mask] = queried_action[query_mask]
    target = np.where(data.ambiguous, 2, data.y)
    correct = action == target
    nonambiguous = ~data.ambiguous
    utility = _reward(action, data, abstain_cost) - query_cost * query_mask
    requested = int(query_mask.sum())
    changed = query_mask & (initial_action != queried_action)
    pre_cost_gain = _reward(queried_action, data, abstain_cost) - _reward(
        initial_action, data, abstain_cost
    )
    return {
        "n": int(len(action)),
        "query_count": requested,
        "query_rate": float(query_mask.mean()),
        "targeted_decision_accuracy": float(correct.mean()),
        "nonambiguous_binary_accuracy": float(correct[nonambiguous].mean()),
        "coverage": float((action != 2).mean()),
        "abstention_rate": float((action == 2).mean()),
        "utility": float(utility.mean()),
        "queried_action_change_rate": float(changed.sum() / requested) if requested else None,
        "queried_pre_cost_reward_gain": float(pre_cost_gain[query_mask].mean()) if requested else None,
        "queried_corrected_error_rate": (
            float(np.mean((initial_action[query_mask] != target[query_mask]) & (queried_action[query_mask] == target[query_mask])))
            if requested
            else None
        ),
    }


@dataclass(frozen=True)
class SelectorRunConfig:
    seed: int
    train_size: int = 6000
    eval_size: int = 4000
    strength: float = 1.0
    noise: float = 0.22
    threshold: float = 0.35
    query_cost: float = 0.10
    abstain_cost: float = 0.20
    budgets: tuple[float, ...] = DEFAULT_BUDGETS
    ridge_alpha: float = 1.0


def run_selector_value(config: SelectorRunConfig) -> dict[str, Any]:
    """Evaluate fixed-budget value, uncertainty, random, and oracle selectors."""

    if config.seed < 0 or config.train_size <= 0 or config.eval_size <= 0:
        raise ValueError("seed and sample sizes must be valid")
    if len(set(config.budgets)) != len(config.budgets):
        raise ValueError("budgets must be distinct")
    if any(not 0.0 <= budget <= 1.0 for budget in config.budgets):
        raise ValueError("budgets must lie in [0, 1]")

    train = generate_dataset(
        config.seed,
        config.train_size,
        DEFAULT_TRAIN_MECHANISMS,
        strength=config.strength,
        noise=config.noise,
    )
    test = generate_dataset(
        config.seed + 200_003,
        config.eval_size,
        DEFAULT_TEST_MECHANISMS,
        strength=config.strength,
        noise=config.noise,
    )

    pair_train = _sum_rule(train.x[:, :2], config.threshold)
    full_train = _sum_rule(train.x, config.threshold)
    gain_train = _reward(full_train, train, config.abstain_cost) - _reward(
        pair_train, train, config.abstain_cost
    )
    fitted_value = _fit_value_ridge(
        _pair_features(train.x[:, :2]), gain_train, alpha=config.ridge_alpha
    )

    pair_test = _sum_rule(test.x[:, :2], config.threshold)
    full_test = _sum_rule(test.x, config.threshold)
    observed = test.x[:, :2]
    value_scores = _predict_value(_pair_features(observed), fitted_value)
    uncertainty_scores = -np.abs(observed.sum(axis=1))
    random_scores = np.random.default_rng(config.seed + 500_003).random(len(test.y))
    oracle_scores = _reward(full_test, test, config.abstain_cost) - _reward(
        pair_test, test, config.abstain_cost
    )

    selectors = {
        "train_only_value": value_scores,
        "pair_uncertainty": uncertainty_scores,
        "matched_random": random_scores,
        "oracle_upper_bound": oracle_scores,
    }
    results: dict[str, dict[str, Any]] = {}
    for budget in config.budgets:
        budget_key = format(budget, ".3f")
        results[budget_key] = {}
        for name, score in selectors.items():
            mask = _top_k(score, budget)
            results[budget_key][name] = _evaluate_selector(
                test,
                pair_test,
                full_test,
                mask,
                query_cost=config.query_cost,
                abstain_cost=config.abstain_cost,
            )
        results[budget_key]["no_query"] = _evaluate_selector(
            test,
            pair_test,
            full_test,
            np.zeros(len(test.y), dtype=bool),
            query_cost=config.query_cost,
            abstain_cost=config.abstain_cost,
        )

    return {
        "schema": SCHEMA,
        "seed": config.seed,
        "contract": {
            "train_size": config.train_size,
            "eval_size": config.eval_size,
            "train_mechanisms": list(DEFAULT_TRAIN_MECHANISMS),
            "evaluation_mechanisms": list(DEFAULT_TEST_MECHANISMS),
            "strength": config.strength,
            "noise": config.noise,
            "score_threshold": config.threshold,
            "query_cost": config.query_cost,
            "abstain_cost": config.abstain_cost,
            "budgets": list(config.budgets),
            "pair_features": [
                "audio_score",
                "video_score",
                "absolute_audio_score",
                "absolute_video_score",
                "pair_sum",
                "absolute_pair_sum",
                "audio_video_product",
            ],
            "selector_model": "standardized_ridge_regression",
            "ridge_alpha": config.ridge_alpha,
            "training_target": "per_example_reward_after_full_score_sum_rule_minus_pair_sum_rule_before_query_cost",
            "terminal_rule": "answer_sign_of_observed_score_sum_if_absolute_sum_at_least_threshold_else_abstain",
            "budget_protocol": "exact_top_k_per_evaluation_batch_using_selector_scores_only",
            "random_seed": config.seed + 500_003,
        },
        "results": results,
    }


def load_seeds(seed_plan: dict[str, Any]) -> tuple[int, ...]:
    primary = seed_plan.get("primary_seeds")
    replication = seed_plan.get("replication_seeds")
    if not isinstance(primary, list) or not isinstance(replication, list):
        raise ValueError("seed plan must include primary_seeds and replication_seeds")
    seeds = tuple(primary + replication)
    if not seeds or any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("seed plan contains invalid seed values")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed plan contains duplicates")
    return seeds


def aggregate_selector_runs(
    records: Iterable[dict[str, Any]], *, expected_seeds: Iterable[int], bootstrap_seed: int = 92309,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    """Validate a complete seed set and summarize paired selector outcomes."""

    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    rows = list(records)
    expected = tuple(expected_seeds)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected seeds must be nonempty and unique")
    by_seed: dict[int, dict[str, Any]] = {}
    for record in rows:
        if record.get("schema") != SCHEMA or not isinstance(record.get("seed"), int):
            raise ValueError("record has an invalid schema or seed")
        if record["seed"] in by_seed:
            raise ValueError(f"duplicate seed {record['seed']}")
        by_seed[record["seed"]] = record
    missing = sorted(set(expected) - set(by_seed))
    extra = sorted(set(by_seed) - set(expected))
    if missing or extra:
        raise ValueError(f"seed coverage mismatch; missing={missing}, extra={extra}")
    ordered = [by_seed[seed] for seed in expected]
    budget_keys = list(ordered[0]["results"])
    normalized_contracts = []
    for record in ordered:
        contract = dict(record["contract"])
        random_seed = contract.pop("random_seed", None)
        if random_seed != record["seed"] + 500_003:
            raise ValueError(f"invalid random-selector seed for seed {record['seed']}")
        normalized_contracts.append(contract)
    contracts = {str(contract) for contract in normalized_contracts}
    if len(contracts) != 1 or any(list(record["results"]) != budget_keys for record in ordered):
        raise ValueError("records have inconsistent experiment contracts")

    metric_names = (
        "targeted_decision_accuracy",
        "nonambiguous_binary_accuracy",
        "utility",
        "coverage",
        "query_rate",
        "abstention_rate",
        "queried_action_change_rate",
        "queried_pre_cost_reward_gain",
        "queried_corrected_error_rate",
    )
    # Reject unreviewed evaluator outputs at their first aggregation boundary.
    # The two count fields are checked exactly; every other outcome is retained.
    expected_methods = {
        "train_only_value", "pair_uncertainty", "matched_random",
        "oracle_upper_bound", "no_query",
    }
    expected_fields = set(metric_names) | {"n", "query_count"}
    for record in ordered:
        for budget_key in budget_keys:
            cells = record["results"][budget_key]
            if set(cells) != expected_methods:
                raise ValueError(f"selector method schema changed: {budget_key}")
            expected_count = int(round(record["contract"]["eval_size"] * float(budget_key)))
            for method, cell in cells.items():
                if set(cell) != expected_fields:
                    raise ValueError(f"selector raw outcome schema changed: {budget_key}/{method}")
                if cell["n"] != record["contract"]["eval_size"]:
                    raise ValueError(f"selector evaluation count changed: {budget_key}/{method}")
                if cell["query_count"] != (0 if method == "no_query" else expected_count):
                    raise ValueError(f"selector request count changed: {budget_key}/{method}")
                if not np.isclose(cell["query_rate"], cell["query_count"] / cell["n"], rtol=0, atol=1e-12):
                    raise ValueError(f"selector query rate disagrees with count: {budget_key}/{method}")
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_indices = rng.integers(0, len(ordered), size=(bootstrap_samples, len(ordered)))
    summary: dict[str, Any] = {}
    for budget_key in budget_keys:
        budget = float(budget_key)
        methods = list(ordered[0]["results"][budget_key])
        expected_query_count = int(round(ordered[0]["contract"]["eval_size"] * budget))
        if "no_query" not in methods:
            raise ValueError("no-query reference is missing")
        for record in ordered:
            for method in ("train_only_value", "pair_uncertainty", "matched_random", "oracle_upper_bound"):
                measured = record["results"][budget_key][method]["query_count"]
                if measured != expected_query_count:
                    raise ValueError(f"{method} does not meet the exact query budget {budget_key}")
            if record["results"][budget_key]["no_query"]["query_count"] != 0:
                raise ValueError("no-query reference requested a source")
        summary[budget_key] = {}
        for method in methods:
            values_by_metric: dict[str, Any] = {}
            for metric in metric_names:
                raw = [record["results"][budget_key][method][metric] for record in ordered]
                values = np.asarray([np.nan if item is None else item for item in raw], dtype=float)
                finite = values[np.isfinite(values)]
                values_by_metric[metric] = {
                    "mean": float(finite.mean()) if finite.size else None,
                    "sample_std": float(finite.std(ddof=1)) if finite.size > 1 else None,
                    "valid_seeds": int(finite.size),
                }
            summary[budget_key][method] = values_by_metric

        comparison_references = {
            "train_only_value": ("matched_random", "no_query", "pair_uncertainty"),
            "pair_uncertainty": ("matched_random", "no_query"),
            "matched_random": ("no_query",),
            "oracle_upper_bound": ("matched_random", "no_query"),
        }
        comparisons: dict[str, Any] = {}
        for method, references in comparison_references.items():
            comparisons[method] = {}
            for reference in references:
                comparisons[method][reference] = {}
                for metric in ("targeted_decision_accuracy", "utility"):
                    delta = np.asarray(
                        [
                            record["results"][budget_key][method][metric]
                            - record["results"][budget_key][reference][metric]
                            for record in ordered
                        ],
                        dtype=float,
                    )
                    draws = delta[bootstrap_indices].mean(axis=1)
                    comparisons[method][reference][metric] = {
                        "paired_mean_difference": float(delta.mean()),
                        "paired_seed_percentile_95_ci": [
                            float(v) for v in np.quantile(draws, [0.025, 0.975])
                        ],
                    }
        summary[budget_key]["paired_comparisons"] = comparisons

    return {
        "schema": "conflictbench.matched-selector-value.aggregate.v2",
        "seed_count": len(ordered),
        "expected_seeds": list(expected),
        "bootstrap": {"method": "paired_seed_percentile", "samples": bootstrap_samples, "seed": bootstrap_seed},
        "summary": summary,
        "interpretation_limit": "oracle_upper_bound uses evaluation labels and the unavailable score; all deployable selector scores use only the initial pair, and all methods receive exact matched query budgets.",
    }
