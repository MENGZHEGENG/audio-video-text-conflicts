#!/usr/bin/env python3
"""Aggregate an exact set of verified CMU-MOSEI value-study results."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any

import numpy as np
from conflictbench.mosei_value_study import (
    configuration_sha256,
    implementation_sha256,
    validate_value_result,
)


def _digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _load(
    paths: list[str],
    mode: str,
    cache_sha256: str | None,
    configuration_digest: str,
    config: dict[str, Any],
) -> list[tuple[pathlib.Path, dict[str, Any]]]:
    records: list[tuple[pathlib.Path, dict[str, Any]]] = []
    seeds: set[int] = set()
    for raw_path in paths:
        path = pathlib.Path(raw_path).resolve()
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        errors = validate_value_result(
            record,
            expected_mode=mode,
            expected_cache_sha256=cache_sha256,
            expected_configuration_sha256=configuration_digest,
            expected_config=config,
        )
        if errors:
            raise ValueError(f"{path}: {errors}")
        seed = int(record["seed"])
        if seed in seeds:
            raise ValueError(f"duplicate {mode} seed: {seed}")
        seeds.add(seed)
        records.append((path, record))
    if not records:
        raise ValueError(f"no {mode} results were supplied")
    return sorted(records, key=lambda item: int(item[1]["seed"]))


def _require_seed_set(
    records: list[tuple[pathlib.Path, dict[str, Any]]],
    expected_seeds: set[int],
    mode: str,
) -> None:
    observed_seeds = {int(record["seed"]) for _, record in records}
    if observed_seeds != expected_seeds:
        raise ValueError(
            f"{mode} seed set mismatch: expected {sorted(expected_seeds)}, observed {sorted(observed_seeds)}"
        )


def _summary(records: list[tuple[pathlib.Path, dict[str, Any]]]) -> dict[str, Any]:
    first = records[0][1]
    policies = first["evaluation"]["policies"]
    output: dict[str, Any] = {}
    for policy, by_budget in policies.items():
        output[policy] = {}
        for budget, metrics in by_budget.items():
            output[policy][budget] = {}
            for metric in (
                "query_rate",
                "base_error",
                "final_error",
                "base_macro_f1",
                "final_macro_f1",
                "error_reduction",
                "error_reduction_video_macro",
                "harmful_query_rate",
                "useful_query_precision",
                "useful_query_recall",
                "candidate_value_tie_rate",
                "candidate_ranking_accuracy",
                "queried_candidate_selection_accuracy",
                "risk_coverage_auc",
                "oracle_regret",
            ):
                values = [record["evaluation"]["policies"][policy][budget][metric] for _, record in records]
                finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
                output[policy][budget][metric] = {
                    "runs": int(len(finite)),
                    "mean": None if not len(finite) else float(np.mean(finite)),
                    "sample_std": None if len(finite) < 2 else float(np.std(finite, ddof=1)),
                }
            selective = [
                record["evaluation"]["selective_curves"][policy][budget]["0.900000"]
                for _, record in records
            ]
            for metric in ("coverage", "abstention_rate", "selective_risk"):
                finite = np.asarray(
                    [entry[metric] for entry in selective if entry[metric] is not None],
                    dtype=np.float64,
                )
                output[policy][budget][f"coverage_0.90_{metric}"] = {
                    "runs": int(len(finite)),
                    "mean": None if not len(finite) else float(np.mean(finite)),
                    "sample_std": None if len(finite) < 2 else float(np.std(finite, ddof=1)),
                }
    return output


def _contrast_summary(records: list[tuple[pathlib.Path, dict[str, Any]]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    budgets = records[0][1]["evaluation"]["paired_contrasts"]
    for budget in budgets:
        values = np.asarray(
            [record["evaluation"]["paired_contrasts"][budget]["error_reduction_difference"] for _, record in records],
            dtype=np.float64,
        )
        risk_values = [
            record["evaluation"]["paired_contrasts"][budget].get("coverage_0.90", {}).get(
                "selective_risk_difference"
            )
            for _, record in records
        ]
        finite_risk = np.asarray([value for value in risk_values if value is not None], dtype=np.float64)
        output[budget] = {
            "runs": len(records),
            "error_reduction_difference_mean": float(np.mean(values)),
            "error_reduction_difference_sample_std": None if len(values) < 2 else float(np.std(values, ddof=1)),
            "coverage_0.90_selective_risk_difference_mean": (
                None if not len(finite_risk) else float(np.mean(finite_risk))
            ),
            "coverage_0.90_selective_risk_difference_sample_std": (
                None if len(finite_risk) < 2 else float(np.std(finite_risk, ddof=1))
            ),
        }
    return output


def _baseline_selection_summary(
    records: list[tuple[pathlib.Path, dict[str, Any]]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    budgets = records[0][1]["evaluation"]["baseline_selection"]
    for budget in budgets:
        selections = [record["evaluation"]["baseline_selection"][budget] for _, record in records]
        policy_counts = {
            policy: int(sum(selection["policy"] == policy for selection in selections))
            for policy in ("source_prior", "confidence")
        }
        output[budget] = {
            "selected_policy_counts": policy_counts,
            "selected_policy_by_seed": {
                str(record["seed"]): record["evaluation"]["baseline_selection"][budget]["policy"]
                for _, record in records
            },
            "calibration_error_by_policy": {
                policy: {
                    "mean": float(
                        np.mean([selection["candidate_errors"][policy] for selection in selections])
                    ),
                    "sample_std": (
                        None
                        if len(selections) < 2
                        else float(
                            np.std(
                                [selection["candidate_errors"][policy] for selection in selections],
                                ddof=1,
                            )
                        )
                    ),
                }
                for policy in ("source_prior", "confidence")
            },
        }
    return output


def _scalar_moments(values: list[float | int]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "runs": int(len(array)),
        "mean": float(np.mean(array)),
        "sample_std": None if len(array) < 2 else float(np.std(array, ddof=1)),
    }


def _evaluation_diagnostic_summary(
    records: list[tuple[pathlib.Path, dict[str, Any]]],
) -> dict[str, Any]:
    fields = {
        "value_regression": ("mean_squared_error", "mean_absolute_error"),
        "benefit_calibration": ("brier", "log_loss"),
        "task_confidence_calibration": ("base_brier", "candidate_brier"),
    }
    return {
        section: {
            metric: _scalar_moments(
                [record["evaluation"][section][metric] for _, record in records]
            )
            for metric in metrics
        }
        for section, metrics in fields.items()
    }


def _task_head_summary(
    records: list[tuple[pathlib.Path, dict[str, Any]]],
) -> dict[str, Any]:
    masks = records[0][1]["task_head_gate"]["validation_accuracy_by_mask"]
    return {
        "runs": len(records),
        "passed_runs": int(sum(bool(record["task_head_gate"]["passed"]) for _, record in records)),
        "all_passed": all(bool(record["task_head_gate"]["passed"]) for _, record in records),
        "validation_accuracy_by_mask": {
            mask: {
                **_scalar_moments(
                    [record["task_head_gate"]["validation_accuracy_by_mask"][mask] for _, record in records]
                ),
                "by_seed": {
                    str(record["seed"]): record["task_head_gate"]["validation_accuracy_by_mask"][mask]
                    for _, record in records
                },
            }
            for mask in masks
        },
    }


def _router_ladder_summary(
    records: list[tuple[pathlib.Path, dict[str, Any]]],
) -> dict[str, Any]:
    families = ("linear_value", "hist_gbt_value", "shallow_mlp_value")
    selected_by_seed = {
        str(record["seed"]): record["router_ladder"]["selection"]["selected_family"]
        for _, record in records
    }
    return {
        "selection_split": "calibration",
        "inner_cv": {
            key: records[0][1]["router_ladder"]["inner_cv"][key]
            for key in ("fold_count", "grouping_unit", "selection_metric", "budget")
        },
        "selected_family_counts": {
            family: int(sum(value == family for value in selected_by_seed.values()))
            for family in families
        },
        "selected_family_by_seed": selected_by_seed,
        "selected_parameters_by_seed": {
            str(record["seed"]): {
                family: record["router_ladder"]["families"][family]["selected_parameters"]
                for family in families
            }
            for _, record in records
        },
        "cv_candidates_by_seed": {
            str(record["seed"]): {
                family: [
                    {
                        "parameters": candidate["parameters"],
                        "mean_held_out_error_reduction": candidate[
                            "mean_held_out_error_reduction"
                        ],
                    }
                    for candidate in record["router_ladder"]["families"][family][
                        "cv_candidates"
                    ]
                ]
                for family in families
            }
            for _, record in records
        },
        "calibration_error_reduction_by_seed": {
            str(record["seed"]): record["router_ladder"]["selection"][
                "candidate_error_reduction"
            ]
            for _, record in records
        },
        "training_attestation_sha256_by_seed": {
            str(record["seed"]): record["router_ladder"]["training_attestation_sha256"]
            for _, record in records
        },
        "library_versions": records[0][1]["router_ladder"]["library_versions"],
    }


def _online_query_policy_summary(
    records: list[tuple[pathlib.Path, dict[str, Any]]],
    repetitions: int,
) -> dict[str, Any]:
    """Summarize calibration-fitted thresholds without evaluation reselection."""

    first_training = records[0][1]["router_ladder"]["online_query_policy"]
    first_evaluation = records[0][1]["evaluation"]["online_query_policy"]
    budgets = first_training["by_target_budget"]
    output: dict[str, Any] = {
        "protocol": first_training["protocol"],
        "fit_split": first_training["fit_split"],
        "priority": first_training["priority"],
        "candidate_choice": first_training["candidate_choice"],
        "candidate_tie_convention": first_evaluation["candidate_tie_convention"],
        "evaluation_batch_access": first_evaluation["evaluation_batch_access"],
        "selected_family_by_seed": {
            str(record["seed"]): record["router_ladder"]["online_query_policy"][
                "selected_family"
            ]
            for _, record in records
        },
        "calibration_priority_sha256_by_seed": {
            str(record["seed"]): record["router_ladder"]["online_query_policy"][
                "calibration_priority_sha256"
            ]
            for _, record in records
        },
        "evaluation_priority_sha256_by_seed": {
            str(record["seed"]): record["evaluation"]["online_query_policy"][
                "priority_sha256"
            ]
            for _, record in records
        },
        "evaluation_choice_sha256_by_seed": {
            str(record["seed"]): record["evaluation"]["online_query_policy"][
                "choice_sha256"
            ]
            for _, record in records
        },
        "evaluation_attestation_sha256_by_seed": {
            str(record["seed"]): record["evaluation"]["online_query_policy"][
                "attestation_sha256"
            ]
            for _, record in records
        },
        "by_target_budget": {},
    }
    for budget in budgets:
        training = [
            record["router_ladder"]["online_query_policy"]["by_target_budget"][budget]
            for _, record in records
        ]
        evaluation = [
            record["evaluation"]["online_query_policy"]["by_target_budget"][budget]
            for _, record in records
        ]
        draws = np.asarray(
            [
                record["bootstrap"]["policy_error_reduction_draws"]
                ["online_selected_router"][budget]
                for _, record in records
            ],
            dtype=np.float64,
        )
        if draws.shape != (len(records), repetitions) or not np.isfinite(draws).all():
            raise ValueError(
                f"online fixed-threshold bootstrap draws are invalid at {budget}"
            )
        pooled_draws = np.mean(draws, axis=0)
        output["by_target_budget"][budget] = {
            "target_query_rate": float(training[0]["target_query_rate"]),
            "target_query_count_by_seed": {
                str(record["seed"]): int(values["target_query_count"])
                for (_, record), values in zip(records, training)
            },
            "fixed_threshold_by_seed": {
                str(record["seed"]): float(values["threshold"])
                for (_, record), values in zip(records, training)
            },
            "tie_convention_by_seed": {
                str(record["seed"]): values["tie_convention"]
                for (_, record), values in zip(records, training)
            },
            "calibration_query_count_by_seed": {
                str(record["seed"]): int(values["calibration_query_count"])
                for (_, record), values in zip(records, training)
            },
            "calibration_realized_query_rate_by_seed": {
                str(record["seed"]): float(values["calibration_realized_query_rate"])
                for (_, record), values in zip(records, training)
            },
            "calibration_realized_query_rate": _scalar_moments(
                [
                    float(values["calibration_realized_query_rate"])
                    for values in training
                ]
            ),
            "evaluation_realized_query_rate": _scalar_moments(
                [
                    float(values["evaluation_realized_query_rate"])
                    for values in evaluation
                ]
            ),
            "evaluation_query_count_by_seed": {
                str(record["seed"]): int(values["evaluation_query_count"])
                for (_, record), values in zip(records, evaluation)
            },
            "evaluation_realized_query_rate_by_seed": {
                str(record["seed"]): float(values["evaluation_realized_query_rate"])
                for (_, record), values in zip(records, evaluation)
            },
            "error_reduction": _scalar_moments(
                [
                    float(
                        record["evaluation"]["policies"]["online_selected_router"]
                        [budget]["error_reduction"]
                    )
                    for _, record in records
                ]
            ),
            "matched_rate_oracle_error_reduction": _scalar_moments(
                [
                    float(
                        record["evaluation"]["policies"]["online_selected_router"]
                        [budget]["oracle_reference_error_reduction"]
                    )
                    for _, record in records
                ]
            ),
            "error_reduction_cluster_ci95": [
                float(value) for value in np.quantile(pooled_draws, (0.025, 0.975))
            ],
            "bootstrap_repetitions": repetitions,
            "bootstrap_method": (
                "aligned_video_cluster_resample_with_fixed_calibration_threshold_then_seed_mean"
            ),
        }
    return output


def _pooled_reselected_contrasts(
    records: list[tuple[pathlib.Path, dict[str, Any]]],
    repetitions: int,
) -> dict[str, Any]:
    """Pool aligned per-seed cluster draws after policy reselection."""

    output: dict[str, Any] = {}
    budgets = records[0][1]["evaluation"]["paired_contrasts"]
    for budget in budgets:
        error_draws = np.asarray(
            [
                record["evaluation"]["paired_contrasts"][budget]["bootstrap_draws"]
                for _, record in records
            ],
            dtype=np.float64,
        )
        selective_draws = np.asarray(
            [
                record["evaluation"]["paired_contrasts"][budget]["coverage_0.90"]["bootstrap_draws"]
                for _, record in records
            ],
            dtype=np.float64,
        )
        if error_draws.shape != (len(records), repetitions) or selective_draws.shape != (
            len(records), repetitions
        ):
            raise ValueError(f"bootstrap draws are incomplete at budget {budget}")
        pooled_error_draws = np.mean(error_draws, axis=0)
        pooled_selective_draws = np.mean(selective_draws, axis=0)
        error_points = np.asarray(
            [record["evaluation"]["paired_contrasts"][budget]["error_reduction_difference"] for _, record in records],
            dtype=np.float64,
        )
        selective_points = np.asarray(
            [
                record["evaluation"]["paired_contrasts"][budget]["coverage_0.90"][
                    "selective_risk_difference"
                ]
                for _, record in records
            ],
            dtype=np.float64,
        )
        output[budget] = {
            "seed_runs": len(records),
            "bootstrap_repetitions": repetitions,
            "bootstrap_method": "aligned_video_cluster_resample_with_policy_reselection_then_seed_mean",
            "error_reduction_difference": float(np.mean(error_points)),
            "error_reduction_difference_cluster_ci95": [
                float(value) for value in np.quantile(pooled_error_draws, (0.025, 0.975))
            ],
            "coverage_0.90_selective_risk_difference": float(np.mean(selective_points)),
            "coverage_0.90_selective_risk_difference_cluster_ci95": [
                float(value) for value in np.quantile(pooled_selective_draws, (0.025, 0.975))
            ],
        }
    return output


def _hybrid_ablation_contrasts(
    records: list[tuple[pathlib.Path, dict[str, Any]]],
    repetitions: int,
) -> dict[str, Any]:
    """Pair the calibration-selected router against timing/choice hybrids."""

    output: dict[str, Any] = {}
    hybrids = (
        "selected_router_query_source_prior_choice",
        "confidence_query_selected_router_choice",
    )
    budgets = records[0][1]["evaluation"]["policies"]["selected_router"]
    for hybrid in hybrids:
        output[hybrid] = {}
        for budget in budgets:
            paired_draws = np.asarray(
                [
                    np.asarray(
                        record["bootstrap"]["policy_error_reduction_draws"]["selected_router"][budget],
                        dtype=np.float64,
                    )
                    - np.asarray(
                        record["bootstrap"]["policy_error_reduction_draws"][hybrid][budget],
                        dtype=np.float64,
                    )
                    for _, record in records
                ],
                dtype=np.float64,
            )
            if paired_draws.shape != (len(records), repetitions):
                raise ValueError(f"hybrid bootstrap draws are incomplete for {hybrid}/{budget}")
            pooled_draws = np.mean(paired_draws, axis=0)
            points = np.asarray(
                [
                    record["evaluation"]["policies"]["selected_router"][budget]["error_reduction"]
                    - record["evaluation"]["policies"][hybrid][budget]["error_reduction"]
                    for _, record in records
                ],
                dtype=np.float64,
            )
            output[hybrid][budget] = {
                "contrast": "selected_router_minus_hybrid_error_reduction",
                "seed_runs": len(records),
                "difference": float(np.mean(points)),
                "sample_std": None if len(points) < 2 else float(np.std(points, ddof=1)),
                "cluster_ci95": [
                    float(value) for value in np.quantile(pooled_draws, (0.025, 0.975))
                ],
                "bootstrap_repetitions": repetitions,
                "bootstrap_method": (
                    "aligned_video_cluster_resample_with_exact_budget_reselection_then_seed_mean"
                ),
            }
    return output


def _context_summary(records: list[tuple[pathlib.Path, dict[str, Any]]]) -> dict[str, Any]:
    def moments(values: list[float | int]) -> dict[str, Any]:
        finite = np.asarray(values, dtype=np.float64)
        return {
            "runs": int(len(finite)),
            "mean": float(np.mean(finite)),
            "sample_std": None if len(finite) < 2 else float(np.std(finite, ddof=1)),
        }

    output: dict[str, Any] = {}
    first_contexts = records[0][1]["evaluation"]["contexts"]
    for context_name, first_context in first_contexts.items():
        output[context_name] = {
            "candidate_modalities": first_context["candidate_modalities"],
            "candidate_diagnostics": {},
            "by_label": {},
            "policies": {},
        }
        for candidate, candidate_metrics in first_context["candidates"].items():
            output[context_name]["candidate_diagnostics"][candidate] = {
                metric: moments(
                    [record["evaluation"]["contexts"][context_name]["candidates"][candidate][metric] for _, record in records]
                )
                for metric in candidate_metrics
            }
        for label, label_metrics in first_context["by_label"].items():
            output[context_name]["by_label"][label] = {
                metric: moments(
                    [record["evaluation"]["contexts"][context_name]["by_label"][label][metric] for _, record in records]
                )
                for metric in label_metrics
            }
        for policy, by_budget in first_context["policy_outcomes"].items():
            output[context_name]["policies"][policy] = {}
            for budget, first_metrics in by_budget.items():
                summary: dict[str, Any] = {}
                for metric in (
                    "query_rate",
                    "base_error",
                    "final_error",
                    "base_macro_f1",
                    "final_macro_f1",
                    "error_reduction",
                    "harmful_query_rate",
                    "useful_query_precision",
                    "candidate_ranking_accuracy",
                    "queried_candidate_selection_accuracy",
                ):
                    values = [
                        record["evaluation"]["contexts"][context_name]["policy_outcomes"][policy][budget][metric]
                        for _, record in records
                    ]
                    finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
                    summary[metric] = {
                        "runs": int(len(finite)),
                        "mean": None if not len(finite) else float(np.mean(finite)),
                        "sample_std": None if len(finite) < 2 else float(np.std(finite, ddof=1)),
                    }
                summary["query_target_counts"] = {
                    modality: moments(
                        [
                            record["evaluation"]["contexts"][context_name]["policy_outcomes"][policy][budget][
                                "query_target_counts"
                            ][modality]
                            for _, record in records
                        ]
                    )
                    for modality in ("audio", "video", "text")
                }
                summary["query_target_outcomes"] = {}
                for modality in ("audio", "video", "text"):
                    summary["query_target_outcomes"][modality] = {}
                    for metric in (
                        "query_count",
                        "realized_value_sum",
                        "harmful_query_count",
                        "mean_realized_value",
                        "harmful_query_rate",
                    ):
                        values = [
                            record["evaluation"]["contexts"][context_name]["policy_outcomes"][policy][budget][
                                "query_target_outcomes"
                            ][modality][metric]
                            for _, record in records
                        ]
                        finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
                        summary["query_target_outcomes"][modality][metric] = {
                            "runs": int(len(finite)),
                            "mean": None if not len(finite) else float(np.mean(finite)),
                            "sample_std": None if len(finite) < 2 else float(np.std(finite, ddof=1)),
                        }
                summary["by_label"] = {
                    label: {
                        metric: moments(
                            [
                                record["evaluation"]["contexts"][context_name]["policy_outcomes"][policy][budget][
                                    "by_label"
                                ][label][metric]
                                for _, record in records
                            ]
                        )
                        for metric in label_metrics
                    }
                    for label, label_metrics in first_metrics["by_label"].items()
                }
                output[context_name]["policies"][policy][budget] = summary
    return output


def _scientific_validation(
    singleton: list[tuple[pathlib.Path, dict[str, Any]]],
    pooled_contrasts: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary_budget = float(config["primary_budget"])
    minimum_primary_gain = float(config["minimum_primary_error_reduction_gain"])
    minimum_video_macro_fraction = float(config["minimum_video_macro_gain_fraction"])
    minimum_nonnegative_contexts = int(config["minimum_nonnegative_start_contexts"])
    maximum_context_harm = float(config["maximum_start_context_harm"])
    headroom_passed = all(bool(record["headroom_gate"]["passed"]) for _, record in singleton)
    task_head_passed = all(bool(record["task_head_gate"]["passed"]) for _, record in singleton)
    budget_key = f"{primary_budget:.6f}"
    if budget_key not in pooled_contrasts:
        raise ValueError(f"primary budget {primary_budget} is absent from pooled contrasts")
    pooled = pooled_contrasts[budget_key]
    router_harmful = np.asarray(
        [
            record["evaluation"]["policies"]["selected_router"][budget_key]["harmful_query_rate"]
            for _, record in singleton
        ],
        dtype=np.float64,
    )
    baseline_harmful = np.asarray(
        [
            record["evaluation"]["policies"]["selected_baseline"][budget_key]["harmful_query_rate"]
            for _, record in singleton
        ],
        dtype=np.float64,
    )
    if not np.isfinite(router_harmful).all() or not np.isfinite(baseline_harmful).all():
        raise ValueError("primary-budget harmful-query rates must be finite")
    contrast = float(pooled["error_reduction_difference"])
    contrast_ci95 = [float(value) for value in pooled["error_reduction_difference_cluster_ci95"]]
    harmful_difference = float(np.mean(baseline_harmful) - np.mean(router_harmful))
    video_macro_difference = float(
        np.mean(
            [
                record["evaluation"]["paired_contrasts"][budget_key]["video_macro_difference"]
                for _, record in singleton
            ]
        )
    )
    video_macro_passed = (
        video_macro_difference > 0.0
        and video_macro_difference >= minimum_video_macro_fraction * contrast
    )
    first_contexts = singleton[0][1]["evaluation"]["contexts"]
    context_differences = {
        context: float(
            np.mean(
                [
                    record["evaluation"]["contexts"][context]["policy_outcomes"]["selected_router"]
                    [budget_key]["error_reduction"]
                    - record["evaluation"]["contexts"][context]["policy_outcomes"]["selected_baseline"]
                    [budget_key]["error_reduction"]
                    for _, record in singleton
                ]
            )
        )
        for context in first_contexts
    }
    nonnegative_contexts = int(sum(value >= 0.0 for value in context_differences.values()))
    context_heterogeneity_passed = (
        nonnegative_contexts >= minimum_nonnegative_contexts
        and min(context_differences.values()) >= -maximum_context_harm
    )
    selection_passed = (
        contrast >= minimum_primary_gain
        and contrast_ci95[0] > 0.0
        and harmful_difference >= 0.0
    )
    gates = {
        "task_head_adequacy": {"passed": task_head_passed},
        "singleton_oracle_headroom": {"passed": headroom_passed},
        "selected_router_selection": {
            "passed": selection_passed,
            "primary_budget": primary_budget,
            "minimum_error_reduction_gain": minimum_primary_gain,
            "error_reduction_difference": contrast,
            "error_reduction_difference_cluster_ci95": contrast_ci95,
            "selected_router_harmful_query_rate_mean": float(np.mean(router_harmful)),
            "selected_baseline_harmful_query_rate_mean": float(np.mean(baseline_harmful)),
            "harmful_query_rate_reduction": harmful_difference,
            "rule": (
                "pooled error-reduction gain at least the predeclared minimum with a positive "
                "95% cluster-bootstrap lower bound and no descriptive increase in mean harmful-query rate"
            ),
        },
        "video_macro_sensitivity": {
            "passed": video_macro_passed,
            "error_reduction_difference_video_macro": video_macro_difference,
            "minimum_fraction_of_utterance_weighted_gain": minimum_video_macro_fraction,
            "required_minimum": minimum_video_macro_fraction * contrast,
            "status": "descriptive_sensitivity_gate",
        },
        "start_context_heterogeneity": {
            "passed": context_heterogeneity_passed,
            "error_reduction_difference_by_start": context_differences,
            "nonnegative_contexts": nonnegative_contexts,
            "minimum_nonnegative_contexts": minimum_nonnegative_contexts,
            "maximum_allowed_context_harm": maximum_context_harm,
            "status": "descriptive_heterogeneity_gate",
        },
    }
    gates["overall_scientific_result"] = {
        "passed": all(
            (
                task_head_passed,
                headroom_passed,
                selection_passed,
                video_macro_passed,
                context_heterogeneity_passed,
            )
        ),
        "status": "positive" if all(
            (
                task_head_passed,
                headroom_passed,
                selection_passed,
                video_macro_passed,
                context_heterogeneity_passed,
            )
        ) else "negative",
    }
    return {"ok": True, "errors": []}, gates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--singleton", nargs="+", required=True)
    parser.add_argument("--pair", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = pathlib.Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    cache_sha256 = config.get("expected_cache_sha256")
    configuration_digest = configuration_sha256(config)
    singleton = _load(args.singleton, "singleton", cache_sha256, configuration_digest, config)
    pair = _load(args.pair, "pair", cache_sha256, configuration_digest, config)
    expected_seeds = {int(value) for value in config.get("seeds", ())}
    if not expected_seeds:
        raise ValueError("configuration must declare the exact seed set")
    for mode, records in (("singleton", singleton), ("pair", pair)):
        _require_seed_set(records, expected_seeds, mode)
    singleton_by_seed = {int(record["seed"]): record for _, record in singleton}
    pair_by_seed = {int(record["seed"]): record for _, record in pair}
    for seed in expected_seeds:
        if (
            singleton_by_seed[seed]["task_head_gate"]["validation_accuracy_by_mask"]
            != pair_by_seed[seed]["task_head_gate"]["validation_accuracy_by_mask"]
        ):
            raise ValueError(f"task-head mask accuracies differ across modes for seed {seed}")
    all_records = singleton + pair
    observed_cache_digests = {record["data_access"]["cache_sha256"] for _, record in all_records}
    if len(observed_cache_digests) != 1:
        raise ValueError("input results use different score caches")
    observed_implementation_digests = {record["implementation_sha256"] for _, record in all_records}
    if observed_implementation_digests != {implementation_sha256()}:
        raise ValueError("input results use a stale or inconsistent implementation")
    observed_group_digests = {
        record["data_access"]["validation_group_sha256"] for _, record in all_records
    }
    if len(observed_group_digests) != 1:
        raise ValueError("input results use different validation video groups")
    observed_group_hash_lists = {
        tuple(record["data_access"]["validation_group_hashes"])
        for _, record in all_records
    }
    if len(observed_group_hash_lists) != 1:
        raise ValueError("input results contain different validation video-group identities")
    bootstrap_protocols = {
        (
            record["bootstrap"]["method"],
            int(record["bootstrap"]["group_draw_seed"]),
            int(record["bootstrap"]["repetitions"]),
        )
        for _, record in all_records
    }
    expected_bootstrap_protocol = {
        (
            "video_cluster_resample_with_policy_reselection",
            int(config.get("bootstrap_seed", 20_270_917)),
            int(config["bootstrap_repetitions"]),
        )
    }
    if bootstrap_protocols != expected_bootstrap_protocol:
        raise ValueError("input results use different bootstrap protocols")
    reference_group_counts: dict[str, int] | None = None
    for _, record in all_records:
        first_budget = next(iter(record["evaluation"]["paired_contrasts"].values()))
        group_counts = {
            entry["group_sha256"]: int(entry["decision_count"])
            for entry in first_budget["video_group_statistics"]
        }
        if reference_group_counts is None:
            reference_group_counts = group_counts
        elif group_counts != reference_group_counts:
            raise ValueError("validation video-group counts differ across inputs")

    singleton_pooled = _pooled_reselected_contrasts(
        singleton,
        int(config["bootstrap_repetitions"]),
    )
    pair_pooled = _pooled_reselected_contrasts(
        pair,
        int(config["bootstrap_repetitions"]),
    )
    scientific_validation, scientific_gates = _scientific_validation(
        singleton,
        singleton_pooled,
        config,
    )
    result = {
        "schema": "conflictbench.mosei-value-aggregate.v2",
        "config_sha256": _digest(config_path),
        "configuration_sha256": configuration_digest,
        "implementation_sha256": next(iter(observed_implementation_digests)),
        "aggregate_script_sha256": _digest(pathlib.Path(__file__).resolve()),
        "cache_sha256": next(iter(observed_cache_digests)),
        "validation_group_sha256": next(iter(observed_group_digests)),
        "inputs": {
            mode: [
                {"path": path.name, "sha256": _digest(path), "seed": int(record["seed"])}
                for path, record in records
            ]
            for mode, records in (("singleton", singleton), ("pair", pair))
        },
        "modes": {
            "singleton": {
                "policies": _summary(singleton),
                "contexts": _context_summary(singleton),
                "diagnostics": _evaluation_diagnostic_summary(singleton),
                "router_ladder": _router_ladder_summary(singleton),
                "online_query_policy": _online_query_policy_summary(
                    singleton,
                    int(config["bootstrap_repetitions"]),
                ),
                "baseline_selection": _baseline_selection_summary(singleton),
                "hybrid_ablation_contrasts": _hybrid_ablation_contrasts(
                    singleton,
                    int(config["bootstrap_repetitions"]),
                ),
                "selected_router_vs_selected_baseline": _contrast_summary(singleton),
                "pooled_video_group_contrasts": singleton_pooled,
            },
            "pair": {
                "policies": _summary(pair),
                "contexts": _context_summary(pair),
                "diagnostics": _evaluation_diagnostic_summary(pair),
                "router_ladder": _router_ladder_summary(pair),
                "online_query_policy": _online_query_policy_summary(
                    pair,
                    int(config["bootstrap_repetitions"]),
                ),
                "baseline_selection": _baseline_selection_summary(pair),
                "hybrid_ablation_contrasts": _hybrid_ablation_contrasts(
                    pair,
                    int(config["bootstrap_repetitions"]),
                ),
                "selected_router_vs_selected_baseline": _contrast_summary(pair),
                "pooled_video_group_contrasts": pair_pooled,
            },
        },
        "headroom_gate": {
            mode: {
                "runs": len(records),
                "passed_runs": int(sum(bool(record["headroom_gate"]["passed"]) for _, record in records)),
                "all_passed": all(bool(record["headroom_gate"]["passed"]) for _, record in records),
            }
            for mode, records in (("singleton", singleton), ("pair", pair))
        },
        "task_head_gate": {
            mode: _task_head_summary(records)
            for mode, records in (("singleton", singleton), ("pair", pair))
        },
        "scientific_gates": scientific_gates,
        "validation": scientific_validation,
    }
    destination = pathlib.Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(destination), "validation": result["validation"]}, sort_keys=True))
    return 0 if scientific_validation["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
