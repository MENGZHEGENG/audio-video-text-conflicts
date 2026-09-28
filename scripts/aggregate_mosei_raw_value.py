#!/usr/bin/env python3
"""Aggregate unprojected released CMU-MOSEI CSD descriptors, not raw media."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from conflictbench.mosei_raw_value_study import (
    RAW_RESULT_SCHEMA,
    RAW_STUDY_SCHEMA,
    configuration_sha256,
    implementation_sha256,
    validate_raw_value_result,
)

AGGREGATE_SCHEMA = "conflictbench.mosei-raw-value-aggregate.v1"
LOCKED_SEEDS = (11, 23, 37, 41, 53)
MODES = ("singleton", "pair")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_RESULT_FIELDS = {
    "schema",
    "study_schema",
    "configuration_sha256",
    "implementation_sha256",
    "mode",
    "seed",
    "data_access",
    "group_partitions",
    "router_ladder",
    "evaluation",
    "headroom_gate",
    "task_head_gate",
    "bootstrap",
    "validation",
    "core_protocol",
    "projection_fit",
}
_POLICY_METRICS = (
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
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _moments(values: Sequence[float | int | None]) -> dict[str, Any]:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    return {
        "runs": len(array),
        "mean": None if not len(array) else float(np.mean(array)),
        "sample_std": None if len(array) < 2 else float(np.std(array, ddof=1)),
    }


def _load_mode_records(
    raw_paths: Sequence[str | pathlib.Path],
    mode: str,
    config: Mapping[str, Any],
) -> list[tuple[pathlib.Path, dict[str, Any]]]:
    expected_cache_sha256 = config.get("expected_cache_sha256")
    records: list[tuple[pathlib.Path, dict[str, Any]]] = []
    resolved_paths: set[pathlib.Path] = set()
    observed_seeds: set[int] = set()
    for value in raw_paths:
        path = pathlib.Path(value).expanduser().resolve()
        if path in resolved_paths:
            raise ValueError(f"duplicate {mode} input path: {path}")
        resolved_paths.add(path)
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        if not isinstance(record, dict):
            raise TypeError(f"{path}: result must be a JSON object")
        if not _REQUIRED_RESULT_FIELDS.issubset(record):
            missing = [name.replace("_", " ") for name in sorted(_REQUIRED_RESULT_FIELDS - set(record))]
            raise ValueError(f"{path}: full core record is incomplete; missing={missing}")
        if record.get("validation") != {"ok": True, "errors": []}:
            raise ValueError(f"{path}: saved structural validation did not pass")
        errors = validate_raw_value_result(
            record,
            expected_config=config,
            expected_mode=mode,
            expected_cache_sha256=expected_cache_sha256,
        )
        if errors:
            raise ValueError(f"{path}: {errors}")
        seed = record.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"{path}: seed is invalid")
        if seed in observed_seeds:
            raise ValueError(f"duplicate {mode} seed: {seed}")
        observed_seeds.add(seed)
        records.append((path, record))
    if observed_seeds != set(LOCKED_SEEDS):
        raise ValueError(
            f"{mode} seed set mismatch: expected {list(LOCKED_SEEDS)}, "
            f"observed {sorted(observed_seeds)}"
        )
    return sorted(records, key=lambda item: int(item[1]["seed"]))


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != RAW_STUDY_SCHEMA:
        raise ValueError(f"configuration schema must be {RAW_STUDY_SCHEMA}")
    if config.get("seeds") != list(LOCKED_SEEDS):
        raise ValueError(f"configuration seeds must be exactly {list(LOCKED_SEEDS)}")
    if config.get("evidence_status") != "exploratory_taskfit_reprojection":
        raise ValueError("configuration must mark this validation-reused lane as exploratory")
    cache_sha256 = config.get("expected_cache_sha256")
    if not isinstance(cache_sha256, str) or _SHA256.fullmatch(cache_sha256) is None:
        raise ValueError("configuration must lock a lowercase raw cache SHA256")


def _validate_shared_protocol(
    by_mode: Mapping[str, list[tuple[pathlib.Path, dict[str, Any]]]],
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    all_records = [item for mode in MODES for item in by_mode[mode]]
    expected_config_sha256 = configuration_sha256(config)
    expected_implementation_sha256 = implementation_sha256()
    expected_cache_sha256 = str(config["expected_cache_sha256"])
    for path, record in all_records:
        if record["schema"] != RAW_RESULT_SCHEMA or record["study_schema"] != RAW_STUDY_SCHEMA:
            raise ValueError(f"{path}: raw result schema is invalid")
        if record["configuration_sha256"] != expected_config_sha256:
            raise ValueError(f"{path}: configuration SHA256 differs from the locked configuration")
        if record["implementation_sha256"] != expected_implementation_sha256:
            raise ValueError(f"{path}: implementation SHA256 differs from the active code")
        if record["data_access"]["cache_sha256"] != expected_cache_sha256:
            raise ValueError(f"{path}: cache SHA256 differs from the locked raw cache")
        if record["data_access"].get("evidence_status") != config["evidence_status"]:
            raise ValueError(f"{path}: evidence status differs from the locked configuration")
        protocol = record["core_protocol"]
        if not isinstance(protocol, Mapping) or set(protocol) != {
            "result_schema",
            "study_schema",
            "configuration_sha256",
            "implementation_sha256",
            "projected_cache_sha256",
            "router_families",
        }:
            raise ValueError(f"{path}: core protocol record is incomplete")

    provenance_records = {
        json.dumps(record["data_access"]["source_provenance"], sort_keys=True)
        for _, record in all_records
    }
    if len(provenance_records) != 1:
        raise ValueError("input results do not share one raw-source provenance record")
    validation_groups = {
        (
            record["data_access"]["validation_group_sha256"],
            tuple(record["data_access"]["validation_group_hashes"]),
        )
        for _, record in all_records
    }
    if len(validation_groups) != 1:
        raise ValueError("input results use different validation video groups")
    reference_records = {
        (
            record["task_head_gate"]["reference_record_sha256"],
            record["task_head_gate"]["reference_validation_accuracy"],
        )
        for _, record in all_records
    }
    expected_reference = {
        (
            config["task_head_reference_sha256"],
            config["reference_full_avt_validation_accuracy"],
        )
    }
    if reference_records != expected_reference:
        raise ValueError("task-head reference record differs from the locked configuration")

    expected_bootstrap = (
        "video_cluster_resample_with_policy_reselection",
        int(config["bootstrap_seed"]),
        int(config["bootstrap_repetitions"]),
    )
    bootstrap_protocols = {
        (
            record["bootstrap"]["method"],
            int(record["bootstrap"]["group_draw_seed"]),
            int(record["bootstrap"]["repetitions"]),
        )
        for _, record in all_records
    }
    if bootstrap_protocols != {expected_bootstrap}:
        raise ValueError("input results use a non-grouped or inconsistent bootstrap protocol")

    reference_group_counts: dict[str, int] | None = None
    for path, record in all_records:
        for budget, contrast in record["evaluation"]["paired_contrasts"].items():
            counts = {
                entry["group_sha256"]: int(entry["decision_count"])
                for entry in contrast["video_group_statistics"]
            }
            if reference_group_counts is None:
                reference_group_counts = counts
            elif counts != reference_group_counts:
                raise ValueError(
                    f"{path}: validation video-group counts differ at budget {budget}"
                )

    singleton_by_seed = {record["seed"]: record for _, record in by_mode["singleton"]}
    pair_by_seed = {record["seed"]: record for _, record in by_mode["pair"]}
    projection_records: dict[str, dict[str, Any]] = {}
    for seed in LOCKED_SEEDS:
        singleton = singleton_by_seed[seed]
        pair = pair_by_seed[seed]
        if singleton["projection_fit"] != pair["projection_fit"]:
            raise ValueError(f"projection attestations differ across modes for seed {seed}")
        if (
            singleton["core_protocol"]["projected_cache_sha256"]
            != pair["core_protocol"]["projected_cache_sha256"]
        ):
            raise ValueError(f"projected cache records differ across modes for seed {seed}")
        if (
            singleton["task_head_gate"]["validation_accuracy_by_mask"]
            != pair["task_head_gate"]["validation_accuracy_by_mask"]
        ):
            raise ValueError(f"task-head mask accuracies differ across modes for seed {seed}")
        projection = singleton["projection_fit"]
        projection_records[str(seed)] = {
            "attestation_sha256": projection["attestation_sha256"],
            "fit_group_digest": projection["fit_group_digest"],
            "fit_utterances": projection["fit_utterances"],
            "fit_video_groups": projection["fit_video_groups"],
            "projected_cache_sha256": singleton["core_protocol"]["projected_cache_sha256"],
            "modalities": projection["modalities"],
        }
    return projection_records


def _policy_summary(records: Sequence[tuple[pathlib.Path, dict[str, Any]]]) -> dict[str, Any]:
    first = records[0][1]["evaluation"]["policies"]
    return {
        policy: {
            budget: {
                metric: _moments(
                    [
                        record["evaluation"]["policies"][policy][budget][metric]
                        for _, record in records
                    ]
                )
                for metric in _POLICY_METRICS
            }
            for budget in by_budget
        }
        for policy, by_budget in first.items()
    }


def _online_query_policy_summary(
    records: Sequence[tuple[pathlib.Path, dict[str, Any]]],
    repetitions: int,
) -> dict[str, Any]:
    """Summarize calibration-fitted thresholds without evaluation reselection."""

    first_training = records[0][1]["router_ladder"]["online_query_policy"]
    first_evaluation = records[0][1]["evaluation"]["online_query_policy"]
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
        "router_training_attestation_sha256_by_seed": {
            str(record["seed"]): record["router_ladder"][
                "training_attestation_sha256"
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
    for budget in first_training["by_target_budget"]:
        training = [
            record["router_ladder"]["online_query_policy"]["by_target_budget"][
                budget
            ]
            for _, record in records
        ]
        evaluation = [
            record["evaluation"]["online_query_policy"]["by_target_budget"][budget]
            for _, record in records
        ]
        metrics = [
            record["evaluation"]["policies"]["online_selected_router"][budget]
            for _, record in records
        ]
        draws = np.asarray(
            [
                record["bootstrap"]["policy_error_reduction_draws"][
                    "online_selected_router"
                ][budget]
                for _, record in records
            ],
            dtype=np.float64,
        )
        expected_shape = (len(records), repetitions)
        if draws.shape != expected_shape or not np.isfinite(draws).all():
            raise ValueError(
                f"online fixed-threshold bootstrap draws are invalid at budget {budget}"
            )
        pooled_draws = np.mean(draws, axis=0)
        paired_error_reduction = {
            "contrast": "online_selected_router_minus_no_query",
            **_moments([float(values["error_reduction"]) for values in metrics]),
            "cluster_ci95": [
                float(value) for value in np.quantile(pooled_draws, (0.025, 0.975))
            ],
            "bootstrap_repetitions": repetitions,
            "bootstrap_method": (
                "aligned_video_cluster_resample_with_fixed_calibration_threshold_then_seed_mean"
            ),
        }
        matched_oracle = {
            "query_rate": _moments(
                [float(values["oracle_reference_query_rate"]) for values in metrics]
            ),
            "error_reduction": _moments(
                [
                    float(values["oracle_reference_error_reduction"])
                    for values in metrics
                ]
            ),
            "regret": _moments([float(values["oracle_regret"]) for values in metrics]),
            "by_seed": {
                str(record["seed"]): {
                    "query_rate": float(values["oracle_reference_query_rate"]),
                    "error_reduction": float(
                        values["oracle_reference_error_reduction"]
                    ),
                    "regret": float(values["oracle_regret"]),
                }
                for (_, record), values in zip(records, metrics)
            },
        }
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
            "query_on_equal_by_seed": {
                str(record["seed"]): bool(values["query_on_equal"])
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
            "calibration_realized_query_rate": _moments(
                [float(values["calibration_realized_query_rate"]) for values in training]
            ),
            "evaluation_query_count_by_seed": {
                str(record["seed"]): int(values["evaluation_query_count"])
                for (_, record), values in zip(records, evaluation)
            },
            "evaluation_realized_query_rate_by_seed": {
                str(record["seed"]): float(values["evaluation_realized_query_rate"])
                for (_, record), values in zip(records, evaluation)
            },
            "evaluation_realized_query_rate": _moments(
                [float(values["evaluation_realized_query_rate"]) for values in evaluation]
            ),
            "paired_error_reduction_vs_no_query": paired_error_reduction,
            "realized_rate_matched_oracle": matched_oracle,
        }
    return output


def _online_fixed_threshold_gate(
    summary: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate the online policy against no query without cohort reselection."""

    target_budget = float(config["primary_budget"])
    budget = f"{target_budget:.6f}"
    if budget not in summary["by_target_budget"]:
        raise ValueError(
            f"primary budget {target_budget} is absent from the online policy summary"
        )
    budget_summary = summary["by_target_budget"][budget]
    paired = budget_summary["paired_error_reduction_vs_no_query"]
    point = float(paired["mean"])
    interval = [float(value) for value in paired["cluster_ci95"]]
    passed = point >= 0.0 and interval[0] > 0.0
    return {
        "passed": passed,
        "status": "positive" if passed else "negative",
        "evaluation_protocol": "calibration_fitted_fixed_threshold_per_example",
        "comparison": "online_selected_router_minus_no_query",
        "target_budget": target_budget,
        "minimum_error_reduction_gain": 0.0,
        "rule": (
            "nonnegative mean paired error reduction with a positive 95% "
            "video-cluster bootstrap lower bound"
        ),
        "paired_error_reduction_vs_no_query": paired,
        "evaluation_realized_query_rate": budget_summary[
            "evaluation_realized_query_rate"
        ],
        "realized_rate_matched_oracle": budget_summary[
            "realized_rate_matched_oracle"
        ],
        "threshold_provenance": {
            "fit_split": summary["fit_split"],
            "priority": summary["priority"],
            "candidate_choice": summary["candidate_choice"],
            "selected_family_by_seed": summary["selected_family_by_seed"],
            "router_training_attestation_sha256_by_seed": summary[
                "router_training_attestation_sha256_by_seed"
            ],
            "calibration_priority_sha256_by_seed": summary[
                "calibration_priority_sha256_by_seed"
            ],
            "evaluation_priority_sha256_by_seed": summary[
                "evaluation_priority_sha256_by_seed"
            ],
            "evaluation_choice_sha256_by_seed": summary[
                "evaluation_choice_sha256_by_seed"
            ],
            "evaluation_attestation_sha256_by_seed": summary[
                "evaluation_attestation_sha256_by_seed"
            ],
            "fixed_threshold_by_seed": budget_summary["fixed_threshold_by_seed"],
            "query_on_equal_by_seed": budget_summary["query_on_equal_by_seed"],
            "tie_convention_by_seed": budget_summary["tie_convention_by_seed"],
        },
    }


def _pooled_contrasts(
    records: Sequence[tuple[pathlib.Path, dict[str, Any]]], repetitions: int
) -> dict[str, Any]:
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
                record["evaluation"]["paired_contrasts"][budget]["coverage_0.90"][
                    "bootstrap_draws"
                ]
                for _, record in records
            ],
            dtype=np.float64,
        )
        expected_shape = (len(LOCKED_SEEDS), repetitions)
        if error_draws.shape != expected_shape or selective_draws.shape != expected_shape:
            raise ValueError(f"grouped bootstrap draws are incomplete at budget {budget}")
        if not np.isfinite(error_draws).all() or not np.isfinite(selective_draws).all():
            raise ValueError(f"grouped bootstrap draws are non-finite at budget {budget}")
        error_points = np.asarray(
            [
                record["evaluation"]["paired_contrasts"][budget][
                    "error_reduction_difference"
                ]
                for _, record in records
            ],
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
        pooled_error = np.mean(error_draws, axis=0)
        pooled_selective = np.mean(selective_draws, axis=0)
        output[budget] = {
            "seed_runs": len(records),
            "bootstrap_repetitions": repetitions,
            "bootstrap_method": (
                "aligned_video_cluster_resample_with_policy_reselection_then_seed_mean"
            ),
            "error_reduction_difference": float(np.mean(error_points)),
            "error_reduction_difference_cluster_ci95": [
                float(value) for value in np.quantile(pooled_error, (0.025, 0.975))
            ],
            "coverage_0.90_selective_risk_difference": float(np.mean(selective_points)),
            "coverage_0.90_selective_risk_difference_cluster_ci95": [
                float(value) for value in np.quantile(pooled_selective, (0.025, 0.975))
            ],
        }
    return output


def _scientific_result(
    singleton: Sequence[tuple[pathlib.Path, dict[str, Any]]],
    pooled: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    primary_budget = float(config["primary_budget"])
    budget = f"{primary_budget:.6f}"
    if budget not in pooled:
        raise ValueError(f"primary budget {primary_budget} is absent from pooled contrasts")
    contrast = float(pooled[budget]["error_reduction_difference"])
    interval = [float(value) for value in pooled[budget]["error_reduction_difference_cluster_ci95"]]
    router_harm = np.asarray(
        [
            record["evaluation"]["policies"]["selected_router"][budget][
                "harmful_query_rate"
            ]
            for _, record in singleton
        ],
        dtype=np.float64,
    )
    baseline_harm = np.asarray(
        [
            record["evaluation"]["policies"]["selected_baseline"][budget][
                "harmful_query_rate"
            ]
            for _, record in singleton
        ],
        dtype=np.float64,
    )
    harmful_reduction = float(np.mean(baseline_harm) - np.mean(router_harm))
    video_macro_difference = float(
        np.mean(
            [
                record["evaluation"]["paired_contrasts"][budget]["video_macro_difference"]
                for _, record in singleton
            ]
        )
    )
    contexts = singleton[0][1]["evaluation"]["contexts"]
    context_differences = {
        context: float(
            np.mean(
                [
                    record["evaluation"]["contexts"][context]["policy_outcomes"]
                    ["selected_router"][budget]["error_reduction"]
                    - record["evaluation"]["contexts"][context]["policy_outcomes"]
                    ["selected_baseline"][budget]["error_reduction"]
                    for _, record in singleton
                ]
            )
        )
        for context in contexts
    }
    nonnegative_contexts = sum(value >= 0.0 for value in context_differences.values())
    selection_passed = (
        contrast >= float(config["minimum_primary_error_reduction_gain"])
        and interval[0] > 0.0
        and harmful_reduction >= 0.0
    )
    video_macro_passed = (
        video_macro_difference > 0.0
        and video_macro_difference
        >= float(config["minimum_video_macro_gain_fraction"]) * contrast
    )
    context_passed = (
        nonnegative_contexts >= int(config["minimum_nonnegative_start_contexts"])
        and min(context_differences.values()) >= -float(config["maximum_start_context_harm"])
    )
    gates = {
        "task_head_adequacy": {
            "passed": all(record["task_head_gate"]["passed"] for _, record in singleton)
        },
        "singleton_oracle_headroom": {
            "passed": all(record["headroom_gate"]["passed"] for _, record in singleton)
        },
        "selected_router_selection": {
            "passed": selection_passed,
            "comparison": (
                "offline_selected_router_minus_calibration_selected_baseline"
            ),
            "primary_budget": primary_budget,
            "error_reduction_difference": contrast,
            "error_reduction_difference_cluster_ci95": interval,
            "harmful_query_rate_reduction": harmful_reduction,
        },
        "video_macro_sensitivity": {
            "passed": video_macro_passed,
            "error_reduction_difference_video_macro": video_macro_difference,
        },
        "start_context_heterogeneity": {
            "passed": context_passed,
            "error_reduction_difference_by_start": context_differences,
            "nonnegative_contexts": int(nonnegative_contexts),
        },
    }
    passed = all(bool(gate["passed"]) for gate in gates.values())
    return {
        "passed": passed,
        "status": "positive" if passed else "negative",
        "evaluation_protocol": (
            "offline_exact_cohort_query_ranking_with_budget_reselection"
        ),
        "comparison": "selected_router_minus_calibration_selected_baseline",
        "gates": gates,
    }


def aggregate_raw_results(
    config_path: str | pathlib.Path,
    singleton_paths: Sequence[str | pathlib.Path],
    pair_paths: Sequence[str | pathlib.Path],
) -> dict[str, Any]:
    """Validate and summarize the exact ten-run unprojected CSD descriptor study."""

    config_source = pathlib.Path(config_path).expanduser().resolve()
    with config_source.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError("configuration must be a JSON object")
    _validate_config(config)
    by_mode = {
        "singleton": _load_mode_records(singleton_paths, "singleton", config),
        "pair": _load_mode_records(pair_paths, "pair", config),
    }
    projection_records = _validate_shared_protocol(by_mode, config)
    pooled = {
        mode: _pooled_contrasts(records, int(config["bootstrap_repetitions"]))
        for mode, records in by_mode.items()
    }
    online = {
        mode: _online_query_policy_summary(
            records, int(config["bootstrap_repetitions"])
        )
        for mode, records in by_mode.items()
    }
    scientific = _scientific_result(by_mode["singleton"], pooled["singleton"], config)
    online_gate = _online_fixed_threshold_gate(online["singleton"], config)
    first = by_mode["singleton"][0][1]
    return {
        "schema": AGGREGATE_SCHEMA,
        "evidence_status": config["evidence_status"],
        "seeds": list(LOCKED_SEEDS),
        "configuration_file_sha256": _sha256(config_source),
        "configuration_sha256": configuration_sha256(config),
        "implementation_sha256": implementation_sha256(),
        "aggregate_script_sha256": _sha256(pathlib.Path(__file__).resolve()),
        "raw_cache_sha256": config["expected_cache_sha256"],
        "source_provenance": first["data_access"]["source_provenance"],
        "task_head_reference": {
            "name": config["task_head_reference_name"],
            "sha256": config["task_head_reference_sha256"],
            "validation_accuracy": config["reference_full_avt_validation_accuracy"],
        },
        "projection_attestations_by_seed": projection_records,
        "inputs": {
            mode: [
                {"file": path.name, "sha256": _sha256(path), "seed": record["seed"]}
                for path, record in records
            ]
            for mode, records in by_mode.items()
        },
        "modes": {
            mode: {
                "policies": _policy_summary(records),
                "online_query_policy": online[mode],
                "pooled_video_group_contrasts": pooled[mode],
                "headroom": {
                    "runs": len(records),
                    "passed_runs": sum(record["headroom_gate"]["passed"] for _, record in records),
                },
                "task_head": {
                    "runs": len(records),
                    "passed_runs": sum(record["task_head_gate"]["passed"] for _, record in records),
                },
            }
            for mode, records in by_mode.items()
        },
        "scientific_result": scientific,
        "online_fixed_threshold_gate": online_gate,
        # A negative scientific result is structurally valid and remains publishable information.
        "validation": {"ok": True, "errors": []},
    }


def _write_new_json(path: pathlib.Path, record: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, destination)
        os.unlink(temporary_name)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--singleton", nargs="+", required=True)
    parser.add_argument("--pair", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    result = aggregate_raw_results(args.config, args.singleton, args.pair)
    _write_new_json(pathlib.Path(args.output), result)
    print(
        json.dumps(
            {
                "output": str(pathlib.Path(args.output)),
                "validation": result["validation"],
                "scientific_result": result["scientific_result"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
