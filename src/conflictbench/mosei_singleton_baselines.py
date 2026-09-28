"""Direct cost-sensitive actions for the singleton CMU-MOSEI study.

This module is intentionally separate from the frozen value-study v2 result
format.  It reads only the official training and validation arrays, fits task
heads on task-fit video groups, fits action risk on router-fit groups, and
fits calibration maps and online thresholds on calibration groups.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import sklearn

from . import mosei_value_study as base
from .value_of_information import masked_observations

STUDY_SCHEMA = "conflictbench.mosei-singleton-baselines.v3"
RESULT_SCHEMA = "conflictbench.mosei-singleton-baseline-result.v3"
CACHE_SCHEMA = "conflictbench.mosei-cache.v2"
MODALITIES = ("audio", "video", "text")
TASK_HEAD_FAMILIES = (
    "separate_mask_ridge",
    "shared_mask_aware_ridge",
    "modality_dropout_ridge",
)
ACTION_ORDER = (
    "answer",
    "abstain",
    "acquire_audio",
    "acquire_video",
    "acquire_text",
)
RISK_ACTIONS = ("answer", "acquire_audio", "acquire_video", "acquire_text")


@dataclass(frozen=True)
class _Split:
    x: np.ndarray
    y: np.ndarray
    sample_ids: tuple[str, ...]
    groups: np.ndarray


@dataclass(frozen=True)
class _ActionDecisions:
    labels: np.ndarray
    state_features: np.ndarray
    base_loss: np.ndarray
    base_confidence: np.ndarray
    candidate_loss: np.ndarray
    candidate_confidence: np.ndarray
    candidate_modality: np.ndarray
    context_index: np.ndarray
    groups: np.ndarray


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def _file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _hash_list_digest(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def configuration_sha256(config: Mapping[str, Any]) -> str:
    return _json_sha256(config)


def implementation_sha256() -> str:
    root = pathlib.Path(__file__).resolve().parents[2]
    paths = (
        pathlib.Path(__file__).resolve(),
        pathlib.Path(base.__file__).resolve(),
        pathlib.Path(__file__).with_name("value_of_information.py").resolve(),
        root / "scripts" / "run_mosei_singleton_baselines.py",
        root / "scripts" / "verify_mosei_singleton_baselines.py",
        root / "scripts" / "aggregate_mosei_singleton_baselines.py",
        root / "scripts" / "verify_mosei_singleton_aggregate.py",
    )
    digest = hashlib.sha256()
    for source in paths:
        digest.update(source.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != STUDY_SCHEMA:
        raise ValueError(f"study configuration schema must be {STUDY_SCHEMA}")
    runtime = config.get("runtime")
    expected_runtime = {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
    if not isinstance(runtime, Mapping) or dict(runtime) != expected_runtime:
        locked_runtime = dict(runtime) if isinstance(runtime, Mapping) else runtime
        raise ValueError(
            f"runtime differs from the locked configuration: expected={locked_runtime!r} "
            f"observed={expected_runtime!r}"
        )
    fractions = np.asarray(config.get("train_group_fractions", ()), dtype=np.float64)
    if (
        fractions.shape != (3,)
        or np.any(fractions <= 0)
        or not np.isclose(fractions.sum(), 1.0)
    ):
        raise ValueError(
            "train_group_fractions must be three positive values summing to one"
        )
    budgets = np.asarray(config.get("budgets", ()), dtype=np.float64)
    if (
        budgets.ndim != 1
        or len(budgets) == 0
        or not np.isfinite(budgets).all()
        or np.any((budgets < 0.0) | (budgets > 1.0))
        or len(np.unique(budgets)) != len(budgets)
    ):
        raise ValueError("budgets must be unique finite values within [0, 1]")
    primary = config.get("primary_budget")
    if (
        isinstance(primary, bool)
        or not isinstance(primary, (int, float))
        or not np.any(np.isclose(budgets, float(primary)))
    ):
        raise ValueError("primary_budget must be one of the configured budgets")
    seeds = config.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in seeds
        )
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be unique nonnegative integers")

    task = config.get("task_heads")
    if (
        not isinstance(task, Mapping)
        or tuple(task.get("families", ())) != TASK_HEAD_FAMILIES
    ):
        raise ValueError("task_heads.families must match the locked family order")
    for name in ("ridge",):
        value = task.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"task_heads.{name} must be nonnegative")
    draws = task.get("modality_dropout_draws_per_example")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("modality_dropout_draws_per_example must be positive")
    minimum_accuracy = task.get("minimum_validation_accuracy_per_mask")
    if (
        isinstance(minimum_accuracy, bool)
        or not isinstance(minimum_accuracy, (int, float))
        or not np.isfinite(minimum_accuracy)
        or not 0.0 <= float(minimum_accuracy) <= 1.0
    ):
        raise ValueError("minimum_validation_accuracy_per_mask must lie in [0, 1]")

    classifier = config.get("action_classifier")
    if not isinstance(classifier, Mapping):
        raise TypeError("action_classifier must be an object")
    if classifier.get("cost_units") != "normalized_zero_one_task_loss":
        raise ValueError("action_classifier.cost_units is invalid")
    if (
        classifier.get("acquisition_cost_semantics")
        != "fixed analysis sensitivity, not measured latency"
    ):
        raise ValueError("action_classifier.acquisition_cost_semantics is invalid")
    if tuple(classifier.get("action_order", ())) != ACTION_ORDER:
        raise ValueError(
            "action_classifier.action_order does not match the locked action set"
        )
    for name in ("risk_ridge", "calibration_ridge"):
        value = classifier.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"action_classifier.{name} must be nonnegative")
    costs = classifier.get("cost_matrix")
    if not isinstance(costs, Mapping) or set(costs) != {
        "correct_answer",
        "incorrect_answer",
        "abstain",
        "acquisition",
    }:
        raise ValueError("action_classifier.cost_matrix is invalid")
    acquisition = costs.get("acquisition")
    if not isinstance(acquisition, Mapping) or set(acquisition) != set(MODALITIES):
        raise ValueError("acquisition costs must define audio, video, and text")
    numeric_costs = [
        costs["correct_answer"],
        costs["incorrect_answer"],
        costs["abstain"],
        *(acquisition[name] for name in MODALITIES),
    ]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or value < 0
        for value in numeric_costs
    ):
        raise ValueError("all action costs must be finite and nonnegative")
    if float(costs["incorrect_answer"]) <= float(costs["correct_answer"]):
        raise ValueError("incorrect_answer must cost more than correct_answer")

    dependence = config.get("task_head_sensitivity")
    if not isinstance(dependence, Mapping):
        raise TypeError("task_head_sensitivity must be an object")
    tolerance = dependence.get("maximum_primary_cost_reduction_range")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not np.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ValueError("task-head sensitivity tolerance must be nonnegative")
    if not isinstance(dependence.get("require_matching_sign"), bool):
        raise TypeError("task-head sensitivity sign setting must be boolean")

    uncertainty = config.get("uncertainty")
    if not isinstance(uncertainty, Mapping):
        raise TypeError("uncertainty must be an object")
    repetitions = uncertainty.get("bootstrap_repetitions")
    bootstrap_seed = uncertainty.get("bootstrap_seed")
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions <= 0
    ):
        raise ValueError("uncertainty.bootstrap_repetitions must be positive")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError("uncertainty.bootstrap_seed must be nonnegative")
    if uncertainty.get("unit") != "validation_video_group":
        raise ValueError("uncertainty.unit must be validation_video_group")

    comparator = config.get("published_comparator")
    if (
        not isinstance(comparator, Mapping)
        or comparator.get("status") != "excluded_incompatible"
        or comparator.get("benchmark_only") is not True
        or comparator.get("selected") is not None
    ):
        raise ValueError("published comparator decision must remain benchmark-only")
    sources = comparator.get("sources")
    if (
        not isinstance(sources, list)
        or len(sources) < 3
        or any(
            not isinstance(source, str) or not source.startswith("https://")
            for source in sources
        )
    ):
        raise ValueError("published comparator sources are incomplete")


def _load_split(
    cache_path: str | pathlib.Path,
    config: Mapping[str, Any],
    split_name: str,
) -> _Split:
    _validate_config(config)
    if split_name not in {"train", "valid"}:
        raise ValueError("only train and valid splits may be opened")
    source = pathlib.Path(cache_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    expected_digest = config.get("expected_cache_sha256")
    observed_digest = _file_sha256(source)
    if expected_digest is not None and str(expected_digest) != observed_digest:
        raise ValueError("cache SHA256 does not match the locked configuration")
    required = {
        "cache_schema",
        "metadata_json",
        "train_x",
        "train_y",
        "train_sample_ids",
        "valid_x",
        "valid_y",
        "valid_sample_ids",
        "test_x",
        "test_y",
        "test_sample_ids",
    }
    with np.load(source, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"cache is missing fields: {missing}")
        if str(np.asarray(archive["cache_schema"]).reshape(()).item()) != CACHE_SCHEMA:
            raise ValueError("unsupported cache schema")
        metadata = json.loads(
            str(np.asarray(archive["metadata_json"]).reshape(()).item())
        )
        if metadata.get("alignment") != "positive_interval_overlap_mean":
            raise ValueError("cache must use utterance-aligned interval pooling")
        x = np.asarray(archive[f"{split_name}_x"], dtype=np.float64)
        y = np.asarray(archive[f"{split_name}_y"], dtype=np.int64)
        sample_ids = tuple(
            str(value) for value in archive[f"{split_name}_sample_ids"].tolist()
        )
    if (
        x.shape != (len(y), 3)
        or len(sample_ids) != len(y)
        or len(set(sample_ids)) != len(y)
    ):
        raise ValueError(f"{split_name} arrays are not aligned")
    if not np.isfinite(x).all() or not np.isin(y, (0, 1)).all():
        raise ValueError(f"{split_name} scores or labels are invalid")
    counts = config.get("expected_split_counts")
    if counts is not None:
        locked = {name: int(counts[name]) for name in ("train", "valid", "test")}
        observed = {
            name: int(metadata.get("loaded_split_counts", {}).get(name, -1))
            for name in locked
        }
        if locked != observed or len(y) != locked[split_name]:
            raise ValueError("cache counts do not match the locked configuration")
    groups = np.asarray([base.video_group_id(value) for value in sample_ids], dtype="U")
    return _Split(x=x, y=y, sample_ids=sample_ids, groups=groups)


def _observed_action_features(
    z: np.ndarray,
    observed_mask: np.ndarray,
    base_score: np.ndarray,
) -> np.ndarray:
    """Build pre-query features without reading an unobserved score."""

    values = np.asarray(z, dtype=np.float64)
    mask = np.asarray(observed_mask)
    scores = np.asarray(base_score, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or mask.shape != (3,)
        or mask.dtype.kind != "b"
        or not np.any(mask)
        or scores.shape != (len(values),)
        or not np.isfinite(scores).all()
    ):
        raise ValueError(
            "action features require valid scores and a nonempty length-three mask"
        )
    observed = values[:, mask]
    expanded_mask = np.broadcast_to(mask, values.shape)
    masked_and_mask = masked_observations(values, expanded_mask)
    summary = np.column_stack(
        (
            np.mean(observed, axis=1),
            np.mean(np.abs(observed), axis=1),
            np.max(observed, axis=1) - np.min(observed, axis=1),
            np.abs(scores),
        )
    )
    return np.column_stack((masked_and_mask, summary))


def _select_exact_budget(
    priority: np.ndarray, budget: float, *, seed: int
) -> np.ndarray:
    """Select an exact rounded fraction with seeded, reproducible tie breaks."""

    values = np.asarray(priority, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("priority must be a finite one-dimensional array")
    if (
        isinstance(budget, bool)
        or not np.isfinite(budget)
        or not 0.0 <= float(budget) <= 1.0
    ):
        raise ValueError("budget must lie within [0, 1]")
    count = int(np.rint(float(budget) * len(values)))
    tie_break = np.random.default_rng(int(seed)).random(len(values))
    order = np.lexsort((tie_break, -values))
    selected = np.zeros(len(values), dtype=bool)
    selected[order[:count]] = True
    return selected


def _fit_task_head(
    family: str,
    z: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float,
    seed: int,
    dropout_draws: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    masks = base._observed_masks()
    targets = 2 * np.asarray(y, dtype=np.float64) - 1.0
    if family == "separate_mask_ridge":
        weights = base._fit_task_model(z, y, ridge)
        digest_values = np.concatenate(
            [np.asarray(weights[base._mask_key(mask)]) for mask in masks]
        )
        return {"kind": family, "weights": weights}, {
            "fit_split": "task_fit",
            "training_examples": len(y),
            "expanded_training_examples": int(len(y) * len(masks)),
            "weights_sha256": _array_sha256(digest_values),
        }

    feature_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    mask_indices: list[np.ndarray] = []
    if family == "shared_mask_aware_ridge":
        for mask_index, mask in enumerate(masks):
            feature_blocks.append(base._task_features(z, mask))
            target_blocks.append(targets)
            mask_indices.append(np.full(len(y), mask_index, dtype=np.int64))
    elif family == "modality_dropout_ridge":
        rng = np.random.default_rng(int(seed) * 1_000_003 + 71)
        for _ in range(int(dropout_draws)):
            assignments = rng.integers(0, len(masks), size=len(y))
            for mask_index, mask in enumerate(masks):
                selected = np.flatnonzero(assignments == mask_index)
                if len(selected):
                    feature_blocks.append(base._task_features(z[selected], mask))
                    target_blocks.append(targets[selected])
                    mask_indices.append(
                        np.full(len(selected), mask_index, dtype=np.int64)
                    )
    else:
        raise ValueError(f"unknown task-head family: {family}")
    features = np.concatenate(feature_blocks)
    expanded_targets = np.concatenate(target_blocks)
    assignments = np.concatenate(mask_indices)
    weights = base._ridge_fit(features, expanded_targets, ridge)
    return {"kind": family, "weights": weights}, {
        "fit_split": "task_fit",
        "training_examples": len(y),
        "expanded_training_examples": len(expanded_targets),
        "mask_assignment_sha256": _array_sha256(assignments),
        "weights_sha256": _array_sha256(weights),
    }


def _task_scores(
    z: np.ndarray,
    mask: np.ndarray,
    task_head: Mapping[str, Any],
) -> np.ndarray:
    kind = task_head.get("kind")
    if kind == "separate_mask_ridge":
        return base._task_scores(z, mask, task_head["weights"])
    if kind in {"shared_mask_aware_ridge", "modality_dropout_ridge"}:
        return base._linear_predict(
            base._task_features(z, mask), np.asarray(task_head["weights"])
        )
    raise ValueError(f"unknown task-head family: {kind}")


def _make_action_decisions(
    z: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    task_head: Mapping[str, Any],
) -> _ActionDecisions:
    state_blocks: list[np.ndarray] = []
    base_losses: list[np.ndarray] = []
    base_confidences: list[np.ndarray] = []
    candidate_losses: list[np.ndarray] = []
    candidate_confidences: list[np.ndarray] = []
    candidate_modalities: list[np.ndarray] = []
    context_indices: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    group_blocks: list[np.ndarray] = []
    for context_index, (observed_mask, candidates) in enumerate(
        base._contexts("singleton")
    ):
        base_score = _task_scores(z, observed_mask, task_head)
        base_prediction = (base_score >= 0).astype(np.int64)
        after_losses: list[np.ndarray] = []
        after_confidences: list[np.ndarray] = []
        for candidate in candidates:
            after_mask = observed_mask.copy()
            after_mask[candidate] = True
            after_score = _task_scores(z, after_mask, task_head)
            after_prediction = (after_score >= 0).astype(np.int64)
            after_losses.append((after_prediction != y).astype(np.float64))
            after_confidences.append(np.abs(after_score))
        state_blocks.append(_observed_action_features(z, observed_mask, base_score))
        base_losses.append((base_prediction != y).astype(np.float64))
        base_confidences.append(np.abs(base_score))
        candidate_losses.append(np.column_stack(after_losses))
        candidate_confidences.append(np.column_stack(after_confidences))
        candidate_modalities.append(
            np.broadcast_to(
                np.asarray(candidates, dtype=np.int64), (len(z), len(candidates))
            )
        )
        context_indices.append(np.full(len(z), context_index, dtype=np.int64))
        label_blocks.append(np.asarray(y, dtype=np.int64))
        group_blocks.append(np.asarray(groups, dtype="U"))
    return _ActionDecisions(
        labels=np.concatenate(label_blocks),
        state_features=np.concatenate(state_blocks),
        base_loss=np.concatenate(base_losses),
        base_confidence=np.concatenate(base_confidences),
        candidate_loss=np.concatenate(candidate_losses),
        candidate_confidence=np.concatenate(candidate_confidences),
        candidate_modality=np.concatenate(candidate_modalities),
        context_index=np.concatenate(context_indices),
        groups=np.concatenate(group_blocks),
    )


def _action_design(
    state_features: np.ndarray,
    action_indices: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    state = (np.asarray(state_features, dtype=np.float64) - mean) / scale
    actions = np.asarray(action_indices, dtype=np.int64)
    if (
        state.ndim != 2
        or actions.shape != (len(state),)
        or np.any((actions < 0) | (actions >= 4))
    ):
        raise ValueError("action design inputs are invalid")
    action_code = np.eye(4, dtype=np.float64)[actions]
    interactions = (action_code[:, :, None] * state[:, None, :]).reshape(len(state), -1)
    return np.column_stack((state, action_code, interactions))


def _risk_targets(
    decisions: _ActionDecisions,
    base_cost: np.ndarray,
    candidate_cost: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        np.asarray(base_cost).shape != (len(decisions.base_loss),)
        or np.asarray(candidate_cost).shape != decisions.candidate_loss.shape
        or not np.isfinite(base_cost).all()
        or not np.isfinite(candidate_cost).all()
    ):
        raise ValueError("terminal-policy cost targets are not aligned")
    state_indices: list[np.ndarray] = [
        np.arange(len(decisions.base_loss), dtype=np.int64)
    ]
    action_indices: list[np.ndarray] = [
        np.zeros(len(decisions.base_loss), dtype=np.int64)
    ]
    targets: list[np.ndarray] = [np.asarray(base_cost, dtype=np.float64)]
    for column in range(decisions.candidate_loss.shape[1]):
        state_indices.append(np.arange(len(decisions.base_loss), dtype=np.int64))
        action_indices.append(decisions.candidate_modality[:, column] + 1)
        targets.append(np.asarray(candidate_cost[:, column], dtype=np.float64))
    return (
        np.concatenate(state_indices),
        np.concatenate(action_indices),
        np.concatenate(targets),
    )


def _fit_action_risk_model(
    decisions: _ActionDecisions,
    base_cost: np.ndarray,
    candidate_cost: np.ndarray,
    *,
    ridge: float,
    maximum_cost: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mean, scale = base._fit_standardizer(decisions.state_features)
    state_indices, action_indices, targets = _risk_targets(
        decisions, base_cost, candidate_cost
    )
    features = _action_design(
        decisions.state_features[state_indices], action_indices, mean, scale
    )
    weights = base._ridge_fit(features, targets, ridge)
    fitted = {
        "mean": mean,
        "scale": scale,
        "weights": weights,
        "maximum_cost": float(maximum_cost),
    }
    raw = _predict_raw_risk(decisions, fitted)
    record = {
        "estimator": "shared_linear_expected_terminal_policy_cost",
        "fit_split": "router_fit",
        "target": "realized_cost_under_calibrated_post_acquisition_terminal_policy",
        "training_decisions": len(decisions.base_loss),
        "training_action_targets": len(targets),
        "feature_width": int(features.shape[1]),
        "state_mean_sha256": _array_sha256(mean),
        "state_scale_sha256": _array_sha256(scale),
        "weights_sha256": _array_sha256(weights),
        "training_risk_sha256": _array_sha256(raw),
    }
    return fitted, record


def _predict_raw_risk(
    decisions: _ActionDecisions,
    model: Mapping[str, Any],
) -> np.ndarray:
    count = len(decisions.base_loss)
    output = np.full((count, 4), np.nan, dtype=np.float64)
    output[:, 0] = np.clip(
        base._linear_predict(
            _action_design(
                decisions.state_features,
                np.zeros(count, dtype=np.int64),
                np.asarray(model["mean"]),
                np.asarray(model["scale"]),
            ),
            np.asarray(model["weights"]),
        ),
        0.0,
        float(model["maximum_cost"]),
    )
    for column in range(decisions.candidate_loss.shape[1]):
        actions = decisions.candidate_modality[:, column] + 1
        predicted = base._linear_predict(
            _action_design(
                decisions.state_features,
                actions,
                np.asarray(model["mean"]),
                np.asarray(model["scale"]),
            ),
            np.asarray(model["weights"]),
        )
        output[np.arange(count), actions] = np.clip(
            predicted, 0.0, float(model["maximum_cost"])
        )
    return output


def _action_target(
    decisions: _ActionDecisions,
    base_cost: np.ndarray,
    candidate_cost: np.ndarray,
    action_index: int,
) -> np.ndarray:
    if action_index == 0:
        return np.asarray(base_cost, dtype=np.float64)
    modality = action_index - 1
    target = np.full(len(decisions.base_loss), np.nan, dtype=np.float64)
    for column in range(decisions.candidate_loss.shape[1]):
        selected = decisions.candidate_modality[:, column] == modality
        target[selected] = candidate_cost[selected, column]
    return target


def _fit_risk_calibrators(
    decisions: _ActionDecisions,
    raw_risk: np.ndarray,
    base_cost: np.ndarray,
    candidate_cost: np.ndarray,
    *,
    penalty: float,
    maximum_cost: float,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, Any]]:
    calibrators: dict[tuple[int, int], np.ndarray] = {}
    records: dict[str, Any] = {}
    for context in np.unique(decisions.context_index):
        context_selected = decisions.context_index == context
        valid_actions = [
            0,
            *(
                int(value) + 1
                for value in decisions.candidate_modality[context_selected][0]
            ),
        ]
        for action in valid_actions:
            target = _action_target(decisions, base_cost, candidate_cost, action)[
                context_selected
            ]
            model = base._ridge_fit(
                raw_risk[context_selected, action, None], target, penalty
            )
            calibrators[(int(context), int(action))] = model
            records[f"{int(context)}:{int(action)}"] = {
                "weights_sha256": _array_sha256(model),
                "examples": int(np.sum(context_selected)),
            }
    calibrated = _predict_calibrated_risk(
        decisions, raw_risk, calibrators, maximum_cost=maximum_cost
    )
    return calibrators, {
        "fit_split": "calibration",
        "estimator": "affine_ridge_cost_calibration",
        "models": records,
        "calibrated_risk_sha256": _array_sha256(calibrated),
    }


def _predict_calibrated_risk(
    decisions: _ActionDecisions,
    raw_risk: np.ndarray,
    calibrators: Mapping[tuple[int, int], np.ndarray],
    *,
    maximum_cost: float,
) -> np.ndarray:
    calibrated = np.full_like(np.asarray(raw_risk, dtype=np.float64), np.nan)
    for context in np.unique(decisions.context_index):
        selected = decisions.context_index == context
        valid_actions = [
            0,
            *(int(value) + 1 for value in decisions.candidate_modality[selected][0]),
        ]
        for action in valid_actions:
            key = (int(context), int(action))
            if key not in calibrators:
                raise ValueError("risk calibrator is missing a valid action")
            calibrated[selected, action] = np.clip(
                base._linear_predict(
                    raw_risk[selected, action, None], np.asarray(calibrators[key])
                ),
                0.0,
                float(maximum_cost),
            )
    return calibrated


def _fit_terminal_correctness(
    decisions: _ActionDecisions,
    *,
    penalty: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit per-mask correctness maps used after the acquisition decision."""

    models = base._fit_confidence_models(decisions, penalty)

    def describe(model: tuple[np.ndarray | None, float]) -> dict[str, Any]:
        weights, rate = model
        return {
            "constant_rate": float(rate),
            "weights_sha256": None if weights is None else _array_sha256(weights),
        }

    model_record = {
        "base": {
            str(context): describe(model)
            for context, model in sorted(models["base"].items())
        },
        "candidate": {
            f"{context}:{modality}": describe(model)
            for (context, modality), model in sorted(models["candidate"].items())
        },
    }
    return models, {
        "fit_split": "calibration",
        "scope": "per_observation_mask_on_calibration_groups",
        "models": model_record,
    }


def _terminal_error_probability(
    decisions: _ActionDecisions,
    models: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    base_correctness, candidate_correctness = base._calibrated_confidence(
        decisions,
        {"confidence_models": models},
    )
    return 1.0 - base_correctness, 1.0 - candidate_correctness


def _cost_values(config: Mapping[str, Any]) -> tuple[float, float, float, np.ndarray]:
    costs = config["action_classifier"]["cost_matrix"]
    return (
        float(costs["correct_answer"]),
        float(costs["incorrect_answer"]),
        float(costs["abstain"]),
        np.asarray(
            [costs["acquisition"][name] for name in MODALITIES], dtype=np.float64
        ),
    )


def _terminal_policy_cost_targets(
    decisions: _ActionDecisions,
    terminal_error: tuple[np.ndarray, np.ndarray],
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Realized costs for the same terminal policy used after acquisition."""

    correct_cost, incorrect_cost, abstain_cost, acquisition_cost = _cost_values(config)
    base_terminal_error, candidate_terminal_error = terminal_error
    if (
        np.asarray(base_terminal_error).shape != (len(decisions.base_loss),)
        or np.asarray(candidate_terminal_error).shape != decisions.candidate_loss.shape
        or not np.isfinite(base_terminal_error).all()
        or not np.isfinite(candidate_terminal_error).all()
    ):
        raise ValueError("terminal error probabilities are not aligned")
    base_answer = (
        correct_cost + base_terminal_error * (incorrect_cost - correct_cost)
        <= abstain_cost
    )
    base_cost = np.where(
        base_answer,
        correct_cost + decisions.base_loss * (incorrect_cost - correct_cost),
        abstain_cost,
    )
    candidate_cost = np.empty_like(decisions.candidate_loss, dtype=np.float64)
    for column in range(decisions.candidate_loss.shape[1]):
        modality = decisions.candidate_modality[:, column]
        candidate_answer = (
            correct_cost
            + candidate_terminal_error[:, column] * (incorrect_cost - correct_cost)
            <= abstain_cost
        )
        terminal_cost = np.where(
            candidate_answer,
            correct_cost
            + decisions.candidate_loss[:, column] * (incorrect_cost - correct_cost),
            abstain_cost,
        )
        candidate_cost[:, column] = acquisition_cost[modality] + terminal_cost
    return np.asarray(base_cost, dtype=np.float64), candidate_cost


def _policy_components(
    decisions: _ActionDecisions,
    calibrated_risk: np.ndarray,
    terminal_error: tuple[np.ndarray, np.ndarray],
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    correct_cost, incorrect_cost, abstain_cost, _ = _cost_values(config)
    base_terminal_error, candidate_terminal_error = terminal_error
    predicted_risk = np.asarray(calibrated_risk, dtype=np.float64)
    if (
        np.asarray(base_terminal_error).shape != (len(decisions.base_loss),)
        or np.asarray(candidate_terminal_error).shape != decisions.candidate_loss.shape
        or not np.isfinite(base_terminal_error).all()
        or not np.isfinite(candidate_terminal_error).all()
        or predicted_risk.shape != (len(decisions.base_loss), 4)
        or not np.isfinite(predicted_risk[:, 0]).all()
    ):
        raise ValueError("terminal policy inputs are not aligned")
    answer_cost = correct_cost + base_terminal_error * (incorrect_cost - correct_cost)
    terminal_answer = answer_cost <= abstain_cost
    no_query_predicted_cost = predicted_risk[:, 0]
    candidate_cost = np.full_like(decisions.candidate_loss, np.inf, dtype=np.float64)
    candidate_answer = np.zeros_like(decisions.candidate_loss, dtype=bool)
    for column in range(decisions.candidate_loss.shape[1]):
        modality = decisions.candidate_modality[:, column]
        risk = predicted_risk[np.arange(len(modality)), modality + 1]
        if not np.isfinite(risk).all():
            raise ValueError("candidate action risk is missing or nonfinite")
        post_query_answer_cost = correct_cost + candidate_terminal_error[:, column] * (
            incorrect_cost - correct_cost
        )
        candidate_answer[:, column] = post_query_answer_cost <= abstain_cost
        candidate_cost[:, column] = risk
    choice_column = np.argmin(candidate_cost, axis=1)
    choice_modality = decisions.candidate_modality[
        np.arange(len(choice_column)), choice_column
    ]
    best_candidate_cost = candidate_cost[np.arange(len(choice_column)), choice_column]
    return {
        "terminal_answer": terminal_answer,
        "no_query_predicted_cost": no_query_predicted_cost,
        "candidate_cost": candidate_cost,
        "choice_column": choice_column,
        "choice_modality": choice_modality,
        "candidate_answer": candidate_answer[
            np.arange(len(choice_column)), choice_column
        ],
        "priority": no_query_predicted_cost - best_candidate_cost,
    }


def _evaluate_actions(
    decisions: _ActionDecisions,
    components: Mapping[str, np.ndarray],
    query: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    selected = np.asarray(query, dtype=bool)
    count = len(decisions.base_loss)
    if selected.shape != (count,):
        raise ValueError("query mask is not aligned with decisions")
    choice_modality = np.asarray(components["choice_modality"], dtype=np.int64)
    terminal_answer = np.asarray(components["terminal_answer"], dtype=bool)
    realized_cost, no_query_realized, final_answer, final_loss = _realized_cost_arrays(
        decisions, components, selected, config
    )
    action_counts = {name: 0 for name in ACTION_ORDER}
    action_counts["answer"] = int(np.sum(~selected & terminal_answer))
    action_counts["abstain"] = int(np.sum(~selected & ~terminal_answer))
    for modality_index, modality in enumerate(MODALITIES):
        action_counts[f"acquire_{modality}"] = int(
            np.sum(selected & (choice_modality == modality_index))
        )
    answered = final_answer
    return {
        "decision_count": int(count),
        "query_count": int(np.sum(selected)),
        "query_rate": float(np.mean(selected)),
        "initial_action_counts": action_counts,
        "final_answer_count": int(np.sum(answered)),
        "final_abstain_count": int(np.sum(~answered)),
        "coverage": float(np.mean(answered)),
        "selective_error": None
        if not np.any(answered)
        else float(np.mean(final_loss[answered])),
        "mean_realized_cost": float(np.mean(realized_cost)),
        "no_query_mean_realized_cost": float(np.mean(no_query_realized)),
        "cost_reduction_vs_no_query": float(np.mean(no_query_realized - realized_cost)),
        "choice_sha256": _array_sha256(choice_modality),
        "query_sha256": _array_sha256(selected.astype(np.float64)),
    }


def _realized_cost_arrays(
    decisions: _ActionDecisions,
    components: Mapping[str, np.ndarray],
    query: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = np.asarray(query, dtype=bool)
    count = len(decisions.base_loss)
    if selected.shape != (count,):
        raise ValueError("query mask is not aligned with decisions")
    choice_column = np.asarray(components["choice_column"], dtype=np.int64)
    choice_modality = np.asarray(components["choice_modality"], dtype=np.int64)
    terminal_answer = np.asarray(components["terminal_answer"], dtype=bool)
    candidate_answer = np.asarray(components["candidate_answer"], dtype=bool)
    final_answer = np.where(selected, candidate_answer, terminal_answer)
    selected_loss = decisions.candidate_loss[np.arange(count), choice_column]
    final_loss = np.where(selected, selected_loss, decisions.base_loss)
    correct_cost, incorrect_cost, abstain_cost, acquisition_cost = _cost_values(config)
    terminal_realized = np.where(
        final_answer,
        correct_cost + final_loss * (incorrect_cost - correct_cost),
        abstain_cost,
    )
    realized_cost = terminal_realized + np.where(
        selected, acquisition_cost[choice_modality], 0.0
    )
    no_query_realized = np.where(
        terminal_answer,
        correct_cost + decisions.base_loss * (incorrect_cost - correct_cost),
        abstain_cost,
    )
    return realized_cost, no_query_realized, final_answer, final_loss


def _cluster_bootstrap_cost_reduction(
    decisions: _ActionDecisions,
    components: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    *,
    query: np.ndarray,
    priority: np.ndarray,
    budget: float,
    exact_budget_reselection: bool,
    tie_seed: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Paired validation-video bootstrap for realized cost reduction."""

    repetitions = int(config["uncertainty"]["bootstrap_repetitions"])
    count = len(decisions.base_loss)
    all_query = np.ones(count, dtype=bool)
    query_cost, no_query_cost, _, _ = _realized_cost_arrays(
        decisions, components, all_query, config
    )
    improvement_if_queried = no_query_cost - query_cost
    unique_groups, inverse = np.unique(decisions.groups, return_inverse=True)
    rng = np.random.default_rng(int(bootstrap_seed))
    sampled = rng.integers(
        0, len(unique_groups), size=(repetitions, len(unique_groups))
    )
    group_multiplicity = np.zeros((repetitions, len(unique_groups)), dtype=np.int32)
    np.add.at(
        group_multiplicity,
        (
            np.repeat(np.arange(repetitions), len(unique_groups)),
            sampled.reshape(-1),
        ),
        1,
    )
    weights = group_multiplicity[:, inverse]
    totals = np.sum(weights, axis=1)
    if exact_budget_reselection:
        values = np.asarray(priority, dtype=np.float64)
        tie_key = np.random.default_rng(int(tie_seed)).random(count)
        order = np.lexsort((tie_key, -values))
        selected_weights, _ = base._weighted_topk_selection(
            weights, order, float(budget)
        )
        method = "video_cluster_resample_with_exact_budget_reselection"
    else:
        selected_weights = weights * np.asarray(query, dtype=np.int32)[None, :]
        method = "video_cluster_resample_with_fixed_calibration_threshold"
    draws = np.sum(selected_weights * improvement_if_queried[None, :], axis=1) / totals
    interval = np.quantile(draws, (0.025, 0.975))
    return {
        "unit": "validation_video_group",
        "method": method,
        "repetitions": repetitions,
        "seed": int(bootstrap_seed),
        "interval_95": [float(interval[0]), float(interval[1])],
        "draws": [float(value) for value in draws],
    }


def _fit_one_family(
    family: str,
    train_z: np.ndarray,
    train_y: np.ndarray,
    train_groups: np.ndarray,
    task_indices: np.ndarray,
    router_indices: np.ndarray,
    calibration_indices: np.ndarray,
    config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], _ActionDecisions]:
    task_config = config["task_heads"]
    task_head, task_record = _fit_task_head(
        family,
        train_z[task_indices],
        train_y[task_indices],
        ridge=float(task_config["ridge"]),
        seed=int(seed),
        dropout_draws=int(task_config["modality_dropout_draws_per_example"]),
    )
    router_decisions = _make_action_decisions(
        train_z[router_indices],
        train_y[router_indices],
        train_groups[router_indices],
        task_head,
    )
    calibration_decisions = _make_action_decisions(
        train_z[calibration_indices],
        train_y[calibration_indices],
        train_groups[calibration_indices],
        task_head,
    )
    terminal_models, terminal_record = _fit_terminal_correctness(
        calibration_decisions,
        penalty=float(config["action_classifier"]["calibration_ridge"]),
    )
    router_terminal_error = _terminal_error_probability(
        router_decisions, terminal_models
    )
    router_base_cost, router_candidate_cost = _terminal_policy_cost_targets(
        router_decisions, router_terminal_error, config
    )
    correct_cost, incorrect_cost, abstain_cost, acquisition_cost = _cost_values(config)
    maximum_cost = max(correct_cost, incorrect_cost, abstain_cost) + float(
        np.max(acquisition_cost)
    )
    action_model, risk_record = _fit_action_risk_model(
        router_decisions,
        router_base_cost,
        router_candidate_cost,
        ridge=float(config["action_classifier"]["risk_ridge"]),
        maximum_cost=maximum_cost,
    )
    calibration_raw = _predict_raw_risk(calibration_decisions, action_model)
    calibration_terminal_error = _terminal_error_probability(
        calibration_decisions,
        terminal_models,
    )
    calibration_base_cost, calibration_candidate_cost = _terminal_policy_cost_targets(
        calibration_decisions, calibration_terminal_error, config
    )
    calibrators, calibration_record = _fit_risk_calibrators(
        calibration_decisions,
        calibration_raw,
        calibration_base_cost,
        calibration_candidate_cost,
        penalty=float(config["action_classifier"]["calibration_ridge"]),
        maximum_cost=maximum_cost,
    )
    calibration_risk = _predict_calibrated_risk(
        calibration_decisions,
        calibration_raw,
        calibrators,
        maximum_cost=maximum_cost,
    )
    calibration_components = _policy_components(
        calibration_decisions,
        calibration_risk,
        calibration_terminal_error,
        config,
    )
    thresholds = {
        f"{float(budget):.6f}": base._fit_online_query_threshold(
            calibration_components["priority"], float(budget)
        )
        for budget in sorted(float(value) for value in config["budgets"])
    }
    fitted = {
        "task_head": task_head,
        "action_model": action_model,
        "calibrators": calibrators,
        "terminal_models": terminal_models,
        "thresholds": thresholds,
    }
    record = {
        "task_head": task_record,
        "action_risk": risk_record,
        "risk_calibration": calibration_record,
        "terminal_correctness_calibration": terminal_record,
        "online_thresholds": thresholds,
    }
    record["training_attestation_sha256"] = _json_sha256(record)
    return fitted, record, calibration_decisions


def _partition_record(
    groups: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(groups)[np.asarray(indices, dtype=np.int64)]
    hashes, digest = base._validation_group_record(selected)
    return {
        "video_groups": len(np.unique(selected)),
        "utterances": len(indices),
        "group_hashes": hashes,
        "group_digest": digest,
        "positive_label_rate": float(np.mean(np.asarray(labels)[indices])),
    }


def run_singleton_baselines(
    cache_path: str | pathlib.Path,
    config: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Fit and evaluate the isolated singleton direct-action extension."""

    _validate_config(config)
    if int(seed) not in config["seeds"]:
        raise ValueError("seed is not declared in the configuration")
    train = _load_split(cache_path, config, "train")
    task_indices, router_indices, calibration_indices = base._partition_indices(
        train.groups,
        config["train_group_fractions"],
        int(seed),
    )
    partitions = {
        "task_fit": task_indices,
        "router_fit": router_indices,
        "calibration": calibration_indices,
    }
    for name, indices in partitions.items():
        if len(np.unique(train.y[indices])) != 2:
            raise ValueError(f"{name} partition must contain both labels")
    sets = [set(train.groups[indices]) for indices in partitions.values()]
    pairwise_disjoint = all(
        not sets[left].intersection(sets[right])
        for left in range(len(sets))
        for right in range(left + 1, len(sets))
    )
    if not pairwise_disjoint:
        raise ValueError("training video groups overlap")
    mean, scale = base._fit_standardizer(train.x[task_indices])
    train_z = (train.x - mean) / scale

    fitted_by_family: dict[str, Any] = {}
    training_by_family: dict[str, Any] = {}
    calibration_by_family: dict[str, _ActionDecisions] = {}
    for family in TASK_HEAD_FAMILIES:
        fitted, record, calibration_decisions = _fit_one_family(
            family,
            train_z,
            train.y,
            train.groups,
            task_indices,
            router_indices,
            calibration_indices,
            config,
            seed=int(seed),
        )
        fitted_by_family[family] = fitted
        training_by_family[family] = record
        calibration_by_family[family] = calibration_decisions

    action_classifier_record = {
        "estimator": "observed_only_linear_expected_terminal_policy_cost",
        "fit_split": "router_fit",
        "calibration_split": "calibration",
        "feature_contract": (
            "observed projected scores, observation mask, observed summaries, "
            "frozen base-task score, and action identity"
        ),
        "decision_contract": {
            "no_query_cost": "calibrated action-risk prediction for answer-or-abstain",
            "acquisition_cost": (
                "calibrated action-risk prediction including acquisition and "
                "post-acquisition answer-or-abstain"
            ),
            "terminal_action": "calibration-only correctness rule",
        },
        "action_order": list(ACTION_ORDER),
        "cost_units": config["action_classifier"]["cost_units"],
        "acquisition_cost_semantics": config["action_classifier"][
            "acquisition_cost_semantics"
        ],
        "cost_matrix": copy.deepcopy(config["action_classifier"]["cost_matrix"]),
        "by_task_head": training_by_family,
        "training_risk_sha256": _json_sha256(
            {
                family: training_by_family[family]["action_risk"][
                    "training_risk_sha256"
                ]
                for family in TASK_HEAD_FAMILIES
            }
        ),
    }
    action_classifier_record["training_attestation_sha256"] = _json_sha256(
        action_classifier_record
    )
    training_attestation = _json_sha256(
        {
            "standardizer_mean_sha256": _array_sha256(mean),
            "standardizer_scale_sha256": _array_sha256(scale),
            "action_classifier": action_classifier_record,
        }
    )

    valid = _load_split(cache_path, config, "valid")
    if set(train.sample_ids).intersection(valid.sample_ids) or set(
        train.groups
    ).intersection(valid.groups):
        raise ValueError("training and validation sources overlap")
    valid_z = (valid.x - mean) / scale
    sorted_budgets = sorted(float(value) for value in config["budgets"])
    family_results: dict[str, Any] = {}
    for family_index, family in enumerate(TASK_HEAD_FAMILIES):
        fitted = fitted_by_family[family]
        task_head = fitted["task_head"]
        validation_accuracy_by_mask = {
            base._mask_name(mask): float(
                np.mean(
                    (_task_scores(valid_z, mask, task_head) >= 0).astype(np.int64)
                    == valid.y
                )
            )
            for mask in base._observed_masks()
        }
        minimum_accuracy = float(
            config["task_heads"]["minimum_validation_accuracy_per_mask"]
        )
        failed_masks = sorted(
            name
            for name, accuracy in validation_accuracy_by_mask.items()
            if accuracy + 1e-12 < minimum_accuracy
        )
        evaluation = _make_action_decisions(valid_z, valid.y, valid.groups, task_head)
        raw_risk = _predict_raw_risk(evaluation, fitted["action_model"])
        calibrated_risk = _predict_calibrated_risk(
            evaluation,
            raw_risk,
            fitted["calibrators"],
            maximum_cost=float(fitted["action_model"]["maximum_cost"]),
        )
        terminal_error = _terminal_error_probability(
            evaluation,
            fitted["terminal_models"],
        )
        components = _policy_components(
            evaluation,
            calibrated_risk,
            terminal_error,
            config,
        )
        exact_budget: dict[str, Any] = {}
        online_threshold: dict[str, Any] = {}
        for budget_index, budget in enumerate(sorted_budgets):
            key = f"{budget:.6f}"
            tie_seed = int(seed) * 1009 + budget_index * 97
            query = _select_exact_budget(
                components["priority"],
                budget,
                seed=tie_seed,
            )
            exact_metrics = _evaluate_actions(evaluation, components, query, config)
            exact_metrics["paired_group_bootstrap"] = _cluster_bootstrap_cost_reduction(
                evaluation,
                components,
                config,
                query=query,
                priority=components["priority"],
                budget=budget,
                exact_budget_reselection=True,
                tie_seed=tie_seed,
                bootstrap_seed=(
                    int(config["uncertainty"]["bootstrap_seed"])
                    + family_index * 1000
                    + budget_index * 2
                ),
            )
            exact_budget[key] = exact_metrics
            threshold_record = fitted["thresholds"][key]
            online_query = base._apply_online_query_threshold(
                components["priority"],
                threshold_record,
            )
            online_metrics = _evaluate_actions(
                evaluation, components, online_query, config
            )
            online_metrics["target_query_rate"] = float(budget)
            online_metrics["threshold"] = float(threshold_record["threshold"])
            online_metrics["query_on_equal"] = bool(threshold_record["query_on_equal"])
            online_metrics["threshold_fit_split"] = "calibration"
            online_metrics["paired_group_bootstrap"] = (
                _cluster_bootstrap_cost_reduction(
                    evaluation,
                    components,
                    config,
                    query=online_query,
                    priority=components["priority"],
                    budget=budget,
                    exact_budget_reselection=False,
                    tie_seed=tie_seed,
                    bootstrap_seed=(
                        int(config["uncertainty"]["bootstrap_seed"])
                        + family_index * 1000
                        + budget_index * 2
                        + 1
                    ),
                )
            )
            online_threshold[key] = online_metrics
        family_results[family] = {
            "validation_accuracy_by_mask": validation_accuracy_by_mask,
            "full_avt_validation_accuracy": validation_accuracy_by_mask[
                "audio+video+text"
            ],
            "validation_accuracy_gate": {
                "minimum_per_mask": minimum_accuracy,
                "failed_masks": failed_masks,
                "passed": not failed_masks,
            },
            "exact_budget": exact_budget,
            "online_threshold": online_threshold,
            "evaluation_risk_sha256": _array_sha256(calibrated_risk),
        }

    primary_budget = float(config["primary_budget"])
    primary_key = f"{primary_budget:.6f}"
    reductions = {
        family: float(
            family_results[family]["exact_budget"][primary_key][
                "cost_reduction_vs_no_query"
            ]
        )
        for family in TASK_HEAD_FAMILIES
    }
    signs = {family: int(np.sign(value)) for family, value in reductions.items()}
    matching_sign = len(set(signs.values())) == 1
    spread = float(max(reductions.values()) - min(reductions.values()))
    sensitivity_config = config["task_head_sensitivity"]
    range_passed = spread <= float(
        sensitivity_config["maximum_primary_cost_reduction_range"]
    )
    sign_passed = matching_sign or not bool(sensitivity_config["require_matching_sign"])
    task_head_validation_passed = all(
        family_results[family]["validation_accuracy_gate"]["passed"]
        for family in TASK_HEAD_FAMILIES
    )
    agreement_passed = bool(range_passed and sign_passed)
    task_head_sensitivity = {
        "primary_budget": primary_budget,
        "cost_reduction_vs_no_query_by_task_head": reductions,
        "sign_by_task_head": signs,
        "matching_sign": bool(matching_sign),
        "observed_range": spread,
        "maximum_allowed_range": float(
            sensitivity_config["maximum_primary_cost_reduction_range"]
        ),
        "agreement_passed": agreement_passed,
        "task_head_validation_passed": bool(task_head_validation_passed),
        "passed": bool(agreement_passed and task_head_validation_passed),
    }
    primary_intervals = {
        family: family_results[family]["exact_budget"][primary_key][
            "paired_group_bootstrap"
        ]["interval_95"]
        for family in TASK_HEAD_FAMILIES
    }
    point_positive = {family: reductions[family] > 0.0 for family in TASK_HEAD_FAMILIES}
    interval_positive = {
        family: float(primary_intervals[family][0]) > 0.0
        for family in TASK_HEAD_FAMILIES
    }
    interval_negative = {
        family: float(primary_intervals[family][1]) < 0.0
        for family in TASK_HEAD_FAMILIES
    }
    positive_supported = bool(
        task_head_sensitivity["passed"] and all(interval_positive.values())
    )
    negative_supported = bool(
        task_head_sensitivity["passed"] and all(interval_negative.values())
    )
    if positive_supported:
        interpretation_status = "positive_acquisition_supported"
    elif negative_supported:
        interpretation_status = "negative_acquisition_supported"
    else:
        interpretation_status = "inconclusive"
    acquisition_evidence = {
        "primary_budget": primary_budget,
        "cost_reduction_vs_no_query_by_task_head": reductions,
        "interval_95_by_task_head": primary_intervals,
        "point_positive_by_task_head": point_positive,
        "interval_positive_by_task_head": interval_positive,
        "interval_negative_by_task_head": interval_negative,
        "positive_acquisition_supported": positive_supported,
        "negative_acquisition_supported": negative_supported,
        "status": interpretation_status,
    }

    partition_records = {
        name: _partition_record(train.groups, train.y, indices)
        for name, indices in partitions.items()
    }
    partition_records["pairwise_disjoint"] = True
    source = pathlib.Path(cache_path).expanduser().resolve()
    training_group_hashes, training_group_digest = base._validation_group_record(
        train.groups
    )
    validation_group_hashes, validation_group_digest = base._validation_group_record(
        valid.groups
    )
    result = {
        "schema": RESULT_SCHEMA,
        "study_schema": STUDY_SCHEMA,
        "configuration_sha256": configuration_sha256(config),
        "implementation_sha256": implementation_sha256(),
        "mode": "singleton",
        "seed": int(seed),
        "runtime": {
            "python": ".".join(str(value) for value in sys.version_info[:3]),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "data_access": {
            "cache_sha256": _file_sha256(source),
            "loaded_splits": ["train", "valid"],
            "test_opened": False,
            "train_utterances": len(train.y),
            "validation_utterances": len(valid.y),
            "train_video_groups": len(np.unique(train.groups)),
            "validation_video_groups": len(np.unique(valid.groups)),
            "training_group_hashes": training_group_hashes,
            "training_group_sha256": training_group_digest,
            "validation_group_hashes": validation_group_hashes,
            "validation_group_sha256": validation_group_digest,
        },
        "group_partitions": partition_records,
        "action_classifier": action_classifier_record,
        "task_heads": family_results,
        "task_head_sensitivity": task_head_sensitivity,
        "acquisition_evidence": acquisition_evidence,
        "published_comparator": copy.deepcopy(config["published_comparator"]),
        "training_attestation_sha256": training_attestation,
        "validation": {"ok": True, "errors": []},
    }
    return result


def _validate_bootstrap_record(
    value: Any,
    *,
    repetitions: int,
    seed: int,
    method: str,
) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "unit",
        "method",
        "repetitions",
        "seed",
        "interval_95",
        "draws",
    }:
        return False
    try:
        draws = np.asarray(value["draws"], dtype=np.float64)
        interval = np.asarray(value["interval_95"], dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if (
        value.get("unit") != "validation_video_group"
        or value.get("method") != method
        or value.get("repetitions") != repetitions
        or value.get("seed") != seed
        or draws.shape != (repetitions,)
        or interval.shape != (2,)
        or not np.isfinite(draws).all()
        or not np.isfinite(interval).all()
    ):
        return False
    return bool(np.allclose(interval, np.quantile(draws, (0.025, 0.975))))


def validate_singleton_baseline_result(
    record: Mapping[str, Any],
    *,
    expected_seed: int | None = None,
    expected_cache_sha256: str | None = None,
    expected_config: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return structural validation errors for one v3 singleton result."""

    errors: list[str] = []
    if (
        record.get("schema") != RESULT_SCHEMA
        or record.get("study_schema") != STUDY_SCHEMA
    ):
        errors.append("schema is invalid")
    if record.get("mode") != "singleton":
        errors.append("mode is invalid")
    if record.get("implementation_sha256") != implementation_sha256():
        errors.append("implementation digest is invalid")
    expected_runtime = {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
    if record.get("runtime") != expected_runtime:
        errors.append("runtime lock is invalid")
    if expected_seed is not None and record.get("seed") != int(expected_seed):
        errors.append("seed is invalid")
    if expected_config is not None:
        try:
            _validate_config(expected_config)
        except (TypeError, ValueError):
            errors.append("expected configuration is invalid")
        else:
            if record.get("configuration_sha256") != configuration_sha256(
                expected_config
            ):
                errors.append("configuration digest is invalid")
    data_access = record.get("data_access")
    data_hashes: dict[str, set[str]] = {}
    data_access_valid = isinstance(data_access, Mapping)
    if not isinstance(data_access, Mapping):
        errors.append("data access record is invalid")
    else:
        if (
            data_access.get("loaded_splits") != ["train", "valid"]
            or data_access.get("test_opened") is not False
        ):
            data_access_valid = False
        if not _is_sha256(data_access.get("cache_sha256")):
            data_access_valid = False
        if (
            expected_cache_sha256 is not None
            and data_access.get("cache_sha256") != expected_cache_sha256
        ):
            errors.append("cache digest is invalid")
        for prefix in ("training", "validation"):
            hashes = data_access.get(f"{prefix}_group_hashes")
            count = data_access.get(
                "train_video_groups"
                if prefix == "training"
                else "validation_video_groups"
            )
            digest = data_access.get(f"{prefix}_group_sha256")
            if (
                not isinstance(hashes, list)
                or not hashes
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count != len(hashes)
                or len(set(hashes)) != len(hashes)
                or hashes != sorted(hashes)
                or any(not _is_sha256(value) for value in hashes)
                or digest != _hash_list_digest(hashes)
            ):
                data_access_valid = False
            else:
                data_hashes[prefix] = set(hashes)
        for count_name in ("train_utterances", "validation_utterances"):
            count = data_access.get(count_name)
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                data_access_valid = False
        if set(data_hashes) == {"training", "validation"} and data_hashes[
            "training"
        ].intersection(data_hashes["validation"]):
            data_access_valid = False
        if not data_access_valid:
            errors.append("data split access is invalid")
    partitions = record.get("group_partitions")
    partition_valid = (
        isinstance(partitions, Mapping)
        and set(partitions)
        == {"task_fit", "router_fit", "calibration", "pairwise_disjoint"}
        and partitions.get("pairwise_disjoint") is True
    )
    partition_hashes: dict[str, set[str]] = {}
    partition_utterances = 0
    if isinstance(partitions, Mapping):
        for name in ("task_fit", "router_fit", "calibration"):
            item = partitions.get(name)
            if not isinstance(item, Mapping):
                partition_valid = False
                continue
            hashes = item.get("group_hashes")
            group_count = item.get("video_groups")
            utterances = item.get("utterances")
            rate = item.get("positive_label_rate")
            if (
                not isinstance(hashes, list)
                or not hashes
                or hashes != sorted(hashes)
                or len(set(hashes)) != len(hashes)
                or any(not _is_sha256(value) for value in hashes)
                or isinstance(group_count, bool)
                or not isinstance(group_count, int)
                or group_count != len(hashes)
                or isinstance(utterances, bool)
                or not isinstance(utterances, int)
                or utterances < group_count
                or not isinstance(rate, (int, float))
                or isinstance(rate, bool)
                or not np.isfinite(rate)
                or not 0.0 < float(rate) < 1.0
                or item.get("group_digest") != _hash_list_digest(hashes)
            ):
                partition_valid = False
                continue
            partition_hashes[name] = set(hashes)
            partition_utterances += utterances
    if set(partition_hashes) == {"task_fit", "router_fit", "calibration"}:
        names = tuple(partition_hashes)
        if any(
            partition_hashes[names[left]].intersection(partition_hashes[names[right]])
            for left in range(len(names))
            for right in range(left + 1, len(names))
        ):
            partition_valid = False
        union = set().union(*partition_hashes.values())
        if data_hashes.get("training") != union:
            partition_valid = False
        if isinstance(data_access, Mapping) and partition_utterances != data_access.get(
            "train_utterances"
        ):
            partition_valid = False
    else:
        partition_valid = False
    if not partition_valid:
        errors.append("group partitions are invalid")

    classifier = record.get("action_classifier")
    if not isinstance(classifier, Mapping):
        errors.append("action classifier record is invalid")
    else:
        try:
            attestation = dict(classifier)
            saved = attestation.pop("training_attestation_sha256")
            if (
                saved != _json_sha256(attestation)
                or re.fullmatch(
                    r"[0-9a-f]{64}", str(classifier["training_risk_sha256"])
                )
                is None
                or classifier.get("fit_split") != "router_fit"
                or classifier.get("calibration_split") != "calibration"
                or tuple(classifier.get("action_order", ())) != ACTION_ORDER
                or classifier.get("decision_contract")
                != {
                    "no_query_cost": (
                        "calibrated action-risk prediction for answer-or-abstain"
                    ),
                    "acquisition_cost": (
                        "calibrated action-risk prediction including acquisition and "
                        "post-acquisition answer-or-abstain"
                    ),
                    "terminal_action": "calibration-only correctness rule",
                }
                or (
                    expected_config is not None
                    and (
                        classifier.get("cost_matrix")
                        != expected_config["action_classifier"]["cost_matrix"]
                        or classifier.get("cost_units")
                        != expected_config["action_classifier"]["cost_units"]
                        or classifier.get("acquisition_cost_semantics")
                        != expected_config["action_classifier"][
                            "acquisition_cost_semantics"
                        ]
                    )
                )
            ):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            errors.append("training provenance is invalid")

    comparator = record.get("published_comparator")
    if (
        not isinstance(comparator, Mapping)
        or comparator.get("status") != "excluded_incompatible"
        or comparator.get("benchmark_only") is not True
        or comparator.get("selected") is not None
    ):
        errors.append("published comparator decision is invalid")

    config_budgets = (
        None
        if expected_config is None
        else sorted(float(value) for value in expected_config["budgets"])
    )
    primary_reductions: dict[str, float] = {}
    primary_intervals: dict[str, list[float]] = {}
    task_head_gate_passes: dict[str, bool] = {}
    task_heads = record.get("task_heads")
    if not isinstance(task_heads, Mapping) or set(task_heads) != set(
        TASK_HEAD_FAMILIES
    ):
        errors.append("task-head results are invalid")
    elif config_budgets is not None and isinstance(data_access, Mapping):
        decision_count = int(data_access.get("validation_utterances", 0)) * 3
        expected_masks = {base._mask_name(mask) for mask in base._observed_masks()}
        minimum_accuracy = float(
            expected_config["task_heads"]["minimum_validation_accuracy_per_mask"]
        )
        primary_key = f"{float(expected_config['primary_budget']):.6f}"
        for family_index, family in enumerate(TASK_HEAD_FAMILIES):
            family_result = task_heads.get(family)
            if not isinstance(family_result, Mapping):
                errors.append("task-head results are invalid")
                continue
            accuracies = family_result.get("validation_accuracy_by_mask")
            gate = family_result.get("validation_accuracy_gate")
            if not isinstance(accuracies, Mapping) or set(accuracies) != expected_masks:
                errors.append("task-head validation accuracy is invalid")
            else:
                try:
                    numeric_accuracies = {
                        str(name): float(value) for name, value in accuracies.items()
                    }
                except (TypeError, ValueError):
                    numeric_accuracies = {}
                if not numeric_accuracies or any(
                    not np.isfinite(value) or not 0.0 <= value <= 1.0
                    for value in numeric_accuracies.values()
                ):
                    errors.append("task-head validation accuracy is invalid")
                else:
                    failed_masks = sorted(
                        name
                        for name, value in numeric_accuracies.items()
                        if value + 1e-12 < minimum_accuracy
                    )
                    expected_gate = {
                        "minimum_per_mask": minimum_accuracy,
                        "failed_masks": failed_masks,
                        "passed": not failed_masks,
                    }
                    if gate != expected_gate:
                        errors.append("task-head validation gate is invalid")
                    else:
                        task_head_gate_passes[family] = bool(gate["passed"])
                    if not np.isclose(
                        float(
                            family_result.get("full_avt_validation_accuracy", np.nan)
                        ),
                        numeric_accuracies["audio+video+text"],
                    ):
                        errors.append("task-head validation accuracy is invalid")
            exact = family_result.get("exact_budget")
            online = family_result.get("online_threshold")
            expected_keys = {f"{budget:.6f}" for budget in config_budgets}
            if (
                not isinstance(exact, Mapping)
                or set(exact) != expected_keys
                or not isinstance(online, Mapping)
                or set(online) != expected_keys
            ):
                errors.append("budget results are invalid")
                continue
            for budget in config_budgets:
                key = f"{budget:.6f}"
                metrics = exact[key]
                expected_count = int(np.rint(budget * decision_count))
                counts = (
                    metrics.get("initial_action_counts")
                    if isinstance(metrics, Mapping)
                    else None
                )
                if (
                    not isinstance(metrics, Mapping)
                    or metrics.get("decision_count") != decision_count
                    or metrics.get("query_count") != expected_count
                    or not isinstance(counts, Mapping)
                    or set(counts) != set(ACTION_ORDER)
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                        for value in counts.values()
                    )
                    or sum(counts.values()) != decision_count
                    or sum(counts[f"acquire_{name}"] for name in MODALITIES)
                    != expected_count
                    or not np.isclose(
                        float(metrics.get("query_rate", np.nan)),
                        expected_count / decision_count,
                    )
                    or metrics.get("final_answer_count", -1)
                    + metrics.get("final_abstain_count", -1)
                    != decision_count
                ):
                    errors.append("exact query budget is invalid")
                    break
                exact_bootstrap = metrics.get("paired_group_bootstrap")
                expected_repetitions = int(
                    expected_config["uncertainty"]["bootstrap_repetitions"]
                )
                expected_exact_seed = (
                    int(expected_config["uncertainty"]["bootstrap_seed"])
                    + family_index * 1000
                    + config_budgets.index(budget) * 2
                )
                if not _validate_bootstrap_record(
                    exact_bootstrap,
                    repetitions=expected_repetitions,
                    seed=expected_exact_seed,
                    method="video_cluster_resample_with_exact_budget_reselection",
                ):
                    errors.append("exact-budget bootstrap is invalid")
                    break
                online_metrics = online[key]
                online_counts = (
                    online_metrics.get("initial_action_counts")
                    if isinstance(online_metrics, Mapping)
                    else None
                )
                if (
                    not isinstance(online_metrics, Mapping)
                    or online_metrics.get("target_query_rate") != budget
                    or online_metrics.get("threshold_fit_split") != "calibration"
                    or online_metrics.get("decision_count") != decision_count
                    or not isinstance(online_counts, Mapping)
                    or set(online_counts) != set(ACTION_ORDER)
                    or sum(online_counts.values()) != decision_count
                    or sum(online_counts[f"acquire_{name}"] for name in MODALITIES)
                    != online_metrics.get("query_count")
                ):
                    errors.append("online threshold result is invalid")
                    break
                if not _validate_bootstrap_record(
                    online_metrics.get("paired_group_bootstrap"),
                    repetitions=expected_repetitions,
                    seed=expected_exact_seed + 1,
                    method="video_cluster_resample_with_fixed_calibration_threshold",
                ):
                    errors.append("online-threshold bootstrap is invalid")
                    break
            if isinstance(exact, Mapping) and isinstance(
                exact.get(primary_key), Mapping
            ):
                try:
                    primary_reductions[family] = float(
                        exact[primary_key]["cost_reduction_vs_no_query"]
                    )
                    primary_intervals[family] = [
                        float(value)
                        for value in exact[primary_key]["paired_group_bootstrap"][
                            "interval_95"
                        ]
                    ]
                except (KeyError, TypeError, ValueError):
                    pass
    sensitivity = record.get("task_head_sensitivity")
    sensitivity_valid = (
        isinstance(sensitivity, Mapping)
        and set(primary_reductions) == set(TASK_HEAD_FAMILIES)
        and set(primary_intervals) == set(TASK_HEAD_FAMILIES)
        and set(task_head_gate_passes) == set(TASK_HEAD_FAMILIES)
    )
    if sensitivity_valid and expected_config is not None:
        signs = {
            family: int(np.sign(value)) for family, value in primary_reductions.items()
        }
        matching_sign = len(set(signs.values())) == 1
        spread = float(
            max(primary_reductions.values()) - min(primary_reductions.values())
        )
        maximum_range = float(
            expected_config["task_head_sensitivity"][
                "maximum_primary_cost_reduction_range"
            ]
        )
        sign_passed = matching_sign or not bool(
            expected_config["task_head_sensitivity"]["require_matching_sign"]
        )
        agreement_passed = bool(spread <= maximum_range and sign_passed)
        task_validation_passed = all(task_head_gate_passes.values())
        try:
            sensitivity_valid = bool(
                sensitivity.get("primary_budget")
                == float(expected_config["primary_budget"])
                and set(sensitivity.get("cost_reduction_vs_no_query_by_task_head", {}))
                == set(TASK_HEAD_FAMILIES)
                and all(
                    np.isclose(
                        float(
                            sensitivity["cost_reduction_vs_no_query_by_task_head"][
                                family
                            ]
                        ),
                        primary_reductions[family],
                    )
                    for family in TASK_HEAD_FAMILIES
                )
                and sensitivity.get("sign_by_task_head") == signs
                and sensitivity.get("matching_sign") is matching_sign
                and np.isclose(float(sensitivity.get("observed_range", np.nan)), spread)
                and sensitivity.get("maximum_allowed_range") == maximum_range
                and sensitivity.get("agreement_passed") is agreement_passed
                and sensitivity.get("task_head_validation_passed")
                is task_validation_passed
                and sensitivity.get("passed")
                is bool(agreement_passed and task_validation_passed)
            )
        except (KeyError, TypeError, ValueError):
            sensitivity_valid = False
    if not sensitivity_valid:
        errors.append("task-head sensitivity result is invalid")

    evidence = record.get("acquisition_evidence")
    evidence_valid = (
        isinstance(evidence, Mapping)
        and sensitivity_valid
        and set(primary_reductions) == set(TASK_HEAD_FAMILIES)
        and set(primary_intervals) == set(TASK_HEAD_FAMILIES)
    )
    if evidence_valid and expected_config is not None:
        point_positive = {
            family: primary_reductions[family] > 0.0 for family in TASK_HEAD_FAMILIES
        }
        interval_positive = {
            family: primary_intervals[family][0] > 0.0 for family in TASK_HEAD_FAMILIES
        }
        interval_negative = {
            family: primary_intervals[family][1] < 0.0 for family in TASK_HEAD_FAMILIES
        }
        positive_supported = bool(
            sensitivity["passed"] and all(interval_positive.values())
        )
        negative_supported = bool(
            sensitivity["passed"] and all(interval_negative.values())
        )
        status = (
            "positive_acquisition_supported"
            if positive_supported
            else "negative_acquisition_supported"
            if negative_supported
            else "inconclusive"
        )
        expected_evidence = {
            "primary_budget": float(expected_config["primary_budget"]),
            "cost_reduction_vs_no_query_by_task_head": primary_reductions,
            "interval_95_by_task_head": primary_intervals,
            "point_positive_by_task_head": point_positive,
            "interval_positive_by_task_head": interval_positive,
            "interval_negative_by_task_head": interval_negative,
            "positive_acquisition_supported": positive_supported,
            "negative_acquisition_supported": negative_supported,
            "status": status,
        }
        evidence_valid = evidence == expected_evidence
    if not evidence_valid:
        errors.append("acquisition evidence result is invalid")
    validation = record.get("validation")
    if validation != {"ok": True, "errors": []}:
        errors.append("validation status is invalid")
    return errors


def aggregate_singleton_baseline_results(
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and combine the complete predeclared five-seed result set."""

    _validate_config(config)
    expected_seeds = [int(value) for value in config["seeds"]]
    if len(records) != len(expected_seeds):
        raise ValueError("aggregate requires exactly the configured result count")
    by_seed: dict[int, Mapping[str, Any]] = {}
    expected_cache_sha256 = str(config["expected_cache_sha256"])
    for record in records:
        seed = record.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed in by_seed:
            raise ValueError("aggregate result seeds are invalid or duplicated")
        errors = validate_singleton_baseline_result(
            record,
            expected_seed=seed,
            expected_cache_sha256=expected_cache_sha256,
            expected_config=config,
        )
        if errors:
            raise ValueError(f"seed {seed} result is invalid: {errors}")
        by_seed[seed] = record
    if sorted(by_seed) != sorted(expected_seeds):
        raise ValueError("aggregate seed set differs from the configuration")

    primary_budget = float(config["primary_budget"])
    key = f"{primary_budget:.6f}"
    families: dict[str, Any] = {}
    for family in TASK_HEAD_FAMILIES:
        family_record: dict[str, Any] = {}
        for policy_name, result_key in (
            ("exact_budget", "exact_budget"),
            ("online_threshold", "online_threshold"),
        ):
            values = np.asarray(
                [
                    by_seed[seed]["task_heads"][family][result_key][key][
                        "cost_reduction_vs_no_query"
                    ]
                    for seed in expected_seeds
                ],
                dtype=np.float64,
            )
            draws = np.asarray(
                [
                    by_seed[seed]["task_heads"][family][result_key][key][
                        "paired_group_bootstrap"
                    ]["draws"]
                    for seed in expected_seeds
                ],
                dtype=np.float64,
            )
            if draws.ndim != 2 or draws.shape[0] != len(expected_seeds):
                raise ValueError("aggregate bootstrap arrays are not aligned")
            averaged_draws = np.mean(draws, axis=0)
            interval = np.quantile(averaged_draws, (0.025, 0.975))
            family_record[policy_name] = {
                "seed_values": {
                    str(seed): float(value)
                    for seed, value in zip(expected_seeds, values)
                },
                "mean_cost_reduction_vs_no_query": float(np.mean(values)),
                "minimum_seed_value": float(np.min(values)),
                "maximum_seed_value": float(np.max(values)),
                "all_seeds_positive": bool(np.all(values > 0.0)),
                "all_seeds_negative": bool(np.all(values < 0.0)),
                "paired_group_interval_95": [
                    float(interval[0]),
                    float(interval[1]),
                ],
                "paired_group_mean_draws": [float(value) for value in averaged_draws],
            }
        family_record["all_task_head_quality_gates_passed"] = all(
            bool(
                by_seed[seed]["task_heads"][family]["validation_accuracy_gate"][
                    "passed"
                ]
            )
            for seed in expected_seeds
        )
        families[family] = family_record

    all_quality = all(
        families[family]["all_task_head_quality_gates_passed"]
        for family in TASK_HEAD_FAMILIES
    )
    sensitivity_passed_by_seed = {
        str(seed): bool(by_seed[seed]["task_head_sensitivity"]["passed"])
        for seed in expected_seeds
    }
    all_sensitivity = all(sensitivity_passed_by_seed.values())
    positive_supported = bool(
        all_quality
        and all_sensitivity
        and all(
            families[family]["exact_budget"]["all_seeds_positive"]
            and families[family]["exact_budget"]["paired_group_interval_95"][0] > 0.0
            for family in TASK_HEAD_FAMILIES
        )
    )
    negative_supported = bool(
        all_quality
        and all_sensitivity
        and all(
            families[family]["exact_budget"]["all_seeds_negative"]
            and families[family]["exact_budget"]["paired_group_interval_95"][1] < 0.0
            for family in TASK_HEAD_FAMILIES
        )
    )
    status = (
        "positive_acquisition_supported"
        if positive_supported
        else "negative_acquisition_supported"
        if negative_supported
        else "inconclusive"
    )
    aggregate = {
        "schema": "conflictbench.mosei-singleton-baseline-aggregate.v3",
        "configuration_sha256": configuration_sha256(config),
        "cache_sha256": expected_cache_sha256,
        "implementation_sha256": implementation_sha256(),
        "runtime": dict(config["runtime"]),
        "seeds": expected_seeds,
        "result_digests": {
            str(seed): _json_sha256(by_seed[seed]) for seed in expected_seeds
        },
        "primary_budget": primary_budget,
        "task_heads": families,
        "decision": {
            "task_head_sensitivity_passed_by_seed": sensitivity_passed_by_seed,
            "all_task_head_sensitivity_passed": all_sensitivity,
            "positive_acquisition_supported": positive_supported,
            "negative_acquisition_supported": negative_supported,
            "status": status,
        },
    }
    aggregate["attestation_sha256"] = _json_sha256(aggregate)
    return aggregate
