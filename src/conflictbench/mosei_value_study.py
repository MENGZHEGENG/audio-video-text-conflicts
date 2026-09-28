"""Leakage-audited acquisition-value study for projected CMU-MOSEI scores.

The pilot reads the official training and validation arrays only.  The test
arrays stay unopened so the test fold cannot affect model choice, thresholds,
or reported pilot results.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor

from .value_of_information import (
    masked_observations,
    realized_acquisition_value,
    source_group_split,
)

STUDY_SCHEMA = "conflictbench.mosei-value-study.v2"
RESULT_SCHEMA = "conflictbench.mosei-value-result.v2"
CACHE_SCHEMA = "conflictbench.mosei-cache.v2"
MODALITIES = ("audio", "video", "text")
ROUTER_FAMILIES = ("linear_value", "hist_gbt_value", "shallow_mlp_value")
ROUTER_POLICIES = (
    "confidence",
    *ROUTER_FAMILIES,
    "selected_router",
    "online_selected_router",
    "oracle",
    "random_matched",
    "source_prior",
    "selected_baseline",
    "selected_router_query_source_prior_choice",
    "confidence_query_selected_router_choice",
)
BASE_BOOTSTRAP_POLICIES = (
    "confidence",
    *ROUTER_FAMILIES,
    "oracle",
    "random_matched",
    "source_prior",
    "selected_router_query_source_prior_choice",
    "confidence_query_selected_router_choice",
)
POLICY_SEED_INDEX = {name: index for index, name in enumerate(BASE_BOOTSTRAP_POLICIES)}


@dataclass(frozen=True)
class StudySplit:
    """One score split with the independent video-group identifier."""

    x: np.ndarray
    y: np.ndarray
    sample_ids: tuple[str, ...]
    groups: np.ndarray


@dataclass(frozen=True)
class _DecisionSet:
    labels: np.ndarray
    base_loss: np.ndarray
    base_confidence: np.ndarray
    candidate_loss: np.ndarray
    candidate_confidence: np.ndarray
    value: np.ndarray
    router_features: np.ndarray
    context_index: np.ndarray
    candidate_modality: np.ndarray
    groups: np.ndarray


def _scalar(value: Any) -> Any:
    return np.asarray(value).reshape(()).item()


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configuration_sha256(config: Mapping[str, Any]) -> str:
    """Hash the semantic JSON configuration with stable serialization."""

    encoded = json.dumps(dict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def implementation_sha256() -> str:
    """Hash the scientific implementation and its run/verification entry points."""

    reproducibility_root = pathlib.Path(__file__).resolve().parents[2]
    paths = (
        pathlib.Path(__file__).resolve(),
        pathlib.Path(__file__).with_name("value_of_information.py").resolve(),
        reproducibility_root / "pyproject.toml",
        reproducibility_root / "scripts" / "run_mosei_value_pilot.py",
        reproducibility_root / "scripts" / "verify_mosei_value.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(reproducibility_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def video_group_id(sample_id: str) -> str:
    """Return the official video identifier for a segment identifier."""

    match = re.fullmatch(r"(.+)\[(\d+)\]", str(sample_id))
    if match is None:
        raise ValueError(f"invalid utterance identifier: {sample_id!r}")
    return match.group(1)


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != STUDY_SCHEMA:
        raise ValueError(f"study configuration schema must be {STUDY_SCHEMA}")
    fractions = np.asarray(config.get("train_group_fractions", ()), dtype=np.float64)
    if fractions.shape != (3,) or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("train_group_fractions must be three positive values summing to one")
    budgets = np.asarray(config.get("budgets", ()), dtype=np.float64)
    if budgets.ndim != 1 or len(budgets) == 0 or np.any((budgets < 0) | (budgets > 1)):
        raise ValueError("budgets must be a nonempty list within [0, 1]")
    if len(np.unique(budgets)) != len(budgets):
        raise ValueError("budgets must be unique")
    for name in ("headroom_budget", "primary_budget"):
        value = config.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or not np.any(np.isclose(budgets, float(value)))
        ):
            raise ValueError(f"{name} must be one of the configured budgets")
    repetitions = int(config.get("bootstrap_repetitions", 0))
    if repetitions <= 0:
        raise ValueError("bootstrap_repetitions must be positive")
    coverage = np.asarray(config.get("coverage_targets", (1.0, 0.9, 0.8)), dtype=np.float64)
    if coverage.ndim != 1 or len(coverage) == 0 or np.any((coverage <= 0) | (coverage > 1)):
        raise ValueError("coverage_targets must be a nonempty list within (0, 1]")
    if 1.0 not in coverage or 0.9 not in coverage or len(np.unique(coverage)) != len(coverage):
        raise ValueError("coverage_targets must uniquely contain 1.0 and 0.9")
    bootstrap_seed = config.get("bootstrap_seed", 20_270_917)
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    for name in (
        "ridge",
        "benefit_ridge",
        "confidence_ridge",
        "minimum_oracle_error_reduction",
        "minimum_primary_error_reduction_gain",
        "minimum_video_macro_gain_fraction",
        "maximum_start_context_harm",
    ):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name in (
        "minimum_oracle_error_reduction",
        "minimum_primary_error_reduction_gain",
        "minimum_video_macro_gain_fraction",
        "maximum_start_context_harm",
    ):
        if float(config[name]) > 1:
            raise ValueError(f"{name} must not exceed one")
    nonnegative_contexts = config.get("minimum_nonnegative_start_contexts")
    if (
        isinstance(nonnegative_contexts, bool)
        or not isinstance(nonnegative_contexts, int)
        or not 0 <= nonnegative_contexts <= 3
    ):
        raise ValueError("minimum_nonnegative_start_contexts must be an integer within [0, 3]")
    reference_accuracy = config.get("reference_full_avt_validation_accuracy")
    maximum_gap = config.get("maximum_task_head_accuracy_gap")
    minimum_mask_accuracy = config.get("minimum_mask_validation_accuracy")
    reference_digest = config.get("task_head_reference_sha256")
    if (
        not isinstance(reference_accuracy, (int, float))
        or isinstance(reference_accuracy, bool)
        or not 0 <= float(reference_accuracy) <= 1
        or not isinstance(maximum_gap, (int, float))
        or isinstance(maximum_gap, bool)
        or not 0 <= float(maximum_gap) <= 1
        or not isinstance(reference_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", reference_digest) is None
        or not isinstance(config.get("task_head_reference_name"), str)
        or not config["task_head_reference_name"].strip()
    ):
        raise ValueError("task-head adequacy reference is invalid")
    if (
        not isinstance(minimum_mask_accuracy, (int, float))
        or isinstance(minimum_mask_accuracy, bool)
        or not np.isfinite(minimum_mask_accuracy)
        or not 0 <= float(minimum_mask_accuracy) <= 1
    ):
        raise ValueError("minimum_mask_validation_accuracy must lie within [0, 1]")
    ladder = config.get("router_ladder")
    if not isinstance(ladder, Mapping):
        raise ValueError("router_ladder must be an object")
    if set(ladder) != {
        "inner_group_folds",
        "selection_budget",
        "selection_metric",
        "family_tie_order",
        "scikit_learn_version",
        "hist_gbt_grid",
        "shallow_mlp_grid",
    }:
        raise ValueError("router_ladder fields do not match the locked protocol")
    folds = ladder.get("inner_group_folds")
    if isinstance(folds, bool) or not isinstance(folds, int) or folds != 3:
        raise ValueError("router_ladder.inner_group_folds must equal three")
    if ladder.get("selection_metric") != "mean_held_out_error_reduction":
        raise ValueError("router_ladder selection metric is invalid")
    selection_budget = ladder.get("selection_budget")
    if (
        isinstance(selection_budget, bool)
        or not isinstance(selection_budget, (int, float))
        or not np.isclose(float(selection_budget), 0.5)
        or not np.isclose(float(selection_budget), float(config["primary_budget"]))
        or not np.any(np.isclose(budgets, float(selection_budget)))
    ):
        raise ValueError("router_ladder selection budget must be the configured 0.50 budget")
    if tuple(ladder.get("family_tie_order", ())) != ROUTER_FAMILIES:
        raise ValueError("router_ladder family tie order is invalid")
    if ladder.get("scikit_learn_version") != sklearn.__version__:
        raise ValueError("router_ladder scikit-learn version differs from the active runtime")
    grid_specs = {
        "hist_gbt_grid": {
            "max_leaf_nodes",
            "learning_rate",
            "max_iter",
            "min_samples_leaf",
            "l2_regularization",
            "max_bins",
            "early_stopping",
        },
        "shallow_mlp_grid": {
            "hidden_layer_sizes",
            "activation",
            "solver",
            "alpha",
            "learning_rate_init",
            "batch_size",
            "max_iter",
            "early_stopping",
        },
    }
    for grid_name, fields in grid_specs.items():
        grid = ladder.get(grid_name)
        if not isinstance(grid, list) or not grid or any(
            not isinstance(entry, Mapping) or set(entry) != fields for entry in grid
        ):
            raise ValueError(f"router_ladder.{grid_name} is invalid")
        if any(entry.get("early_stopping") is not False for entry in grid):
            raise ValueError(f"router_ladder.{grid_name} must disable internal early stopping")
        serialized = [json.dumps(dict(entry), sort_keys=True) for entry in grid]
        if len(set(serialized)) != len(serialized):
            raise ValueError(f"router_ladder.{grid_name} contains duplicate settings")
    for entry in ladder["hist_gbt_grid"]:
        if (
            isinstance(entry["max_leaf_nodes"], bool)
            or not isinstance(entry["max_leaf_nodes"], int)
            or entry["max_leaf_nodes"] < 2
            or isinstance(entry["max_iter"], bool)
            or not isinstance(entry["max_iter"], int)
            or entry["max_iter"] <= 0
            or isinstance(entry["min_samples_leaf"], bool)
            or not isinstance(entry["min_samples_leaf"], int)
            or entry["min_samples_leaf"] <= 0
            or isinstance(entry["max_bins"], bool)
            or not isinstance(entry["max_bins"], int)
            or not 2 <= entry["max_bins"] <= 255
            or not np.isfinite(float(entry["learning_rate"]))
            or float(entry["learning_rate"]) <= 0
            or not np.isfinite(float(entry["l2_regularization"]))
            or float(entry["l2_regularization"]) < 0
        ):
            raise ValueError("router_ladder.hist_gbt_grid contains invalid settings")
    for entry in ladder["shallow_mlp_grid"]:
        hidden = entry["hidden_layer_sizes"]
        if (
            not isinstance(hidden, list)
            or len(hidden) != 1
            or isinstance(hidden[0], bool)
            or not isinstance(hidden[0], int)
            or hidden[0] <= 0
            or entry["activation"] != "relu"
            or entry["solver"] != "adam"
            or isinstance(entry["batch_size"], bool)
            or not isinstance(entry["batch_size"], int)
            or entry["batch_size"] <= 0
            or isinstance(entry["max_iter"], bool)
            or not isinstance(entry["max_iter"], int)
            or entry["max_iter"] <= 0
            or not np.isfinite(float(entry["alpha"]))
            or float(entry["alpha"]) < 0
            or not np.isfinite(float(entry["learning_rate_init"]))
            or float(entry["learning_rate_init"]) <= 0
        ):
            raise ValueError("router_ladder.shallow_mlp_grid contains invalid settings")


def _load_score_split(
    cache_path: str | pathlib.Path,
    config: Mapping[str, Any],
    split_name: str,
) -> StudySplit:
    """Read one named score split without opening the other data arrays."""

    _validate_config(config)
    if split_name not in {"train", "valid"}:
        raise ValueError("only train and valid splits may be opened")
    source = pathlib.Path(cache_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    expected_digest = config.get("expected_cache_sha256")
    observed_digest = _sha256(source)
    if expected_digest is not None and observed_digest != str(expected_digest):
        raise ValueError("cache SHA256 does not match the locked configuration")

    required_names = {
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
        missing = sorted(required_names - set(archive.files))
        if missing:
            raise ValueError(f"cache is missing fields: {missing}")
        schema = str(_scalar(archive["cache_schema"]))
        if schema != CACHE_SCHEMA:
            raise ValueError(f"unsupported cache schema: {schema}")
        metadata = json.loads(str(_scalar(archive["metadata_json"])))
        if not isinstance(metadata, dict):
            raise ValueError("cache metadata must be an object")
        if metadata.get("alignment") != "positive_interval_overlap_mean":
            raise ValueError("cache must use utterance-aligned interval pooling")

        x = np.asarray(archive[f"{split_name}_x"], dtype=np.float64)
        y = np.asarray(archive[f"{split_name}_y"], dtype=np.int64)
        sample_ids = tuple(str(value) for value in archive[f"{split_name}_sample_ids"].tolist())
    if x.shape != (len(y), len(MODALITIES)):
        raise ValueError(f"{split_name} score shape must be (n, 3)")
    if len(sample_ids) != len(y) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"{split_name} identifiers must be unique and aligned")
    if not np.isfinite(x).all() or not np.isin(y, (0, 1)).all():
        raise ValueError(f"{split_name} scores or labels are invalid")
    expected_counts = config.get("expected_split_counts")
    if expected_counts is not None:
        metadata_counts = metadata.get("loaded_split_counts")
        if not isinstance(metadata_counts, dict):
            raise ValueError("cache metadata lacks loaded split counts")
        locked_counts = {name: int(expected_counts[name]) for name in ("train", "valid", "test")}
        observed_counts = {name: int(metadata_counts.get(name, -1)) for name in locked_counts}
        if observed_counts != locked_counts:
            raise ValueError("cache metadata counts do not match the locked configuration")
        if len(y) != locked_counts[split_name]:
            raise ValueError(f"{split_name} array count does not match the locked configuration")
    groups = np.asarray([video_group_id(value) for value in sample_ids], dtype="U")
    return StudySplit(x=x, y=y, sample_ids=sample_ids, groups=groups)


def _validate_split_isolation(train: StudySplit, valid: StudySplit) -> None:
    if set(train.sample_ids).intersection(valid.sample_ids):
        raise ValueError("training and validation segment identifiers overlap")
    if set(train.groups).intersection(valid.groups):
        raise ValueError("training and validation video groups overlap")


def load_training_and_validation(
    cache_path: str | pathlib.Path,
    config: Mapping[str, Any],
) -> dict[str, StudySplit]:
    """Read and validate training/validation data without opening test arrays."""

    train = _load_score_split(cache_path, config, "train")
    valid = _load_score_split(cache_path, config, "valid")
    _validate_split_isolation(train, valid)
    return {"train": train, "valid": valid}


def _partition_indices(
    groups: np.ndarray,
    fractions: Sequence[float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = source_group_split(
        np.asarray(groups),
        fractions=tuple(float(value) for value in fractions),  # type: ignore[arg-type]
        seed=int(seed),
    )
    return split["train"], split["dev"], split["test"]


def _group_digest(groups: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(str(value) for value in groups))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validation_group_record(groups: np.ndarray) -> tuple[list[str], str]:
    hashes = sorted(_group_digest([str(group)]) for group in np.unique(groups))
    digest = hashlib.sha256("\n".join(hashes).encode("utf-8")).hexdigest()
    return hashes, digest


def _observed_masks() -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray([(bits >> index) & 1 for index in range(3)], dtype=bool)
        for bits in range(1, 8)
    )


def _contexts(mode: str) -> tuple[tuple[np.ndarray, tuple[int, ...]], ...]:
    if mode == "singleton":
        return tuple(
            (
                np.asarray([index == observed for index in range(3)], dtype=bool),
                tuple(index for index in range(3) if index != observed),
            )
            for observed in range(3)
        )
    if mode == "pair":
        return tuple(
            (
                np.asarray([index != missing for index in range(3)], dtype=bool),
                (missing,),
            )
            for missing in range(3)
        )
    raise ValueError("mode must be 'singleton' or 'pair'")


def _fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def _task_features(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if z.ndim != 2 or z.shape[1] != 3 or mask.shape != (3,) or not np.any(mask):
        raise ValueError("task features require n-by-3 scores and a nonempty length-three mask")
    observed = z[:, mask]
    expanded_mask = np.broadcast_to(mask, z.shape)
    masked_and_mask = masked_observations(z, expanded_mask)
    summary = np.column_stack(
        (
            np.mean(observed, axis=1),
            np.mean(np.abs(observed), axis=1),
            np.max(observed, axis=1) - np.min(observed, axis=1),
            np.full(len(z), int(mask.sum()), dtype=np.float64),
        )
    )
    return np.column_stack((masked_and_mask, summary))


def _ridge_fit(features: np.ndarray, targets: np.ndarray, penalty: float) -> np.ndarray:
    if penalty < 0:
        raise ValueError("ridge penalty must be nonnegative")
    design = np.column_stack((np.ones(len(features)), np.asarray(features, dtype=np.float64)))
    regularizer = np.eye(design.shape[1], dtype=np.float64) * float(penalty)
    regularizer[0, 0] = 0.0
    return np.linalg.pinv(design.T @ design + regularizer) @ design.T @ np.asarray(targets, dtype=np.float64)


def _linear_predict(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(features)), np.asarray(features, dtype=np.float64)))
    return design @ weights


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _logistic_fit(features: np.ndarray, targets: np.ndarray, penalty: float) -> tuple[np.ndarray | None, float]:
    labels = np.asarray(targets, dtype=np.float64)
    smoothed_rate = float((labels.sum() + 0.5) / (len(labels) + 1.0))
    if len(np.unique(labels)) < 2:
        return None, smoothed_rate
    design = np.column_stack((np.ones(len(features)), np.asarray(features, dtype=np.float64)))
    weights = np.zeros(design.shape[1], dtype=np.float64)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * float(penalty)
    regularizer[0, 0] = 0.0
    for _ in range(60):
        probability = _sigmoid(design @ weights)
        curvature = np.maximum(probability * (1.0 - probability), 1e-7)
        gradient = design.T @ (probability - labels) + regularizer @ weights
        hessian = (design.T * curvature) @ design + regularizer
        update = np.linalg.pinv(hessian) @ gradient
        weights -= update
        if float(np.max(np.abs(update))) < 1e-8:
            break
    return weights, smoothed_rate


def _logistic_predict(features: np.ndarray, model: tuple[np.ndarray | None, float]) -> np.ndarray:
    weights, rate = model
    if weights is None:
        return np.full(len(features), rate, dtype=np.float64)
    return _sigmoid(_linear_predict(features, weights))


def _mask_key(mask: np.ndarray) -> int:
    return int(sum((1 << index) for index, observed in enumerate(mask) if observed))


def _mask_name(mask: np.ndarray) -> str:
    return "+".join(MODALITIES[index] for index in np.flatnonzero(mask))


def _fit_task_model(z: np.ndarray, y: np.ndarray, penalty: float) -> dict[int, np.ndarray]:
    """Fit one frozen task head per observable modality mask."""

    targets = 2 * np.asarray(y, dtype=np.float64) - 1.0
    return {
        _mask_key(mask): _ridge_fit(_task_features(z, mask), targets, penalty)
        for mask in _observed_masks()
    }


def _task_scores(z: np.ndarray, mask: np.ndarray, weights: Mapping[int, np.ndarray]) -> np.ndarray:
    key = _mask_key(mask)
    if key not in weights:
        raise ValueError(f"task model lacks observation mask {key}")
    return _linear_predict(_task_features(z, mask), np.asarray(weights[key]))


def _router_features(
    z: np.ndarray,
    observed_mask: np.ndarray,
    candidate: int,
    base_score: np.ndarray,
) -> np.ndarray:
    if observed_mask[candidate]:
        raise ValueError("candidate modality is already observed")
    observed = z[:, observed_mask]
    candidate_code = np.zeros((len(z), 3), dtype=np.float64)
    candidate_code[:, candidate] = 1.0
    expanded_mask = np.broadcast_to(observed_mask, z.shape)
    masked_and_mask = masked_observations(z, expanded_mask)
    observable_features = np.column_stack(
        (
            masked_and_mask,
            np.mean(observed, axis=1),
            np.mean(np.abs(observed), axis=1),
            np.max(observed, axis=1) - np.min(observed, axis=1),
            np.abs(base_score),
        )
    )
    candidate_interactions = (
        candidate_code[:, :, None] * observable_features[:, None, :]
    ).reshape(len(z), -1)
    return np.column_stack((observable_features, candidate_code, candidate_interactions))


def _make_decisions(
    z: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    task_weights: Mapping[int, np.ndarray],
    mode: str,
) -> _DecisionSet:
    contexts = _contexts(mode)
    candidate_count = len(contexts[0][1])
    base_losses: list[np.ndarray] = []
    base_confidences: list[np.ndarray] = []
    candidate_losses: list[np.ndarray] = []
    candidate_confidences: list[np.ndarray] = []
    values: list[np.ndarray] = []
    router_blocks: list[np.ndarray] = []
    context_indices: list[np.ndarray] = []
    modality_blocks: list[np.ndarray] = []
    group_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    for context_index, (observed_mask, candidates) in enumerate(contexts):
        base_score = _task_scores(z, observed_mask, task_weights)
        base_prediction = (base_score >= 0).astype(np.int64)
        base_loss = (base_prediction != y).astype(np.float64)
        loss_columns = []
        confidence_columns = []
        feature_columns = []
        for candidate in candidates:
            after_mask = observed_mask.copy()
            after_mask[candidate] = True
            after_score = _task_scores(z, after_mask, task_weights)
            after_prediction = (after_score >= 0).astype(np.int64)
            loss_columns.append((after_prediction != y).astype(np.float64))
            confidence_columns.append(np.abs(after_score))
            feature_columns.append(_router_features(z, observed_mask, candidate, base_score))
        losses = np.column_stack(loss_columns)
        full_after_loss = np.zeros((len(z), 3), dtype=np.float64)
        full_observed_mask = np.ones((len(z), 3), dtype=bool)
        for column, candidate in enumerate(candidates):
            full_after_loss[:, candidate] = losses[:, column]
            full_observed_mask[:, candidate] = False
        full_value = realized_acquisition_value(base_loss, full_after_loss, full_observed_mask)
        context_value = np.column_stack([full_value[:, candidate] for candidate in candidates])
        base_losses.append(base_loss)
        base_confidences.append(np.abs(base_score))
        candidate_losses.append(losses)
        candidate_confidences.append(np.column_stack(confidence_columns))
        values.append(context_value)
        router_blocks.append(np.stack(feature_columns, axis=1))
        context_indices.append(np.full(len(z), context_index, dtype=np.int64))
        modality_blocks.append(np.broadcast_to(np.asarray(candidates, dtype=np.int64), (len(z), candidate_count)))
        group_blocks.append(np.asarray(groups, dtype="U"))
        label_blocks.append(np.asarray(y, dtype=np.int64))
    return _DecisionSet(
        labels=np.concatenate(label_blocks),
        base_loss=np.concatenate(base_losses),
        base_confidence=np.concatenate(base_confidences),
        candidate_loss=np.concatenate(candidate_losses),
        candidate_confidence=np.concatenate(candidate_confidences),
        value=np.concatenate(values),
        router_features=np.concatenate(router_blocks),
        context_index=np.concatenate(context_indices),
        candidate_modality=np.concatenate(modality_blocks),
        groups=np.concatenate(group_blocks),
    )


def _subset_decisions(decisions: _DecisionSet, indices: np.ndarray) -> _DecisionSet:
    selected = np.asarray(indices, dtype=np.int64)
    return _DecisionSet(
        **{
            field: np.asarray(getattr(decisions, field))[selected]
            for field in _DecisionSet.__dataclass_fields__
        }
    )


def _json_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, value in parameters.items():
        if isinstance(value, tuple):
            output[name] = list(value)
        elif isinstance(value, np.generic):
            output[name] = value.item()
        else:
            output[name] = value
    return output


def _fit_value_model(
    family: str,
    features: np.ndarray,
    targets: np.ndarray,
    parameters: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if family == "linear_value":
        weights = _ridge_fit(x, y, float(parameters["ridge"]))
        return {"kind": "linear", "weights": weights}, {
            "status": "closed_form",
            "converged": True,
        }
    if family == "hist_gbt_value":
        estimator = HistGradientBoostingRegressor(
            **dict(parameters),
            random_state=int(seed),
        )
        estimator.fit(x, y)
        return {"kind": "hist_gbt", "estimator": estimator}, {
            "status": "fixed_iterations_completed",
            "converged": True,
            "iterations": int(estimator.n_iter_),
        }
    if family == "shallow_mlp_value":
        mean, scale = _fit_standardizer(x)
        normalized = (x - mean) / scale
        settings = dict(parameters)
        settings["hidden_layer_sizes"] = tuple(int(value) for value in settings["hidden_layer_sizes"])
        settings["batch_size"] = min(int(settings["batch_size"]), len(normalized))
        estimator = MLPRegressor(**settings, random_state=int(seed))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(normalized, y)
        convergence_messages = [
            str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)
        ]
        return {
            "kind": "shallow_mlp",
            "estimator": estimator,
            "mean": mean,
            "scale": scale,
        }, {
            "status": "converged" if not convergence_messages else "maximum_iterations_reached",
            "converged": not convergence_messages,
            "iterations": int(estimator.n_iter_),
            "loss": float(estimator.loss_),
            "warnings": convergence_messages,
            "effective_batch_size": int(settings["batch_size"]),
        }
    raise ValueError(f"unknown router family: {family}")


def _predict_value_model(model: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    kind = model.get("kind")
    if kind == "linear":
        return _linear_predict(x, np.asarray(model["weights"], dtype=np.float64))
    if kind == "hist_gbt":
        return np.asarray(model["estimator"].predict(x), dtype=np.float64)
    if kind == "shallow_mlp":
        normalized = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
        return np.asarray(model["estimator"].predict(normalized), dtype=np.float64)
    raise ValueError("router model kind is invalid")


def _predict_decision_values(decisions: _DecisionSet, model: Mapping[str, Any]) -> np.ndarray:
    shape = decisions.value.shape
    flat = decisions.router_features.reshape(-1, decisions.router_features.shape[-1])
    predictions = _predict_value_model(model, flat).reshape(shape)
    if not np.isfinite(predictions).all():
        raise ValueError("router produced nonfinite value predictions")
    return predictions


def _predicted_value_error_reduction(
    decisions: _DecisionSet,
    predictions: np.ndarray,
    budget: float,
    seed: int,
) -> float:
    predicted = np.asarray(predictions, dtype=np.float64)
    if predicted.shape != decisions.value.shape or not np.isfinite(predicted).all():
        raise ValueError("router predictions do not match decision candidates")
    priority = np.max(predicted, axis=1)
    choice = np.argmax(predicted, axis=1)
    count = int(np.rint(float(budget) * len(decisions.base_loss)))
    order = _seeded_descending_order(priority, int(seed) + 2_000_003)
    query = np.zeros(len(decisions.base_loss), dtype=bool)
    query[order[:count]] = True
    chosen_loss = decisions.candidate_loss[np.arange(len(choice)), choice]
    final_loss = np.where(query, chosen_loss, decisions.base_loss)
    return float(np.mean(decisions.base_loss - final_loss))


def _select_router_family(scores: Mapping[str, float], tie_order: Sequence[str]) -> str:
    order = tuple(str(value) for value in tie_order)
    if set(scores) != set(order) or len(set(order)) != len(order):
        raise ValueError("router-family scores and tie order must contain the same unique families")
    if not all(np.isfinite(float(value)) for value in scores.values()):
        raise ValueError("router-family scores must be finite")
    return max(order, key=lambda name: (float(scores[name]), -order.index(name)))


def _router_cv_folds(
    decisions: _DecisionSet,
    fold_count: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    if len(np.unique(decisions.groups)) < fold_count:
        raise ValueError("router-fit partition has fewer video groups than inner folds")
    splitter = GroupKFold(n_splits=int(fold_count))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    records: list[dict[str, Any]] = []
    for fold_index, (fit_indices, heldout_indices) in enumerate(
        splitter.split(np.zeros(len(decisions.groups)), groups=decisions.groups)
    ):
        fit_groups = decisions.groups[fit_indices]
        heldout_groups = decisions.groups[heldout_indices]
        fit_hashes, fit_digest = _validation_group_record(fit_groups)
        heldout_hashes, heldout_digest = _validation_group_record(heldout_groups)
        disjoint = not set(fit_groups).intersection(set(heldout_groups))
        folds.append((np.asarray(fit_indices), np.asarray(heldout_indices)))
        records.append(
            {
                "fold": int(fold_index),
                "train_decisions": int(len(fit_indices)),
                "heldout_decisions": int(len(heldout_indices)),
                "train_group_hashes": fit_hashes,
                "heldout_group_hashes": heldout_hashes,
                "train_group_digest": fit_digest,
                "heldout_group_digest": heldout_digest,
                "group_disjoint": bool(disjoint),
            }
        )
    return folds, records


def _cross_validate_family(
    family: str,
    decisions: _DecisionSet,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    parameter_grid: Sequence[Mapping[str, Any]],
    budget: float,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for parameters in parameter_grid:
        fold_scores: list[float] = []
        fold_records: list[dict[str, Any]] = []
        for fold_index, (fit_indices, heldout_indices) in enumerate(folds):
            fit = _subset_decisions(decisions, fit_indices)
            heldout = _subset_decisions(decisions, heldout_indices)
            flat_features = fit.router_features.reshape(-1, fit.router_features.shape[-1])
            flat_values = fit.value.reshape(-1)
            model, fit_state = _fit_value_model(
                family,
                flat_features,
                flat_values,
                parameters,
                seed=int(seed) * 100_003 + fold_index,
            )
            predictions = _predict_decision_values(heldout, model)
            score = _predicted_value_error_reduction(
                heldout,
                predictions,
                budget,
                int(seed) * 1009 + fold_index * 97,
            )
            fold_scores.append(score)
            fold_records.append(
                {
                    "fold": int(fold_index),
                    "error_reduction": float(score),
                    "prediction_sha256": _array_sha256(predictions),
                    "fit_state": fit_state,
                }
            )
        candidates.append(
            {
                "parameters": _json_parameters(parameters),
                "folds": fold_records,
                "mean_held_out_error_reduction": float(np.mean(fold_scores)),
            }
        )
    selected_index = max(
        range(len(candidates)),
        key=lambda index: (candidates[index]["mean_held_out_error_reduction"], -index),
    )
    return dict(parameter_grid[selected_index]), candidates


def _fit_router_ladder(
    decisions: _DecisionSet,
    calibration: _DecisionSet,
    config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ladder = config["router_ladder"]
    budget = float(ladder["selection_budget"])
    folds, fold_records = _router_cv_folds(decisions, int(ladder["inner_group_folds"]))
    grids: dict[str, list[dict[str, Any]]] = {
        "linear_value": [{"ridge": float(config["ridge"])}],
        "hist_gbt_value": [dict(entry) for entry in ladder["hist_gbt_grid"]],
        "shallow_mlp_value": [dict(entry) for entry in ladder["shallow_mlp_grid"]],
    }
    models: dict[str, Any] = {}
    family_records: dict[str, Any] = {}
    calibration_scores: dict[str, float] = {}
    for family_index, family in enumerate(ROUTER_FAMILIES):
        parameters, candidate_records = _cross_validate_family(
            family,
            decisions,
            folds,
            grids[family],
            budget,
            int(seed) + family_index * 10_007,
        )
        flat_features = decisions.router_features.reshape(-1, decisions.router_features.shape[-1])
        flat_values = decisions.value.reshape(-1)
        model, fit_state = _fit_value_model(
            family,
            flat_features,
            flat_values,
            parameters,
            seed=int(seed) * 1_000_003 + family_index,
        )
        models[family] = model
        calibration_predictions = _predict_decision_values(calibration, model)
        calibration_scores[family] = _predicted_value_error_reduction(
            calibration,
            calibration_predictions,
            budget,
            int(seed) * 1009,
        )
        family_records[family] = {
            "estimator": {
                "linear_value": "closed_form_ridge",
                "hist_gbt_value": "sklearn.ensemble.HistGradientBoostingRegressor",
                "shallow_mlp_value": "sklearn.neural_network.MLPRegressor",
            }[family],
            "grid": [_json_parameters(entry) for entry in grids[family]],
            "cv_candidates": candidate_records,
            "selected_parameters": _json_parameters(parameters),
            "fit_state": fit_state,
            "calibration_prediction_sha256": _array_sha256(calibration_predictions),
        }
    selected_family = _select_router_family(
        calibration_scores,
        tuple(ladder["family_tie_order"]),
    )
    selected_calibration_predictions = _predict_decision_values(
        calibration,
        models[selected_family],
    )
    calibration_priority = np.max(selected_calibration_predictions, axis=1)
    online_thresholds = {
        f"{float(target):.6f}": _fit_online_query_threshold(
            calibration_priority,
            float(target),
        )
        for target in sorted(float(value) for value in config["budgets"])
    }
    flat_features = decisions.router_features.reshape(-1, decisions.router_features.shape[-1])
    flat_values = decisions.value.reshape(-1)
    benefit_model = _logistic_fit(
        flat_features,
        (flat_values > 0).astype(np.float64),
        float(config["benefit_ridge"]),
    )
    contexts = int(np.max(decisions.context_index)) + 1
    source_prior: dict[int, dict[int, float]] = {}
    for context in range(contexts):
        selected = decisions.context_index == context
        source_prior[context] = {}
        for column, modality in enumerate(decisions.candidate_modality[selected][0]):
            source_prior[context][int(modality)] = float(np.mean(decisions.value[selected, column]))
    router = {
        "value_models": models,
        "selected_family": selected_family,
        "benefit_model": benefit_model,
        "source_prior": source_prior,
        "online_query_thresholds": online_thresholds,
    }
    record: dict[str, Any] = {
        "protocol": "router_fit_group_cv_then_calibration_family_selection",
        "inner_cv": {
            "fold_count": int(ladder["inner_group_folds"]),
            "grouping_unit": "video",
            "selection_metric": str(ladder["selection_metric"]),
            "budget": budget,
            "folds": fold_records,
        },
        "families": family_records,
        "selection": {
            "split": "calibration",
            "budget": budget,
            "metric": "error_reduction",
            "family_tie_order": list(ladder["family_tie_order"]),
            "candidate_error_reduction": {
                name: float(calibration_scores[name]) for name in ROUTER_FAMILIES
            },
            "selected_family": selected_family,
        },
        "online_query_policy": {
            "protocol": "calibration_fixed_value_threshold_per_target_budget",
            "fit_split": "calibration",
            "priority": "maximum_selected_router_predicted_value",
            "candidate_choice": "argmax_selected_router_predicted_value",
            "selected_family": selected_family,
            "calibration_priority_sha256": _array_sha256(calibration_priority),
            "by_target_budget": online_thresholds,
        },
        "library_versions": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    record["training_attestation_sha256"] = _json_sha256(record)
    return router, record


def _fit_confidence_models(decisions: _DecisionSet, penalty: float) -> dict[str, Any]:
    """Calibrate task correctness separately for every observable mask."""

    base: dict[int, tuple[np.ndarray | None, float]] = {}
    candidate: dict[tuple[int, int], tuple[np.ndarray | None, float]] = {}
    for context in np.unique(decisions.context_index):
        selected = decisions.context_index == context
        base[int(context)] = _logistic_fit(
            decisions.base_confidence[selected, None],
            1.0 - decisions.base_loss[selected],
            penalty,
        )
        for column, modality in enumerate(decisions.candidate_modality[selected][0]):
            candidate[(int(context), int(modality))] = _logistic_fit(
                decisions.candidate_confidence[selected, column, None],
                1.0 - decisions.candidate_loss[selected, column],
                penalty,
            )
    return {"base": base, "candidate": candidate}


def _calibrated_confidence(
    decisions: _DecisionSet,
    router: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    models = router.get("confidence_models")
    if not isinstance(models, Mapping):
        raise ValueError("router is missing context-specific confidence calibration")
    base_probability = np.empty(len(decisions.base_loss), dtype=np.float64)
    candidate_probability = np.empty_like(decisions.candidate_confidence, dtype=np.float64)
    for context in np.unique(decisions.context_index):
        selected = decisions.context_index == context
        base_probability[selected] = _logistic_predict(
            decisions.base_confidence[selected, None], models["base"][int(context)]
        )
        for column, modality in enumerate(decisions.candidate_modality[selected][0]):
            candidate_probability[selected, column] = _logistic_predict(
                decisions.candidate_confidence[selected, column, None],
                models["candidate"][(int(context), int(modality))],
            )
    return base_probability, candidate_probability


def _router_predictions(
    decisions: _DecisionSet,
    router: Mapping[str, Any],
    family: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    selected = str(router["selected_family"] if family is None else family)
    if selected == "selected_router":
        selected = str(router["selected_family"])
    models = router.get("value_models")
    if not isinstance(models, Mapping) or selected not in models:
        raise ValueError(f"router is missing value model {selected}")
    value = _predict_decision_values(decisions, models[selected])
    flat = decisions.router_features.reshape(-1, decisions.router_features.shape[-1])
    shape = decisions.value.shape
    benefit = _logistic_predict(flat, router["benefit_model"]).reshape(shape)
    return value, benefit


def _source_prior_arrays(decisions: _DecisionSet, router: Mapping[str, Any]) -> np.ndarray:
    output = np.empty_like(decisions.value)
    priors = router["source_prior"]
    for index in range(len(output)):
        context = int(decisions.context_index[index])
        for column, modality in enumerate(decisions.candidate_modality[index]):
            output[index, column] = float(priors[context][int(modality)])
    return output


def _seeded_descending_order(scores: np.ndarray, seed: int) -> np.ndarray:
    """Rank finite scores while using a seeded key only for exact ties."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("ranking scores must be a finite vector")
    tie_key = np.random.default_rng(seed).random(len(values))
    return np.lexsort((tie_key, -values))


def _fit_online_query_threshold(
    calibration_priority: np.ndarray,
    target_query_rate: float,
) -> dict[str, Any]:
    """Fit a finite per-example threshold using calibration priorities only."""

    priority = np.asarray(calibration_priority, dtype=np.float64)
    target = float(target_query_rate)
    if (
        priority.ndim != 1
        or len(priority) == 0
        or not np.isfinite(priority).all()
        or not np.isfinite(target)
        or target < 0.0
        or target > 1.0
    ):
        raise ValueError(
            "online threshold fitting requires finite priorities and a target in [0, 1]"
        )
    target_count = int(np.rint(target * len(priority)))
    descending = np.sort(priority)[::-1]
    if target_count == 0:
        # Use the largest finite float so the strict rule is a true
        # never-query policy for every valid future priority, not merely for
        # values bounded by the calibration maximum.
        threshold = float(np.finfo(np.float64).max)
        query_on_equal = False
    elif target_count == len(priority):
        # Every finite priority is at least the smallest finite float, making
        # the inclusive rule a true always-query policy under distribution
        # shift as well as on calibration.
        threshold = float(-np.finfo(np.float64).max)
        query_on_equal = True
    else:
        threshold = float(descending[target_count - 1])
        inclusive_count = int(np.sum(priority >= threshold))
        exclusive_count = int(np.sum(priority > threshold))
        # A scalar threshold cannot split an exact score tie. Choose the
        # closest feasible rate and prefer the lower-cost option on an exact
        # distance tie.
        query_on_equal = abs(inclusive_count - target_count) < abs(
            exclusive_count - target_count
        )
    query = (priority > threshold) | (query_on_equal & (priority == threshold))
    query_count = int(np.sum(query))
    return {
        "target_query_rate": target,
        "target_query_count": target_count,
        "calibration_decisions": len(priority),
        "calibration_query_count": query_count,
        "calibration_realized_query_rate": float(np.mean(query)),
        "threshold": threshold,
        "query_on_equal": bool(query_on_equal),
        "tie_convention": (
            "priority_greater_than_or_equal_to_threshold"
            if query_on_equal
            else "priority_strictly_greater_than_threshold"
        ),
    }


def _apply_online_query_threshold(
    priority: np.ndarray,
    fitted: Mapping[str, Any],
) -> np.ndarray:
    """Apply one frozen threshold independently to each arriving example."""

    values = np.asarray(priority, dtype=np.float64)
    threshold = fitted.get("threshold")
    query_on_equal = fitted.get("query_on_equal")
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or isinstance(threshold, bool)
        or not isinstance(threshold, (int, float, np.integer, np.floating))
        or not np.isfinite(threshold)
        or not isinstance(query_on_equal, (bool, np.bool_))
    ):
        raise ValueError("online query threshold record is invalid")
    return (values > float(threshold)) | (
        bool(query_on_equal) & (values == float(threshold))
    )


def _policy_priority_and_choice(
    decisions: _DecisionSet,
    router: Mapping[str, Any],
    policy: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pre-query priority and candidate choice for every decision."""

    family = policy if policy in ROUTER_FAMILIES else "selected_router"
    predicted_value, _ = _router_predictions(decisions, router, family)
    source_prior = _source_prior_arrays(decisions, router)
    if policy == "oracle":
        priority = np.max(decisions.value, axis=1)
        choice = np.argmax(decisions.value, axis=1)
    elif policy in {*ROUTER_FAMILIES, "selected_router", "online_selected_router"}:
        priority = np.max(predicted_value, axis=1)
        choice = np.argmax(predicted_value, axis=1)
    elif policy == "selected_router_query_source_prior_choice":
        priority = np.max(predicted_value, axis=1)
        choice = np.argmax(source_prior, axis=1)
    elif policy == "confidence_query_selected_router_choice":
        base_probability, _ = _calibrated_confidence(decisions, router)
        source_choice = np.argmax(source_prior, axis=1)
        priority = -base_probability + 1e-10 * source_prior[
            np.arange(len(source_choice)), source_choice
        ]
        choice = np.argmax(predicted_value, axis=1)
    elif policy == "confidence":
        base_probability, _ = _calibrated_confidence(decisions, router)
        choice = np.argmax(source_prior, axis=1)
        priority = -base_probability + 1e-10 * source_prior[np.arange(len(choice)), choice]
    elif policy == "source_prior":
        choice = np.argmax(source_prior, axis=1)
        priority = source_prior[np.arange(len(choice)), choice]
    elif policy == "random_matched":
        priority = np.random.default_rng(seed).random(len(decisions.base_loss))
        choice_rng = np.random.default_rng(seed + 1_000_003)
        choice = np.asarray(
            [choice_rng.integers(decisions.value.shape[1]) for _ in range(len(decisions.value))],
            dtype=np.int64,
        )
    else:
        raise ValueError(f"unknown policy: {policy}")
    return np.asarray(priority, dtype=np.float64), np.asarray(choice, dtype=np.int64)


def _policy_seed(seed: int, policy: str, router: Mapping[str, Any]) -> int:
    canonical = policy
    if policy in {
        "selected_router",
        "online_selected_router",
        "selected_router_query_source_prior_choice",
    }:
        canonical = str(router["selected_family"])
    elif policy == "confidence_query_selected_router_choice":
        canonical = "confidence"
    if canonical not in POLICY_SEED_INDEX:
        raise ValueError(f"policy seed is undefined for {policy}")
    return int(seed) * 1009 + POLICY_SEED_INDEX[canonical] * 97


def _policy_arrays(
    decisions: _DecisionSet,
    router: Mapping[str, Any],
    policy: str,
    budget: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    priority, choice = _policy_priority_and_choice(decisions, router, policy, seed)
    if policy == "online_selected_router":
        thresholds = router.get("online_query_thresholds")
        key = f"{float(budget):.6f}"
        if not isinstance(thresholds, Mapping) or key not in thresholds:
            raise ValueError(f"router is missing online query threshold for {key}")
        query = _apply_online_query_threshold(priority, thresholds[key])
    else:
        count = int(np.rint(float(budget) * len(decisions.base_loss)))
        if count < 0 or count > len(decisions.base_loss):
            raise ValueError("query budget is outside the decision range")
        order = _seeded_descending_order(priority, seed + 2_000_003)
        query = np.zeros(len(decisions.base_loss), dtype=bool)
        query[order[:count]] = True
    chosen_loss = decisions.candidate_loss[np.arange(len(choice)), choice]
    base_confidence, candidate_confidence = _calibrated_confidence(decisions, router)
    chosen_confidence = candidate_confidence[np.arange(len(choice)), choice]
    final_loss = np.where(query, chosen_loss, decisions.base_loss)
    final_confidence = np.where(query, chosen_confidence, base_confidence)
    return query, choice, final_loss, final_confidence


def _group_macro(values: np.ndarray, groups: np.ndarray) -> float:
    return float(np.mean([np.mean(values[groups == group]) for group in np.unique(groups)]))


def _binary_macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    values: list[float] = []
    for target in (0, 1):
        true_positive = int(np.sum((labels == target) & (predictions == target)))
        false_positive = int(np.sum((labels != target) & (predictions == target)))
        false_negative = int(np.sum((labels == target) & (predictions != target)))
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return float(np.mean(values))


def _risk_coverage_auc(losses: np.ndarray, confidence: np.ndarray) -> float:
    order = np.argsort(-np.asarray(confidence), kind="stable")
    sorted_confidence = np.asarray(confidence)[order]
    cumulative_errors = np.cumsum(np.asarray(losses)[order])
    block_ends = np.flatnonzero(
        np.concatenate((sorted_confidence[1:] != sorted_confidence[:-1], np.asarray([True])))
    )
    coverages = (block_ends + 1) / len(losses)
    risks = cumulative_errors[block_ends] / (block_ends + 1)
    widths = np.diff(np.concatenate((np.asarray([0.0]), coverages)))
    return float(np.sum(widths * risks))


def _weighted_topk_sums(
    weights: np.ndarray,
    order: np.ndarray,
    values: np.ndarray,
    fractions: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Sum values under exact weighted top-k selection for many resamples."""

    ordered_weights = np.asarray(weights[:, order], dtype=np.int64)
    ordered_values = np.asarray(values, dtype=np.float64)[order]
    cumulative_count = np.cumsum(ordered_weights, axis=1)
    cumulative_value = np.cumsum(ordered_weights * ordered_values[None, :], axis=1)
    totals = np.sum(ordered_weights, axis=1)
    sums = np.zeros((len(weights), len(fractions)), dtype=np.float64)
    counts = np.zeros((len(weights), len(fractions)), dtype=np.int64)
    for row in range(len(weights)):
        for column, fraction in enumerate(fractions):
            count = int(np.rint(float(fraction) * int(totals[row])))
            counts[row, column] = count
            if count == 0:
                continue
            boundary = int(np.searchsorted(cumulative_count[row], count, side="left"))
            previous_count = 0 if boundary == 0 else int(cumulative_count[row, boundary - 1])
            previous_value = 0.0 if boundary == 0 else float(cumulative_value[row, boundary - 1])
            sums[row, column] = previous_value + (count - previous_count) * ordered_values[boundary]
    return sums, counts


def _weighted_topk_selection(
    weights: np.ndarray,
    order: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate an exact weighted top-k count, including a partial boundary."""

    if fraction < 0 or fraction > 1:
        raise ValueError("selection fraction must lie within [0, 1]")
    raw_weights = np.asarray(weights, dtype=np.int64)
    ordered_weights = raw_weights[:, order]
    cumulative_count = np.cumsum(ordered_weights, axis=1)
    totals = np.sum(ordered_weights, axis=1)
    target_counts = np.rint(float(fraction) * totals).astype(np.int64)
    selected_ordered = np.zeros_like(ordered_weights)
    for row, count in enumerate(target_counts):
        if count == 0:
            continue
        boundary = int(np.searchsorted(cumulative_count[row], count, side="left"))
        if boundary:
            selected_ordered[row, :boundary] = ordered_weights[row, :boundary]
        previous_count = 0 if boundary == 0 else int(cumulative_count[row, boundary - 1])
        selected_ordered[row, boundary] = int(count) - previous_count
    selected = np.zeros_like(raw_weights)
    selected[:, order] = selected_ordered
    return selected, target_counts


def _policy_reselected_cluster_bootstrap(
    decisions: _DecisionSet,
    router: Mapping[str, Any],
    policy: str,
    budgets: Sequence[float],
    repetitions: int,
    policy_seed: int,
    bootstrap_seed: int,
) -> np.ndarray:
    """Bootstrap video groups and reselect every exact query budget."""

    priority, choice = _policy_priority_and_choice(decisions, router, policy, policy_seed)
    order = _seeded_descending_order(priority, policy_seed + 2_000_003)
    queried_loss = decisions.candidate_loss[np.arange(len(choice)), choice]
    improvement = decisions.base_loss - queried_loss
    unique, inverse = np.unique(decisions.groups, return_inverse=True)
    rng = np.random.default_rng(bootstrap_seed)
    estimates = np.empty((repetitions, len(budgets)), dtype=np.float64)
    chunk_size = min(250, repetitions)
    for start in range(0, repetitions, chunk_size):
        stop = min(start + chunk_size, repetitions)
        rows = stop - start
        selected_groups = rng.integers(0, len(unique), size=(rows, len(unique)))
        group_multiplicity = np.zeros((rows, len(unique)), dtype=np.int32)
        np.add.at(
            group_multiplicity,
            (np.repeat(np.arange(rows), len(unique)), selected_groups.reshape(-1)),
            1,
        )
        weights = group_multiplicity[:, inverse]
        selected_sums, _ = _weighted_topk_sums(weights, order, improvement, budgets)
        totals = np.sum(weights, axis=1)
        estimates[start:stop] = selected_sums / totals[:, None]
    return estimates


def _fixed_threshold_cluster_bootstrap(
    decisions: _DecisionSet,
    query: np.ndarray,
    choice: np.ndarray,
    repetitions: int,
    bootstrap_seed: int,
) -> np.ndarray:
    """Bootstrap video groups while keeping every threshold decision fixed."""

    selected = np.asarray(query, dtype=bool)
    selected_choice = np.asarray(choice, dtype=np.int64)
    count = len(decisions.base_loss)
    if (
        selected.shape != (count,)
        or selected_choice.shape != (count,)
        or np.any(selected_choice < 0)
        or np.any(selected_choice >= decisions.value.shape[1])
        or repetitions <= 0
    ):
        raise ValueError("fixed-threshold bootstrap inputs are invalid")
    queried_loss = decisions.candidate_loss[np.arange(count), selected_choice]
    improvement = np.where(selected, decisions.base_loss - queried_loss, 0.0)
    unique, inverse = np.unique(decisions.groups, return_inverse=True)
    rng = np.random.default_rng(bootstrap_seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    chunk_size = min(250, repetitions)
    for start in range(0, repetitions, chunk_size):
        stop = min(start + chunk_size, repetitions)
        rows = stop - start
        selected_groups = rng.integers(0, len(unique), size=(rows, len(unique)))
        group_multiplicity = np.zeros((rows, len(unique)), dtype=np.int32)
        np.add.at(
            group_multiplicity,
            (np.repeat(np.arange(rows), len(unique)), selected_groups.reshape(-1)),
            1,
        )
        weights = group_multiplicity[:, inverse]
        totals = np.sum(weights, axis=1)
        estimates[start:stop] = np.sum(weights * improvement[None, :], axis=1) / totals
    return estimates


def _policy_metrics(
    decisions: _DecisionSet,
    router: Mapping[str, Any],
    policy: str,
    budget: float,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    query, choice, final_loss, final_confidence = _policy_arrays(decisions, router, policy, budget, seed)
    selected_value = decisions.value[np.arange(len(choice)), choice]
    improvement = decisions.base_loss - final_loss
    queried = selected_value[query]
    optimal = np.max(decisions.value, axis=1)
    has_choice = decisions.value.shape[1] > 1
    candidate_tie = np.sum(decisions.value == optimal[:, None], axis=1) > 1
    source_correct = (selected_value == optimal) & ~candidate_tie
    unequal = ~candidate_tie if has_choice else np.zeros(len(candidate_tie), dtype=bool)
    unequal_query = query & unequal
    predictions = np.where(final_loss == 0, decisions.labels, 1 - decisions.labels)
    base_predictions = np.where(decisions.base_loss == 0, decisions.labels, 1 - decisions.labels)
    useful_available = optimal > 0
    useful_selected = query & (selected_value > 0)
    group_statistics = []
    for group in np.unique(decisions.groups):
        selected_group = decisions.groups == group
        group_statistics.append(
            {
                "group_sha256": _group_digest([str(group)]),
                "decision_count": int(np.sum(selected_group)),
                "error_reduction_sum": float(np.sum(improvement[selected_group])),
            }
        )
    metrics = {
        "query_count": int(query.sum()),
        "query_rate": float(np.mean(query)),
        "base_error": float(np.mean(decisions.base_loss)),
        "final_error": float(np.mean(final_loss)),
        "base_macro_f1": _binary_macro_f1(decisions.labels, base_predictions),
        "final_macro_f1": _binary_macro_f1(decisions.labels, predictions),
        "error_reduction": float(np.mean(improvement)),
        "error_reduction_video_macro": _group_macro(improvement, decisions.groups),
        "harmful_query_rate": None if not len(queried) else float(np.mean(queried < 0)),
        "useful_query_precision": None if not len(queried) else float(np.mean(queried > 0)),
        "useful_query_recall": None if not np.any(useful_available) else float(np.sum(useful_selected) / np.sum(useful_available)),
        "candidate_value_tie_rate": None if not has_choice else float(np.mean(candidate_tie)),
        "candidate_ranking_evaluable": int(np.sum(unequal)),
        "candidate_ranking_accuracy": None if not np.any(unequal) else float(np.mean(source_correct[unequal])),
        "queried_candidate_selection_accuracy": (
            None if not np.any(unequal_query) else float(np.mean(source_correct[unequal_query]))
        ),
        "risk_coverage_auc": _risk_coverage_auc(final_loss, final_confidence),
        "video_group_statistics": group_statistics,
    }
    return metrics, final_confidence


def _coverage_curves(
    selection: _DecisionSet,
    evaluation: _DecisionSet,
    router: Mapping[str, Any],
    policy: str,
    budget: float,
    coverage_targets: Sequence[float],
    seed: int,
) -> dict[str, Any]:
    curves: dict[str, Any] = {}
    for target in coverage_targets:
        target_value = float(target)
        state = _selective_state(selection, evaluation, router, policy, budget, target_value, seed)
        selection_accepted = state["selection_accepted"]
        accepted = state["evaluation_accepted"]
        selection_loss = state["selection_loss"]
        evaluation_loss = state["evaluation_loss"]
        curves[f"{target_value:.6f}"] = {
            "threshold": state["threshold"],
            "coverage": float(np.mean(accepted)),
            "abstention_rate": float(1.0 - np.mean(accepted)),
            "selective_risk": None if not np.any(accepted) else float(np.mean(evaluation_loss[accepted])),
            "selection_error_at_threshold": (
                None if not np.any(selection_accepted) else float(np.mean(selection_loss[selection_accepted]))
            ),
            "accepted_count": int(np.sum(accepted)),
            "accepted_error_sum": float(np.sum(evaluation_loss[accepted])),
            "selection_accepted_count": int(np.sum(selection_accepted)),
            "selection_accepted_error_sum": float(np.sum(selection_loss[selection_accepted])),
        }
    return curves


def _selective_state(
    selection: _DecisionSet,
    evaluation: _DecisionSet,
    router: Mapping[str, Any],
    policy: str,
    budget: float,
    target: float,
    seed: int,
) -> dict[str, Any]:
    if target <= 0 or target > 1:
        raise ValueError("coverage targets must lie within (0, 1]")
    _, _, selection_loss, selection_confidence = _policy_arrays(selection, router, policy, budget, seed)
    _, choice, evaluation_loss, evaluation_confidence = _policy_arrays(
        evaluation, router, policy, budget, seed
    )
    evaluation_priority, priority_choice = _policy_priority_and_choice(
        evaluation, router, policy, seed
    )
    if not np.array_equal(choice, priority_choice):
        raise RuntimeError("policy choice changed between evaluation passes")
    base_confidence, candidate_confidence = _calibrated_confidence(evaluation, router)
    queried_loss = evaluation.candidate_loss[np.arange(len(choice)), choice]
    queried_confidence = candidate_confidence[np.arange(len(choice)), choice]
    # Full coverage is a protocol endpoint, not a calibrated operating point.
    # A selection-derived minimum can exclude lower-confidence evaluation cases.
    threshold = (
        -float(np.finfo(np.float64).max)
        if target == 1.0
        else float(np.quantile(selection_confidence, 1.0 - target, method="lower"))
    )
    return {
        "threshold": threshold,
        "selection_loss": selection_loss,
        "selection_accepted": selection_confidence >= threshold,
        "evaluation_loss": evaluation_loss,
        "evaluation_confidence": evaluation_confidence,
        "evaluation_accepted": evaluation_confidence >= threshold,
        "query_priority": evaluation_priority,
        "query_order_seed": int(seed) + 2_000_003,
        "query_budget": float(budget),
        "base_loss": np.asarray(evaluation.base_loss, dtype=np.float64),
        "queried_loss": np.asarray(queried_loss, dtype=np.float64),
        "base_confidence": np.asarray(base_confidence, dtype=np.float64),
        "queried_confidence": np.asarray(queried_confidence, dtype=np.float64),
    }


def _nested_selective_risk_draws(
    state: Mapping[str, Any],
    weights: np.ndarray,
    target_coverage: float,
    tie_seed: int,
) -> np.ndarray:
    """Reselect the exact query budget, then exact coverage, in each replicate."""

    priority = np.asarray(state["query_priority"], dtype=np.float64)
    query_order = _seeded_descending_order(priority, int(state["query_order_seed"]))
    query_weights, _ = _weighted_topk_selection(
        weights,
        query_order,
        float(state["query_budget"]),
    )
    unqueried_weights = np.asarray(weights, dtype=np.int64) - query_weights
    outcome_weights = np.concatenate((query_weights, unqueried_weights), axis=1)
    outcome_loss = np.concatenate(
        (
            np.asarray(state["queried_loss"], dtype=np.float64),
            np.asarray(state["base_loss"], dtype=np.float64),
        )
    )
    outcome_confidence = np.concatenate(
        (
            np.asarray(state["queried_confidence"], dtype=np.float64),
            np.asarray(state["base_confidence"], dtype=np.float64),
        )
    )
    # Give queried and unqueried copies of one original decision the same tie
    # key. Otherwise changing only query allocation can change which identities
    # survive a confidence tie and create a spurious paired difference.
    identity_tie_key = np.random.default_rng(tie_seed).random(len(priority))
    outcome_tie_key = np.concatenate((identity_tie_key, identity_tie_key))
    outcome_state_key = np.concatenate(
        (np.zeros(len(priority), dtype=np.int64), np.ones(len(priority), dtype=np.int64))
    )
    coverage_order = np.lexsort((outcome_state_key, outcome_tie_key, -outcome_confidence))
    error_sums, accepted_counts = _weighted_topk_sums(
        outcome_weights,
        coverage_order,
        outcome_loss,
        (target_coverage,),
    )
    if np.any(accepted_counts[:, 0] <= 0):
        raise RuntimeError("selective bootstrap accepted no decisions")
    return error_sums[:, 0] / accepted_counts[:, 0]


def _exact_coverage_mask(confidence: np.ndarray, target: float, seed: int) -> np.ndarray:
    """Accept exactly round(target * n) items using confidence and seeded ties."""

    values = np.asarray(confidence, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or target <= 0 or target > 1:
        raise ValueError("exact coverage requires finite confidence and a target in (0, 1]")
    count = int(np.rint(float(target) * len(values)))
    accepted = np.zeros(len(values), dtype=bool)
    accepted[_seeded_descending_order(values, seed)[:count]] = True
    return accepted


def _paired_selective_risk_contrast(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    groups: np.ndarray,
    repetitions: int,
    target_coverage: float,
    tie_seed: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    left_confidence = np.asarray(left["evaluation_confidence"], dtype=np.float64)
    right_confidence = np.asarray(right["evaluation_confidence"], dtype=np.float64)
    left_accepted = _exact_coverage_mask(left_confidence, target_coverage, tie_seed)
    right_accepted = _exact_coverage_mask(right_confidence, target_coverage, tie_seed)
    left_loss = np.asarray(left["evaluation_loss"], dtype=np.float64)
    right_loss = np.asarray(right["evaluation_loss"], dtype=np.float64)
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(bootstrap_seed)
    differences = np.empty(repetitions, dtype=np.float64)
    chunk_size = min(250, repetitions)
    for start in range(0, repetitions, chunk_size):
        stop = min(start + chunk_size, repetitions)
        rows = stop - start
        selected_groups = rng.integers(0, len(unique), size=(rows, len(unique)))
        group_multiplicity = np.zeros((rows, len(unique)), dtype=np.int32)
        np.add.at(
            group_multiplicity,
            (np.repeat(np.arange(rows), len(unique)), selected_groups.reshape(-1)),
            1,
        )
        weights = group_multiplicity[:, inverse]
        left_risk_draws = _nested_selective_risk_draws(
            left, weights, target_coverage, tie_seed
        )
        right_risk_draws = _nested_selective_risk_draws(
            right, weights, target_coverage, tie_seed
        )
        differences[start:stop] = right_risk_draws - left_risk_draws
    interval = [float(value) for value in np.quantile(differences, (0.025, 0.975))]
    left_risk = None if not np.any(left_accepted) else float(np.mean(left_loss[left_accepted]))
    right_risk = None if not np.any(right_accepted) else float(np.mean(right_loss[right_accepted]))
    group_statistics = []
    for group in unique:
        selected = groups == group
        group_statistics.append(
            {
                "group_sha256": _group_digest([str(group)]),
                "left_accepted_count": int(np.sum(left_accepted[selected])),
                "left_accepted_error_sum": float(np.sum(left_loss[selected] * left_accepted[selected])),
                "right_accepted_count": int(np.sum(right_accepted[selected])),
                "right_accepted_error_sum": float(np.sum(right_loss[selected] * right_accepted[selected])),
            }
        )
    return {
        "coverage_definition": "exact_evaluation_count",
        "target_coverage": float(target_coverage),
        "accepted_count": int(np.sum(left_accepted)),
        "left_coverage": float(np.mean(left_accepted)),
        "right_coverage": float(np.mean(right_accepted)),
        "left_selective_risk": left_risk,
        "right_selective_risk": right_risk,
        "selective_risk_difference": (
            None if left_risk is None or right_risk is None else float(right_risk - left_risk)
        ),
        "cluster_ci95": interval,
        "bootstrap_draws": [float(value) for value in differences],
        "bootstrap_repetitions": int(repetitions),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_method": (
            "video_cluster_resample_with_exact_query_budget_and_exact_coverage_reselection"
        ),
        "video_group_statistics": group_statistics,
    }


def _choose_matched_baseline(
    selection: _DecisionSet,
    router: Mapping[str, Any],
    budget: float,
    policy_seeds: Mapping[str, int],
) -> dict[str, Any]:
    candidates: dict[str, float] = {}
    for policy in ("source_prior", "confidence"):
        _, _, final_loss, _ = _policy_arrays(selection, router, policy, budget, int(policy_seeds[policy]))
        candidates[policy] = float(np.mean(final_loss))
    chosen = min(candidates, key=lambda name: (candidates[name], name != "source_prior", name))
    return {
        "policy": chosen,
        "selection_error": candidates[chosen],
        "candidate_errors": candidates,
        "policy_seeds": {name: int(policy_seeds[name]) for name in candidates},
    }


def _paired_policy_contrast(
    decisions: _DecisionSet,
    router: Mapping[str, Any],
    left_policy: str,
    right_policy: str,
    budget: float,
    left_seed: int,
    right_seed: int,
    reselected_cluster_ci95: Sequence[float],
) -> dict[str, Any]:
    _, _, left_loss, _ = _policy_arrays(decisions, router, left_policy, budget, left_seed)
    _, _, right_loss, _ = _policy_arrays(decisions, router, right_policy, budget, right_seed)
    paired_gain = right_loss - left_loss
    group_statistics = []
    for group in np.unique(decisions.groups):
        selected = decisions.groups == group
        group_statistics.append(
            {
                "group_sha256": _group_digest([str(group)]),
                "decision_count": int(np.sum(selected)),
                "error_reduction_difference_sum": float(np.sum(paired_gain[selected])),
            }
        )
    return {
        "left_policy": left_policy,
        "right_policy": right_policy,
        "error_reduction_difference": float(np.mean(paired_gain)),
        "video_macro_difference": _group_macro(paired_gain, decisions.groups),
        "cluster_ci95": [float(value) for value in reselected_cluster_ci95],
        "bootstrap_method": "video_cluster_resample_with_exact_budget_reselection",
        "video_group_statistics": group_statistics,
    }


def _context_diagnostics(
    decisions: _DecisionSet,
    predicted_value: np.ndarray,
    mode: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for context_index, (observed_mask, candidates) in enumerate(_contexts(mode)):
        selected = decisions.context_index == context_index
        name = "+".join(MODALITIES[index] for index in np.flatnonzero(observed_mask))
        candidate_results: dict[str, Any] = {}
        for column, candidate in enumerate(candidates):
            values = decisions.value[selected, column]
            estimates = predicted_value[selected, column]
            candidate_results[MODALITIES[candidate]] = {
                "mean_realized_value": float(np.mean(values)),
                "positive_value_rate": float(np.mean(values > 0)),
                "negative_value_rate": float(np.mean(values < 0)),
                "value_mean_squared_error": float(np.mean((estimates - values) ** 2)),
            }
        by_label = {}
        for label in (0, 1):
            label_selected = selected & (decisions.labels == label)
            by_label[str(label)] = {
                "decisions": int(np.sum(label_selected)),
                "base_error": float(np.mean(decisions.base_loss[label_selected])),
                "mean_oracle_value": float(np.mean(np.max(decisions.value[label_selected], axis=1))),
            }
        output[name] = {
            "decisions": int(np.sum(selected)),
            "base_error": float(np.mean(decisions.base_loss[selected])),
            "candidate_modalities": [MODALITIES[index] for index in candidates],
            "candidates": candidate_results,
            "by_label": by_label,
        }
    return output


def _subgroup_policy_summary(
    decisions: _DecisionSet,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    selected: np.ndarray,
) -> dict[str, Any]:
    query, choice, final_loss, _ = arrays
    indices = np.flatnonzero(selected)
    local_query = query[indices]
    local_choice = choice[indices]
    local_final_loss = final_loss[indices]
    local_base_loss = decisions.base_loss[indices]
    labels = decisions.labels[indices]
    selected_value = decisions.value[indices, local_choice]
    optimal = np.max(decisions.value[indices], axis=1)
    has_choice = decisions.value.shape[1] > 1
    ties = np.sum(decisions.value[indices] == optimal[:, None], axis=1) > 1
    evaluable = ~ties if has_choice else np.zeros(len(indices), dtype=bool)
    queried_evaluable = local_query & evaluable
    target_modality = decisions.candidate_modality[indices, local_choice]
    target_counts = {
        MODALITIES[modality]: int(np.sum(local_query & (target_modality == modality)))
        for modality in range(3)
    }
    target_outcomes: dict[str, Any] = {}
    for modality in range(3):
        use = local_query & (target_modality == modality)
        target_outcomes[MODALITIES[modality]] = {
            "query_count": int(np.sum(use)),
            "realized_value_sum": float(np.sum(selected_value[use])),
            "harmful_query_count": int(np.sum(selected_value[use] < 0)),
            "mean_realized_value": None if not np.any(use) else float(np.mean(selected_value[use])),
            "harmful_query_rate": None if not np.any(use) else float(np.mean(selected_value[use] < 0)),
        }
    predictions = np.where(local_final_loss == 0, labels, 1 - labels)
    base_predictions = np.where(local_base_loss == 0, labels, 1 - labels)
    summary: dict[str, Any] = {
        "decisions": int(len(indices)),
        "query_count": int(np.sum(local_query)),
        "query_rate": float(np.mean(local_query)),
        "base_error": float(np.mean(local_base_loss)),
        "final_error": float(np.mean(local_final_loss)),
        "base_macro_f1": _binary_macro_f1(labels, base_predictions),
        "final_macro_f1": _binary_macro_f1(labels, predictions),
        "error_reduction": float(np.mean(local_base_loss - local_final_loss)),
        "harmful_query_rate": (
            None if not np.any(local_query) else float(np.mean(selected_value[local_query] < 0))
        ),
        "useful_query_precision": (
            None if not np.any(local_query) else float(np.mean(selected_value[local_query] > 0))
        ),
        "candidate_ranking_evaluable": int(np.sum(evaluable)),
        "candidate_ranking_accuracy": (
            None if not np.any(evaluable) else float(np.mean(selected_value[evaluable] == optimal[evaluable]))
        ),
        "queried_candidate_selection_accuracy": (
            None
            if not np.any(queried_evaluable)
            else float(np.mean(selected_value[queried_evaluable] == optimal[queried_evaluable]))
        ),
        "query_target_counts": target_counts,
        "query_target_outcomes": target_outcomes,
        "by_label": {},
    }
    for label in (0, 1):
        use = labels == label
        summary["by_label"][str(label)] = {
            "decisions": int(np.sum(use)),
            "query_rate": float(np.mean(local_query[use])),
            "base_error": float(np.mean(local_base_loss[use])),
            "final_error": float(np.mean(local_final_loss[use])),
            "error_reduction": float(np.mean(local_base_loss[use] - local_final_loss[use])),
        }
    return summary


def _attach_context_policy_outcomes(
    context_records: dict[str, Any],
    decisions: _DecisionSet,
    policy_arrays: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    mode: str,
) -> None:
    for context_index, (observed_mask, _) in enumerate(_contexts(mode)):
        name = "+".join(MODALITIES[index] for index in np.flatnonzero(observed_mask))
        selected = decisions.context_index == context_index
        context_records[name]["policy_outcomes"] = {}
        for (policy, budget), arrays in policy_arrays.items():
            context_records[name]["policy_outcomes"].setdefault(policy, {})[budget] = _subgroup_policy_summary(
                decisions,
                arrays,
                selected,
            )


def run_value_study(
    cache_path: str | pathlib.Path,
    config: Mapping[str, Any],
    *,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    """Run one deterministic train-grouped pilot on official validation data."""

    train = _load_score_split(cache_path, config, "train")
    task_indices, router_indices, calibration_indices = _partition_indices(
        train.groups,
        config["train_group_fractions"],
        int(seed),
    )
    for name, indices in (
        ("task_fit", task_indices),
        ("router_fit", router_indices),
        ("calibration", calibration_indices),
    ):
        if len(np.unique(train.y[indices])) != 2:
            raise ValueError(f"{name} partition must contain both labels")
    mean, scale = _fit_standardizer(train.x[task_indices])
    train_z = (train.x - mean) / scale
    ridge = float(config.get("ridge", 0.01))
    confidence_ridge = float(config.get("confidence_ridge", 0.1))
    task_weights = _fit_task_model(train_z[task_indices], train.y[task_indices], ridge)
    router_decisions = _make_decisions(
        train_z[router_indices], train.y[router_indices], train.groups[router_indices], task_weights, mode
    )
    calibration_decisions = _make_decisions(
        train_z[calibration_indices],
        train.y[calibration_indices],
        train.groups[calibration_indices],
        task_weights,
        mode,
    )
    router, router_ladder_record = _fit_router_ladder(
        router_decisions,
        calibration_decisions,
        config,
        seed=int(seed),
    )
    router["confidence_models"] = _fit_confidence_models(calibration_decisions, confidence_ridge)
    valid = _load_score_split(cache_path, config, "valid")
    _validate_split_isolation(train, valid)
    valid_z = (valid.x - mean) / scale
    full_mask = np.ones(3, dtype=bool)
    full_avt_score = _task_scores(valid_z, full_mask, task_weights)
    full_avt_accuracy = float(np.mean((full_avt_score >= 0).astype(np.int64) == valid.y))
    validation_accuracy_by_mask = {
        _mask_name(mask): float(
            np.mean((_task_scores(valid_z, mask, task_weights) >= 0).astype(np.int64) == valid.y)
        )
        for mask in _observed_masks()
    }
    evaluation_decisions = _make_decisions(valid_z, valid.y, valid.groups, task_weights, mode)
    family_predictions = {
        family: _router_predictions(evaluation_decisions, router, family)[0]
        for family in ROUTER_FAMILIES
    }
    predicted_value = family_predictions[str(router["selected_family"])]
    _, benefit_probability = _router_predictions(
        evaluation_decisions,
        router,
        str(router["selected_family"]),
    )
    router_ladder_record["official_validation_prediction_sha256"] = {
        family: _array_sha256(predictions)
        for family, predictions in family_predictions.items()
    }
    router_ladder_record["attestation_sha256"] = _json_sha256(router_ladder_record)
    base_correctness_probability, candidate_correctness_probability = _calibrated_confidence(
        evaluation_decisions, router
    )
    benefit_target = (evaluation_decisions.value > 0).astype(np.float64)
    clipped = np.clip(benefit_probability, 1e-8, 1.0 - 1e-8)

    repetitions = int(config["bootstrap_repetitions"])
    policies: dict[str, Any] = {}
    coverage: dict[str, Any] = {}
    saved_policy_arrays: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    baseline_selection: dict[str, Any] = {}
    sorted_budgets = sorted(float(value) for value in config["budgets"])
    policy_names = ROUTER_POLICIES
    for budget_index, budget in enumerate(sorted_budgets):
        key = f"{budget:.6f}"
        policy_seeds = {
            policy: _policy_seed(int(seed), policy, router)
            for policy in ("source_prior", "confidence")
        }
        baseline_selection[key] = _choose_matched_baseline(
            calibration_decisions,
            router,
            budget,
            policy_seeds,
        )
    coverage_targets = tuple(float(value) for value in config.get("coverage_targets", (1.0, 0.9, 0.8)))
    concrete_policies = tuple(
        policy for policy in policy_names if policy not in {"selected_router", "selected_baseline"}
    )
    for policy in concrete_policies:
        policies[policy] = {}
        coverage[policy] = {}
        for budget in sorted_budgets:
            key = f"{budget:.6f}"
            local_seed = _policy_seed(int(seed), policy, router)
            metrics, _ = _policy_metrics(
                evaluation_decisions,
                router,
                policy,
                budget,
                local_seed,
            )
            policies[policy][key] = metrics
            saved_policy_arrays[(policy, key)] = _policy_arrays(
                evaluation_decisions,
                router,
                policy,
                budget,
                local_seed,
            )
            coverage[policy][key] = _coverage_curves(
                calibration_decisions,
                evaluation_decisions,
                router,
                policy,
                budget,
                coverage_targets,
                local_seed,
            )
    selected_family = str(router["selected_family"])
    policies["selected_router"] = copy.deepcopy(policies[selected_family])
    coverage["selected_router"] = copy.deepcopy(coverage[selected_family])
    for budget in sorted_budgets:
        key = f"{budget:.6f}"
        saved_policy_arrays[("selected_router", key)] = saved_policy_arrays[(selected_family, key)]
    policies["selected_baseline"] = {}
    coverage["selected_baseline"] = {}
    for budget in sorted_budgets:
        key = f"{budget:.6f}"
        chosen = str(baseline_selection[key]["policy"])
        policies["selected_baseline"][key] = copy.deepcopy(policies[chosen][key])
        policies["selected_baseline"][key]["selected_policy"] = chosen
        coverage["selected_baseline"][key] = copy.deepcopy(coverage[chosen][key])
        saved_policy_arrays[("selected_baseline", key)] = saved_policy_arrays[(chosen, key)]

    online_priority, online_choice = _policy_priority_and_choice(
        evaluation_decisions,
        router,
        "online_selected_router",
        _policy_seed(int(seed), "online_selected_router", router),
    )
    online_evaluation_by_budget: dict[str, Any] = {}
    for budget in sorted_budgets:
        key = f"{budget:.6f}"
        fitted = router["online_query_thresholds"][key]
        query = saved_policy_arrays[("online_selected_router", key)][0]
        online_evaluation_by_budget[key] = {
            "target_query_rate": float(fitted["target_query_rate"]),
            "threshold": float(fitted["threshold"]),
            "query_on_equal": bool(fitted["query_on_equal"]),
            "tie_convention": str(fitted["tie_convention"]),
            "evaluation_decisions": len(query),
            "evaluation_query_count": int(np.sum(query)),
            "evaluation_realized_query_rate": float(np.mean(query)),
        }
    online_evaluation_record = {
        "protocol": "fixed_calibration_threshold_per_example",
        "fit_split": "calibration",
        "selected_family": selected_family,
        "priority": "maximum_selected_router_predicted_value",
        "candidate_choice": "argmax_selected_router_predicted_value",
        "candidate_tie_convention": "first_candidate_in_locked_modality_order",
        "evaluation_batch_access": "independent_per_example",
        "priority_sha256": _array_sha256(online_priority),
        "choice_sha256": _array_sha256(online_choice),
        "by_target_budget": online_evaluation_by_budget,
    }
    online_evaluation_record["attestation_sha256"] = _json_sha256(
        online_evaluation_record
    )

    bootstrap_seed = int(config.get("bootstrap_seed", 20_270_917))
    bootstrap_draws = {
        policy: _policy_reselected_cluster_bootstrap(
            evaluation_decisions,
            router,
            policy,
            sorted_budgets,
            repetitions,
            _policy_seed(int(seed), policy, router),
            bootstrap_seed,
        )
        for policy in BASE_BOOTSTRAP_POLICIES
    }
    bootstrap_draws["selected_router"] = bootstrap_draws[selected_family]
    bootstrap_draws["online_selected_router"] = np.column_stack(
        [
            _fixed_threshold_cluster_bootstrap(
                evaluation_decisions,
                saved_policy_arrays[("online_selected_router", f"{budget:.6f}")][0],
                saved_policy_arrays[("online_selected_router", f"{budget:.6f}")][1],
                repetitions,
                bootstrap_seed,
            )
            for budget in sorted_budgets
        ]
    )
    for policy in policy_names:
        for budget_index, budget in enumerate(sorted_budgets):
            key = f"{budget:.6f}"
            effective_policy = (
                str(baseline_selection[key]["policy"])
                if policy == "selected_baseline"
                else policy
            )
            draws = bootstrap_draws[effective_policy][:, budget_index]
            policies[policy][key]["error_reduction_cluster_ci95"] = [
                float(value) for value in np.quantile(draws, (0.025, 0.975))
            ]
            policies[policy][key]["bootstrap_method"] = (
                "video_cluster_resample_with_fixed_calibration_threshold"
                if policy == "online_selected_router"
                else "video_cluster_resample_with_exact_budget_reselection"
            )

    paired_contrasts: dict[str, Any] = {}
    for budget_index, budget in enumerate(sorted_budgets):
        key = f"{budget:.6f}"
        chosen_baseline = str(baseline_selection[key]["policy"])
        left_seed = _policy_seed(int(seed), "selected_router", router)
        right_seed = _policy_seed(int(seed), chosen_baseline, router)
        paired_draws = (
            bootstrap_draws["selected_router"][:, budget_index]
            - bootstrap_draws[chosen_baseline][:, budget_index]
        )
        paired_contrasts[key] = _paired_policy_contrast(
            evaluation_decisions,
            router,
            "selected_router",
            chosen_baseline,
            budget,
            left_seed,
            right_seed,
            [float(value) for value in np.quantile(paired_draws, (0.025, 0.975))],
        )
        paired_contrasts[key]["bootstrap_draws"] = [float(value) for value in paired_draws]
        left_state = _selective_state(
            calibration_decisions,
            evaluation_decisions,
            router,
            "selected_router",
            budget,
            0.9,
            left_seed,
        )
        right_state = _selective_state(
            calibration_decisions,
            evaluation_decisions,
            router,
            chosen_baseline,
            budget,
            0.9,
            right_seed,
        )
        paired_contrasts[key]["coverage_0.90"] = _paired_selective_risk_contrast(
            left_state,
            right_state,
            evaluation_decisions.groups,
            repetitions,
            0.9,
            int(seed) * 3571 + budget_index,
            bootstrap_seed + 10_000 + budget_index,
        )
    for key in policies["oracle"]:
        oracle_reduction = float(policies["oracle"][key]["error_reduction"])
        for policy in policies:
            if policy == "online_selected_router":
                realized_rate = float(policies[policy][key]["query_rate"])
                matched_oracle, _ = _policy_metrics(
                    evaluation_decisions,
                    router,
                    "oracle",
                    realized_rate,
                    _policy_seed(int(seed), "oracle", router),
                )
                matched_reduction = float(matched_oracle["error_reduction"])
                policies[policy][key]["oracle_reference_query_rate"] = realized_rate
                policies[policy][key][
                    "oracle_reference_error_reduction"
                ] = matched_reduction
                policies[policy][key]["oracle_regret"] = matched_reduction - float(
                    policies[policy][key]["error_reduction"]
                )
            else:
                policies[policy][key]["oracle_regret"] = oracle_reduction - float(
                    policies[policy][key]["error_reduction"]
                )

    headroom_budget = float(config["headroom_budget"])
    headroom_key = f"{headroom_budget:.6f}"
    if headroom_key not in policies["oracle"]:
        raise ValueError("headroom_budget must appear in budgets")
    headroom = policies["oracle"][headroom_key]
    minimum = float(config["minimum_oracle_error_reduction"])
    lower = float(headroom["error_reduction_cluster_ci95"][0])
    gate_passed = float(headroom["error_reduction"]) >= minimum and lower > 0.0
    reference_accuracy = float(config["reference_full_avt_validation_accuracy"])
    maximum_accuracy_gap = float(config["maximum_task_head_accuracy_gap"])
    minimum_mask_accuracy = float(config["minimum_mask_validation_accuracy"])
    task_head_passed = (
        full_avt_accuracy >= reference_accuracy - maximum_accuracy_gap
        and min(validation_accuracy_by_mask.values()) >= minimum_mask_accuracy
    )

    partition_groups = {
        "task_fit": train.groups[task_indices],
        "router_fit": train.groups[router_indices],
        "calibration": train.groups[calibration_indices],
    }
    sets = [set(values) for values in partition_groups.values()]
    pairwise_disjoint = all(not sets[left].intersection(sets[right]) for left in range(3) for right in range(left + 1, 3))
    source = pathlib.Path(cache_path).expanduser().resolve()
    context_records = _context_diagnostics(evaluation_decisions, predicted_value, mode)
    _attach_context_policy_outcomes(context_records, evaluation_decisions, saved_policy_arrays, mode)
    training_group_hashes, training_group_digest = _validation_group_record(train.groups)
    validation_group_hashes, validation_group_digest = _validation_group_record(valid.groups)
    partition_records: dict[str, Any] = {}
    for (name, values), indices in zip(
        partition_groups.items(),
        (task_indices, router_indices, calibration_indices),
    ):
        group_hashes, group_digest = _validation_group_record(values)
        partition_records[name] = {
            "video_groups": int(len(np.unique(values))),
            "utterances": int(len(indices)),
            "group_hashes": group_hashes,
            "group_digest": group_digest,
            "positive_label_rate": float(np.mean(train.y[indices])),
        }
    partition_records["pairwise_disjoint"] = bool(pairwise_disjoint)
    result = {
        "schema": RESULT_SCHEMA,
        "study_schema": STUDY_SCHEMA,
        "configuration_sha256": configuration_sha256(config),
        "implementation_sha256": implementation_sha256(),
        "mode": mode,
        "seed": int(seed),
        "data_access": {
            "cache_sha256": _sha256(source),
            "loaded_splits": ["train", "valid"],
            "test_opened": False,
            "train_utterances": int(len(train.y)),
            "validation_utterances": int(len(valid.y)),
            "train_video_groups": int(len(np.unique(train.groups))),
            "validation_video_groups": int(len(np.unique(valid.groups))),
            "training_group_hashes": training_group_hashes,
            "training_group_sha256": training_group_digest,
            "validation_group_hashes": validation_group_hashes,
            "validation_group_sha256": validation_group_digest,
            "input_representation": str(config.get("input_representation", "unspecified")),
            "upstream_projection_scope": str(config.get("upstream_projection_scope", "unspecified")),
            "evidence_status": str(config.get("evidence_status", "exploratory")),
        },
        "group_partitions": partition_records,
        "router_ladder": router_ladder_record,
        "evaluation": {
            "split": "official_validation",
            "decision_instances": int(len(evaluation_decisions.base_loss)),
            "candidate_instances": int(evaluation_decisions.value.size),
            "policies": policies,
            "online_query_policy": online_evaluation_record,
            "selective_curves": coverage,
            "baseline_selection": baseline_selection,
            "paired_contrasts": paired_contrasts,
            "contexts": context_records,
            "value_regression": {
                "mean_squared_error": float(np.mean((predicted_value - evaluation_decisions.value) ** 2)),
                "mean_absolute_error": float(np.mean(np.abs(predicted_value - evaluation_decisions.value))),
            },
            "benefit_calibration": {
                "brier": float(np.mean((benefit_probability - benefit_target) ** 2)),
                "log_loss": float(
                    -np.mean(benefit_target * np.log(clipped) + (1.0 - benefit_target) * np.log(1.0 - clipped))
                ),
            },
            "task_confidence_calibration": {
                "scope": "per_observation_mask_on_calibration_groups",
                "base_brier": float(
                    np.mean((base_correctness_probability - (1.0 - evaluation_decisions.base_loss)) ** 2)
                ),
                "candidate_brier": float(
                    np.mean(
                        (candidate_correctness_probability - (1.0 - evaluation_decisions.candidate_loss)) ** 2
                    )
                ),
            },
        },
        "headroom_gate": {
            "budget": headroom_budget,
            "minimum_oracle_error_reduction": minimum,
            "observed_oracle_error_reduction": float(headroom["error_reduction"]),
            "cluster_ci95": list(headroom["error_reduction_cluster_ci95"]),
            "bootstrap_draws": [
                float(value)
                for value in bootstrap_draws["oracle"][:, sorted_budgets.index(headroom_budget)]
            ],
            "passed": bool(gate_passed),
        },
        "task_head_gate": {
            "predictor_scope": "separate_frozen_ridge_per_observation_mask_fit_on_task_groups",
            "validation_accuracy_by_mask": validation_accuracy_by_mask,
            "full_avt_validation_accuracy": full_avt_accuracy,
            "reference_name": str(config["task_head_reference_name"]),
            "reference_validation_accuracy": reference_accuracy,
            "reference_record_sha256": str(config["task_head_reference_sha256"]),
            "maximum_accuracy_gap": maximum_accuracy_gap,
            "minimum_mask_validation_accuracy": minimum_mask_accuracy,
            "passed": bool(task_head_passed),
        },
        "bootstrap": {
            "group_draw_seed": bootstrap_seed,
            "repetitions": repetitions,
            "method": "video_cluster_resample_with_policy_reselection",
            "online_policy_method": (
                "video_cluster_resample_with_fixed_calibration_threshold"
            ),
            "policy_error_reduction_draws": {
                policy: {
                    f"{budget:.6f}": [float(value) for value in draws[:, budget_index]]
                    for budget_index, budget in enumerate(sorted_budgets)
                }
                for policy, draws in bootstrap_draws.items()
            },
        },
        "validation": {"ok": bool(pairwise_disjoint), "errors": [] if pairwise_disjoint else ["training groups overlap"]},
    }
    return result


def validate_value_result(
    record: Mapping[str, Any],
    *,
    expected_mode: str | None = None,
    expected_seed: int | None = None,
    expected_cache_sha256: str | None = None,
    expected_configuration_sha256: str | None = None,
    expected_config: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return validation errors for a saved value-study result."""

    errors: list[str] = []

    def finite_number(value: Any) -> bool:
        return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.integer, np.floating)) and bool(
            np.isfinite(value)
        )

    def valid_rate(value: Any, *, nullable: bool = False) -> bool:
        if nullable and value is None:
            return True
        return finite_number(value) and -1e-12 <= float(value) <= 1.0 + 1e-12

    def valid_count(value: Any, *, maximum: int | None = None) -> bool:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            return False
        return int(value) >= 0 and (maximum is None or int(value) <= maximum)

    def macro_f1_from_label_errors(
        label_counts: Mapping[str, float],
        label_errors: Mapping[str, float],
    ) -> float:
        class_scores: list[float] = []
        for label in ("0", "1"):
            other = "1" if label == "0" else "0"
            true_positive = float(label_counts[label]) - float(label_errors[label])
            false_negative = float(label_errors[label])
            false_positive = float(label_errors[other])
            denominator = 2.0 * true_positive + false_positive + false_negative
            class_scores.append(0.0 if denominator == 0.0 else 2.0 * true_positive / denominator)
        return float(np.mean(class_scores))

    if record.get("schema") != RESULT_SCHEMA:
        errors.append("unexpected result schema")
    if record.get("study_schema") != STUDY_SCHEMA:
        errors.append("unexpected study schema")
    configuration_digest = record.get("configuration_sha256")
    if not isinstance(configuration_digest, str) or re.fullmatch(r"[0-9a-f]{64}", configuration_digest) is None:
        errors.append("configuration SHA256 is invalid")
    if expected_configuration_sha256 is not None and configuration_digest != expected_configuration_sha256:
        errors.append("configuration SHA256 does not match the expected value")
    if expected_config is not None and configuration_digest != configuration_sha256(expected_config):
        errors.append("configuration SHA256 does not match the supplied configuration")
    implementation_digest = record.get("implementation_sha256")
    if (
        not isinstance(implementation_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", implementation_digest) is None
        or implementation_digest != implementation_sha256()
    ):
        errors.append("implementation SHA256 does not match the active scientific code")
    mode = record.get("mode")
    if mode not in {"singleton", "pair"}:
        errors.append("invalid mode")
    if expected_mode is not None and mode != expected_mode:
        errors.append("mode does not match the expected value")
    if expected_seed is not None and record.get("seed") != expected_seed:
        errors.append("seed does not match the expected value")
    access = record.get("data_access")
    training_group_hashes: set[str] = set()
    validation_group_hashes: set[str] = set()
    if not isinstance(access, Mapping):
        errors.append("data access record is missing")
    else:
        if access.get("loaded_splits") != ["train", "valid"] or access.get("test_opened") is not False:
            errors.append("pilot data access is not restricted to training and validation")
        digest = access.get("cache_sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            errors.append("cache SHA256 is invalid")
        if expected_cache_sha256 is not None and digest != expected_cache_sha256:
            errors.append("cache SHA256 does not match the expected value")
        if expected_config is not None:
            expected_counts = expected_config.get("expected_split_counts", {})
            if access.get("train_utterances") != expected_counts.get("train"):
                errors.append("training utterance count does not match the configuration")
            if access.get("validation_utterances") != expected_counts.get("valid"):
                errors.append("validation utterance count does not match the configuration")
        raw_training_group_hashes = access.get("training_group_hashes")
        training_group_digest = access.get("training_group_sha256")
        if (
            not isinstance(raw_training_group_hashes, list)
            or not raw_training_group_hashes
            or len(set(raw_training_group_hashes)) != len(raw_training_group_hashes)
            or any(
                not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in raw_training_group_hashes
            )
            or len(raw_training_group_hashes) != access.get("train_video_groups")
            or not isinstance(training_group_digest, str)
            or training_group_digest
            != hashlib.sha256(
                "\n".join(sorted(raw_training_group_hashes)).encode("utf-8")
            ).hexdigest()
        ):
            errors.append("training group identity record is invalid")
        else:
            training_group_hashes = set(raw_training_group_hashes)
        raw_group_hashes = access.get("validation_group_hashes")
        group_digest = access.get("validation_group_sha256")
        if (
            not isinstance(raw_group_hashes, list)
            or not raw_group_hashes
            or len(set(raw_group_hashes)) != len(raw_group_hashes)
            or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in raw_group_hashes)
            or len(raw_group_hashes) != access.get("validation_video_groups")
            or not isinstance(group_digest, str)
            or group_digest
            != hashlib.sha256("\n".join(sorted(raw_group_hashes)).encode("utf-8")).hexdigest()
        ):
            errors.append("validation group identity record is invalid")
        else:
            validation_group_hashes = set(raw_group_hashes)
        if training_group_hashes.intersection(validation_group_hashes):
            errors.append("training and validation video-group identities overlap")
    partitions = record.get("group_partitions")
    if not isinstance(partitions, Mapping) or partitions.get("pairwise_disjoint") is not True:
        errors.append("training group partitions are not verified as disjoint")
    else:
        partition_names = ("task_fit", "router_fit", "calibration")
        partition_sets: list[set[str]] = []
        total_utterances = 0
        total_groups = 0
        valid_partitions = set(partitions) == {*partition_names, "pairwise_disjoint"}
        for name in partition_names:
            entry = partitions.get(name)
            if not isinstance(entry, Mapping):
                valid_partitions = False
                continue
            hashes = entry.get("group_hashes")
            digest = entry.get("group_digest")
            utterances = entry.get("utterances")
            video_groups = entry.get("video_groups")
            positive_rate = entry.get("positive_label_rate")
            if (
                not isinstance(hashes, list)
                or not hashes
                or len(set(hashes)) != len(hashes)
                or any(
                    not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                    for value in hashes
                )
                or not valid_count(video_groups)
                or int(video_groups) != len(hashes)
                or not valid_count(utterances)
                or int(utterances) <= 0
                or not valid_rate(positive_rate)
                or not isinstance(digest, str)
                or digest
                != hashlib.sha256("\n".join(sorted(hashes)).encode("utf-8")).hexdigest()
            ):
                valid_partitions = False
                continue
            partition_sets.append(set(hashes))
            total_utterances += int(utterances)
            total_groups += int(video_groups)
        if (
            not valid_partitions
            or len(partition_sets) != len(partition_names)
            or any(
                partition_sets[left].intersection(partition_sets[right])
                for left in range(len(partition_sets))
                for right in range(left + 1, len(partition_sets))
            )
            or set().union(*partition_sets) != training_group_hashes
            or total_utterances != access.get("train_utterances")
            or total_groups != access.get("train_video_groups")
        ):
            errors.append("training group partition identities or counts are invalid")
    ladder_record = record.get("router_ladder")
    selected_router_family: str | None = None
    online_training_record: Mapping[str, Any] | None = None
    if not isinstance(ladder_record, Mapping):
        errors.append("router-ladder provenance is missing")
    else:
        try:
            expected_ladder_keys = {
                "protocol",
                "inner_cv",
                "families",
                "selection",
                "online_query_policy",
                "library_versions",
                "training_attestation_sha256",
                "official_validation_prediction_sha256",
                "attestation_sha256",
            }
            if set(ladder_record) != expected_ladder_keys:
                raise ValueError("unexpected router-ladder fields")
            training_payload = copy.deepcopy(dict(ladder_record))
            training_digest = training_payload.pop("training_attestation_sha256")
            training_payload.pop("official_validation_prediction_sha256")
            training_payload.pop("attestation_sha256")
            full_payload = copy.deepcopy(dict(ladder_record))
            full_digest = full_payload.pop("attestation_sha256")
            if (
                not isinstance(training_digest, str)
                or training_digest != _json_sha256(training_payload)
                or not isinstance(full_digest, str)
                or full_digest != _json_sha256(full_payload)
            ):
                raise ValueError("router-ladder attestation mismatch")
            versions = ladder_record["library_versions"]
            if (
                not isinstance(versions, Mapping)
                or set(versions) != {"numpy", "scikit_learn"}
                or versions["scikit_learn"] != sklearn.__version__
                or (
                    expected_config is not None
                    and versions["scikit_learn"]
                    != expected_config["router_ladder"]["scikit_learn_version"]
                )
            ):
                raise ValueError("router-ladder versions mismatch")
            inner_cv = ladder_record["inner_cv"]
            folds = inner_cv["folds"]
            expected_fold_count = (
                int(expected_config["router_ladder"]["inner_group_folds"])
                if expected_config is not None
                else int(inner_cv["fold_count"])
            )
            if (
                inner_cv.get("grouping_unit") != "video"
                or inner_cv.get("selection_metric") != "mean_held_out_error_reduction"
                or (
                    expected_config is not None
                    and not np.isclose(
                        float(inner_cv.get("budget", np.nan)),
                        float(expected_config["router_ladder"]["selection_budget"]),
                    )
                )
                or int(inner_cv.get("fold_count", -1)) != expected_fold_count
                or not isinstance(folds, list)
                or len(folds) != expected_fold_count
            ):
                raise ValueError("router-ladder inner CV mismatch")
            heldout_hashes: list[str] = []
            for fold_index, fold in enumerate(folds):
                train_hashes = fold["train_group_hashes"]
                local_heldout = fold["heldout_group_hashes"]
                if (
                    fold.get("fold") != fold_index
                    or fold.get("group_disjoint") is not True
                    or not isinstance(train_hashes, list)
                    or not isinstance(local_heldout, list)
                    or set(train_hashes).intersection(local_heldout)
                    or fold.get("train_group_digest")
                    != hashlib.sha256("\n".join(sorted(train_hashes)).encode("utf-8")).hexdigest()
                    or fold.get("heldout_group_digest")
                    != hashlib.sha256("\n".join(sorted(local_heldout)).encode("utf-8")).hexdigest()
                ):
                    raise ValueError("router-ladder fold mismatch")
                heldout_hashes.extend(local_heldout)
            router_partition = partitions.get("router_fit") if isinstance(partitions, Mapping) else None
            if (
                not isinstance(router_partition, Mapping)
                or sorted(heldout_hashes) != sorted(router_partition.get("group_hashes", []))
            ):
                raise ValueError("router-ladder fold coverage mismatch")
            families = ladder_record["families"]
            if not isinstance(families, Mapping) or set(families) != set(ROUTER_FAMILIES):
                raise ValueError("router-ladder family set mismatch")
            expected_grids = None
            if expected_config is not None:
                expected_grids = {
                    "linear_value": [{"ridge": float(expected_config["ridge"])}],
                    "hist_gbt_value": expected_config["router_ladder"]["hist_gbt_grid"],
                    "shallow_mlp_value": expected_config["router_ladder"]["shallow_mlp_grid"],
                }
            for family in ROUTER_FAMILIES:
                family_record = families[family]
                candidates = family_record["cv_candidates"]
                grid = family_record["grid"]
                if (
                    not isinstance(grid, list)
                    or not isinstance(candidates, list)
                    or len(grid) != len(candidates)
                    or (expected_grids is not None and grid != expected_grids[family])
                    or family_record.get("selected_parameters") not in grid
                    or not isinstance(family_record.get("fit_state"), Mapping)
                    or re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(family_record.get("calibration_prediction_sha256", "")),
                    )
                    is None
                ):
                    raise ValueError("router-ladder family record mismatch")
                for candidate, parameters in zip(candidates, grid):
                    fold_scores = candidate["folds"]
                    scores = [float(item["error_reduction"]) for item in fold_scores]
                    if (
                        candidate.get("parameters") != parameters
                        or len(fold_scores) != expected_fold_count
                        or [item.get("fold") for item in fold_scores]
                        != list(range(expected_fold_count))
                        or any(
                            re.fullmatch(r"[0-9a-f]{64}", str(item.get("prediction_sha256", "")))
                            is None
                            or not isinstance(item.get("fit_state"), Mapping)
                            for item in fold_scores
                        )
                        or not np.isclose(
                            float(candidate["mean_held_out_error_reduction"]),
                            float(np.mean(scores)),
                        )
                    ):
                        raise ValueError("router-ladder CV record mismatch")
            selection = ladder_record["selection"]
            selection_scores = selection["candidate_error_reduction"]
            tie_order = tuple(selection["family_tie_order"])
            selected_router_family = str(selection["selected_family"])
            if (
                selection.get("split") != "calibration"
                or selection.get("metric") != "error_reduction"
                or set(selection_scores) != set(ROUTER_FAMILIES)
                or tie_order != ROUTER_FAMILIES
                or selected_router_family != _select_router_family(selection_scores, tie_order)
                or (
                    expected_config is not None
                    and not np.isclose(
                        float(selection["budget"]),
                        float(expected_config["router_ladder"]["selection_budget"]),
                    )
                )
            ):
                raise ValueError("router-ladder family selection mismatch")
            online_training_record = ladder_record["online_query_policy"]
            if (
                not isinstance(online_training_record, Mapping)
                or set(online_training_record)
                != {
                    "protocol",
                    "fit_split",
                    "priority",
                    "candidate_choice",
                    "selected_family",
                    "calibration_priority_sha256",
                    "by_target_budget",
                }
                or online_training_record.get("protocol")
                != "calibration_fixed_value_threshold_per_target_budget"
                or online_training_record.get("fit_split") != "calibration"
                or online_training_record.get("priority")
                != "maximum_selected_router_predicted_value"
                or online_training_record.get("candidate_choice")
                != "argmax_selected_router_predicted_value"
                or online_training_record.get("selected_family")
                != selected_router_family
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(online_training_record.get("calibration_priority_sha256", "")),
                )
                is None
                or not isinstance(
                    online_training_record.get("by_target_budget"), Mapping
                )
            ):
                raise ValueError("online query threshold provenance mismatch")
            validation_hashes = ladder_record["official_validation_prediction_sha256"]
            if (
                not isinstance(validation_hashes, Mapping)
                or set(validation_hashes) != set(ROUTER_FAMILIES)
                or any(
                    re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
                    for value in validation_hashes.values()
                )
            ):
                raise ValueError("router-ladder validation predictions mismatch")
        except (KeyError, TypeError, ValueError):
            errors.append("router-ladder provenance is invalid")
    evaluation = record.get("evaluation")
    if not isinstance(evaluation, Mapping) or evaluation.get("split") != "official_validation":
        errors.append("evaluation is not restricted to official validation")
        return errors
    try:
        decision_count = int(evaluation["decision_instances"])
        policies = evaluation["policies"]
    except (KeyError, TypeError, ValueError):
        errors.append("evaluation summary is incomplete")
        return errors
    if decision_count <= 0 or not isinstance(policies, Mapping) or "oracle" not in policies:
        errors.append("evaluation policies are incomplete")
        return errors
    budget_keys = set(policies["oracle"])
    if expected_config is not None:
        expected_budget_keys = {f"{float(value):.6f}" for value in expected_config.get("budgets", ())}
        if budget_keys != expected_budget_keys:
            errors.append("result budget set does not match the configuration")
    online_evaluation_record = evaluation.get("online_query_policy")
    calibration_partition = (
        partitions.get("calibration") if isinstance(partitions, Mapping) else None
    )
    calibration_decisions = (
        3 * int(calibration_partition["utterances"])
        if isinstance(calibration_partition, Mapping)
        and isinstance(calibration_partition.get("utterances"), int)
        else None
    )
    try:
        if online_training_record is None or calibration_decisions is None:
            raise ValueError("online training record is missing")
        training_by_budget = online_training_record["by_target_budget"]
        if set(training_by_budget) != budget_keys:
            raise ValueError("online training budget set mismatch")
        if (
            not isinstance(online_evaluation_record, Mapping)
            or set(online_evaluation_record)
            != {
                "protocol",
                "fit_split",
                "selected_family",
                "priority",
                "candidate_choice",
                "candidate_tie_convention",
                "evaluation_batch_access",
                "priority_sha256",
                "choice_sha256",
                "by_target_budget",
                "attestation_sha256",
            }
            or online_evaluation_record.get("protocol")
            != "fixed_calibration_threshold_per_example"
            or online_evaluation_record.get("fit_split") != "calibration"
            or online_evaluation_record.get("selected_family") != selected_router_family
            or online_evaluation_record.get("priority")
            != "maximum_selected_router_predicted_value"
            or online_evaluation_record.get("candidate_choice")
            != "argmax_selected_router_predicted_value"
            or online_evaluation_record.get("candidate_tie_convention")
            != "first_candidate_in_locked_modality_order"
            or online_evaluation_record.get("evaluation_batch_access")
            != "independent_per_example"
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(online_evaluation_record.get("priority_sha256", "")),
            )
            is None
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(online_evaluation_record.get("choice_sha256", "")),
            )
            is None
        ):
            raise ValueError("online evaluation record mismatch")
        evaluation_attestation_payload = copy.deepcopy(
            dict(online_evaluation_record)
        )
        evaluation_attestation = evaluation_attestation_payload.pop(
            "attestation_sha256"
        )
        if (
            not isinstance(evaluation_attestation, str)
            or evaluation_attestation
            != _json_sha256(evaluation_attestation_payload)
        ):
            raise ValueError("online evaluation attestation mismatch")
        evaluation_by_budget = online_evaluation_record["by_target_budget"]
        if (
            not isinstance(evaluation_by_budget, Mapping)
            or set(evaluation_by_budget) != budget_keys
        ):
            raise ValueError("online evaluation budget set mismatch")
        for key in budget_keys:
            target = float(key)
            fitted = training_by_budget[key]
            observed = evaluation_by_budget[key]
            if not isinstance(fitted, Mapping) or not isinstance(observed, Mapping):
                raise TypeError(f"online query threshold record for {key} is invalid")
            expected_target_count = int(np.rint(target * calibration_decisions))
            tie_convention = (
                "priority_greater_than_or_equal_to_threshold"
                if fitted.get("query_on_equal") is True
                else "priority_strictly_greater_than_threshold"
            )
            if (
                set(fitted)
                != {
                    "target_query_rate",
                    "target_query_count",
                    "calibration_decisions",
                    "calibration_query_count",
                    "calibration_realized_query_rate",
                    "threshold",
                    "query_on_equal",
                    "tie_convention",
                }
                or not valid_rate(fitted["target_query_rate"])
                or not np.isclose(float(fitted["target_query_rate"]), target)
                or not valid_count(
                    fitted["target_query_count"], maximum=calibration_decisions
                )
                or int(fitted["target_query_count"]) != expected_target_count
                or not valid_count(fitted["calibration_decisions"])
                or int(fitted["calibration_decisions"]) != calibration_decisions
                or not valid_count(
                    fitted["calibration_query_count"], maximum=calibration_decisions
                )
                or not valid_rate(fitted["calibration_realized_query_rate"])
                or not np.isclose(
                    float(fitted["calibration_realized_query_rate"]),
                    int(fitted["calibration_query_count"]) / calibration_decisions,
                )
                or not finite_number(fitted["threshold"])
                or not isinstance(fitted["query_on_equal"], bool)
                or fitted["tie_convention"] != tie_convention
                or set(observed)
                != {
                    "target_query_rate",
                    "threshold",
                    "query_on_equal",
                    "tie_convention",
                    "evaluation_decisions",
                    "evaluation_query_count",
                    "evaluation_realized_query_rate",
                }
                or not valid_rate(observed["target_query_rate"])
                or not np.isclose(float(observed["target_query_rate"]), target)
                or observed["threshold"] != fitted["threshold"]
                or observed["query_on_equal"] != fitted["query_on_equal"]
                or observed["tie_convention"] != fitted["tie_convention"]
                or not valid_count(observed["evaluation_decisions"])
                or int(observed["evaluation_decisions"]) != decision_count
                or not valid_count(
                    observed["evaluation_query_count"], maximum=decision_count
                )
                or not valid_rate(observed["evaluation_realized_query_rate"])
                or not np.isclose(
                    float(observed["evaluation_realized_query_rate"]),
                    int(observed["evaluation_query_count"]) / decision_count,
                )
            ):
                raise ValueError(f"online query threshold record for {key} is invalid")
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        errors.append("online query policy provenance is invalid")
    validation_utterances = access.get("validation_utterances") if isinstance(access, Mapping) else None
    if isinstance(validation_utterances, int) and decision_count != 3 * validation_utterances:
        errors.append("decision count does not match three starting contexts per utterance")
    candidate_count = evaluation.get("candidate_instances")
    expected_multiplier = 2 if mode == "singleton" else 1
    if not isinstance(candidate_count, int) or candidate_count != decision_count * expected_multiplier:
        errors.append("candidate count does not match the study mode")
    required_policies = set(ROUTER_POLICIES)
    if set(policies) != required_policies:
        errors.append("policy set does not match the locked protocol")
        return errors
    expected_repetitions = (
        int(expected_config["bootstrap_repetitions"])
        if expected_config is not None
        else None
    )
    bootstrap = record.get("bootstrap")
    bootstrap_draws: dict[str, dict[str, np.ndarray]] = {}
    base_policies = {
        *BASE_BOOTSTRAP_POLICIES,
        "selected_router",
        "online_selected_router",
    }
    if (
        not isinstance(bootstrap, Mapping)
        or bootstrap.get("method") != "video_cluster_resample_with_policy_reselection"
        or bootstrap.get("online_policy_method")
        != "video_cluster_resample_with_fixed_calibration_threshold"
        or not isinstance(bootstrap.get("repetitions"), int)
        or int(bootstrap["repetitions"]) <= 0
        or (
            expected_repetitions is not None
            and bootstrap.get("repetitions") != expected_repetitions
        )
        or (
            expected_config is not None
            and bootstrap.get("group_draw_seed")
            != int(expected_config.get("bootstrap_seed", 20_270_917))
        )
    ):
        errors.append("bootstrap protocol record is invalid")
    else:
        raw_draws = bootstrap.get("policy_error_reduction_draws")
        if not isinstance(raw_draws, Mapping) or set(raw_draws) != base_policies:
            errors.append("policy bootstrap draws are incomplete")
        else:
            for policy, by_budget in raw_draws.items():
                if not isinstance(by_budget, Mapping) or set(by_budget) != budget_keys:
                    errors.append(f"{policy} bootstrap budget set is invalid")
                    continue
                bootstrap_draws[policy] = {}
                for key, values in by_budget.items():
                    try:
                        array = np.asarray(values, dtype=np.float64)
                    except (TypeError, ValueError):
                        errors.append(f"{policy}/{key} bootstrap draws are invalid")
                        continue
                    if (
                        array.shape != (int(bootstrap["repetitions"]),)
                        or not np.isfinite(array).all()
                        or np.any(np.abs(array) > 1.0 + 1e-12)
                    ):
                        errors.append(f"{policy}/{key} bootstrap draws are invalid")
                        continue
                    bootstrap_draws[policy][key] = array
    canonical_group_counts: dict[str, int] | None = None
    for policy, by_budget in policies.items():
        if set(by_budget) != budget_keys:
            errors.append(f"{policy} budget set differs from the oracle budget set")
            continue
        for key, metrics in by_budget.items():
            try:
                budget = float(key)
                query_count = int(metrics["query_count"])
                query_rate = float(metrics["query_rate"])
                base_error = float(metrics["base_error"])
                final_error = float(metrics["final_error"])
                reduction = float(metrics["error_reduction"])
                interval = [float(value) for value in metrics["error_reduction_cluster_ci95"]]
            except (KeyError, TypeError, ValueError):
                errors.append(f"{policy}/{key} has invalid metrics")
                continue
            expected_count = int(np.rint(budget * decision_count))
            if policy == "online_selected_router":
                online_budget = (
                    online_evaluation_record.get("by_target_budget", {}).get(key, {})
                    if isinstance(online_evaluation_record, Mapping)
                    else {}
                )
                if (
                    not valid_count(query_count, maximum=decision_count)
                    or not np.isclose(query_rate, query_count / decision_count)
                    or online_budget.get("evaluation_query_count") != query_count
                    or not np.isclose(
                        float(
                            online_budget.get(
                                "evaluation_realized_query_rate", np.nan
                            )
                        ),
                        query_rate,
                    )
                ):
                    errors.append(
                        f"{policy}/{key} does not match its fixed-threshold record"
                    )
            elif query_count != expected_count or not np.isclose(
                query_rate, expected_count / decision_count
            ):
                errors.append(f"{policy}/{key} does not use the exact query budget")
            if (
                not valid_rate(base_error)
                or not valid_rate(final_error)
                or not np.isclose(base_error - final_error, reduction)
                or not finite_number(reduction)
                or abs(reduction) > 1.0 + 1e-12
                or len(interval) != 2
                or not np.isfinite(interval).all()
                or any(abs(value) > 1.0 + 1e-12 for value in interval)
            ):
                errors.append(f"{policy}/{key} has nonfinite evaluation values")
            expected_bootstrap_method = (
                "video_cluster_resample_with_fixed_calibration_threshold"
                if policy == "online_selected_router"
                else "video_cluster_resample_with_exact_budget_reselection"
            )
            if metrics.get("bootstrap_method") != expected_bootstrap_method:
                errors.append(f"{policy}/{key} has an invalid bootstrap method")
            if policy != "selected_baseline" and key in bootstrap_draws.get(policy, {}):
                expected_interval = np.quantile(bootstrap_draws[policy][key], (0.025, 0.975))
                if not np.allclose(interval, expected_interval):
                    errors.append(f"{policy}/{key} interval does not match saved bootstrap draws")
            for rate_name in (
                "base_macro_f1",
                "final_macro_f1",
                "harmful_query_rate",
                "useful_query_precision",
                "useful_query_recall",
                "candidate_value_tie_rate",
                "candidate_ranking_accuracy",
                "queried_candidate_selection_accuracy",
                "risk_coverage_auc",
            ):
                if not valid_rate(metrics.get(rate_name), nullable=rate_name in {
                    "harmful_query_rate",
                    "useful_query_precision",
                    "useful_query_recall",
                    "candidate_value_tie_rate",
                    "candidate_ranking_accuracy",
                    "queried_candidate_selection_accuracy",
                }):
                    errors.append(f"{policy}/{key} has invalid {rate_name}")
            video_macro = metrics.get("error_reduction_video_macro")
            if not finite_number(video_macro) or abs(float(video_macro)) > 1.0 + 1e-12:
                errors.append(f"{policy}/{key} has invalid video-macro error reduction")
            policy_group_statistics = metrics.get("video_group_statistics")
            if not isinstance(policy_group_statistics, list) or not policy_group_statistics:
                errors.append(f"{policy}/{key} video-group statistics are missing")
            else:
                try:
                    group_hashes = [entry["group_sha256"] for entry in policy_group_statistics]
                    group_counts = [entry["decision_count"] for entry in policy_group_statistics]
                    group_sums = [entry["error_reduction_sum"] for entry in policy_group_statistics]
                    local_group_counts = {
                        str(group_hash): int(count)
                        for group_hash, count in zip(group_hashes, group_counts)
                    }
                    if (
                        len(set(group_hashes)) != len(group_hashes)
                        or set(group_hashes) != validation_group_hashes
                        or any(not valid_count(value, maximum=decision_count) or int(value) <= 0 for value in group_counts)
                        or any(not finite_number(value) for value in group_sums)
                        or any(abs(float(value)) > int(count) + 1e-12 for value, count in zip(group_sums, group_counts))
                        or sum(int(value) for value in group_counts) != decision_count
                        or not np.isclose(
                            sum(float(value) for value in group_sums) / decision_count,
                            reduction,
                        )
                        or not np.isclose(
                            np.mean(
                                [float(value) / int(count) for value, count in zip(group_sums, group_counts)]
                            ),
                            float(video_macro),
                        )
                    ):
                        errors.append(f"{policy}/{key} video-group statistics are inconsistent")
                    elif canonical_group_counts is None:
                        canonical_group_counts = local_group_counts
                    elif local_group_counts != canonical_group_counts:
                        errors.append(f"{policy}/{key} video-group counts differ from the canonical evaluation groups")
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    errors.append(f"{policy}/{key} video-group statistics are invalid")
            evaluable = metrics.get("candidate_ranking_evaluable")
            if not isinstance(evaluable, int) or evaluable < 0 or evaluable > decision_count:
                errors.append(f"{policy}/{key} has invalid source-selection count")
            if mode == "pair" and (
                evaluable != 0
                or metrics.get("candidate_ranking_accuracy") is not None
                or metrics.get("queried_candidate_selection_accuracy") is not None
                or metrics.get("candidate_value_tie_rate") is not None
            ):
                errors.append(f"{policy}/{key} reports source selection for pair mode")
            oracle_reduction = float(policies["oracle"][key]["error_reduction"])
            if policy == "online_selected_router":
                oracle_reference_rate = metrics.get("oracle_reference_query_rate")
                oracle_reference_reduction = metrics.get(
                    "oracle_reference_error_reduction"
                )
                if (
                    not valid_rate(oracle_reference_rate)
                    or not np.isclose(float(oracle_reference_rate), query_rate)
                    or not finite_number(oracle_reference_reduction)
                    or abs(float(oracle_reference_reduction)) > 1.0 + 1e-12
                ):
                    errors.append(f"{policy}/{key} has an invalid matched-rate oracle")
                    oracle_reduction = float("nan")
                else:
                    oracle_reduction = float(oracle_reference_reduction)
            if np.isfinite(oracle_reduction) and reduction > oracle_reduction + 1e-12:
                errors.append(f"{policy}/{key} exceeds the oracle")
            regret = metrics.get("oracle_regret")
            if not finite_number(regret) or not np.isclose(float(regret), oracle_reduction - reduction) or float(regret) < -1e-12:
                errors.append(f"{policy}/{key} has invalid oracle regret")
    baseline_selection = evaluation.get("baseline_selection")
    paired_contrasts = evaluation.get("paired_contrasts")
    if not isinstance(baseline_selection, Mapping) or set(baseline_selection) != budget_keys:
        errors.append("matched-baseline selection is incomplete")
    else:
        for budget_index, key in enumerate(sorted(baseline_selection, key=float)):
            selection = baseline_selection[key]
            if selection.get("policy") not in {"source_prior", "confidence"}:
                errors.append(f"matched baseline for {key} is invalid")
                continue
            candidate_errors = selection.get("candidate_errors")
            policy_seeds = selection.get("policy_seeds")
            if not isinstance(candidate_errors, Mapping) or set(candidate_errors) != {"source_prior", "confidence"}:
                errors.append(f"matched baseline candidate errors for {key} are invalid")
                continue
            chosen_from_errors = min(
                candidate_errors,
                key=lambda name: (float(candidate_errors[name]), name != "source_prior", name),
            )
            if selection.get("policy") != chosen_from_errors or not np.isclose(
                float(selection.get("selection_error", np.nan)),
                float(candidate_errors[chosen_from_errors]),
            ):
                errors.append(f"matched baseline selection for {key} is inconsistent")
            expected_policy_seeds = {
                "confidence": int(record.get("seed", 0)) * 1009
                + POLICY_SEED_INDEX["confidence"] * 97,
                "source_prior": int(record.get("seed", 0)) * 1009
                + POLICY_SEED_INDEX["source_prior"] * 97,
            }
            if policy_seeds != expected_policy_seeds:
                errors.append(f"matched baseline policy seeds for {key} are inconsistent")
            chosen = str(selection["policy"])
            selected_metrics = policies["selected_baseline"][key]
            if selected_metrics.get("selected_policy") != chosen:
                errors.append(f"selected baseline for {key} is inconsistent")
            selected_payload = {
                name: value for name, value in selected_metrics.items() if name != "selected_policy"
            }
            if selected_payload != policies[chosen][key]:
                errors.append(f"selected baseline for {key} does not exactly reuse {chosen}")
            if key in bootstrap_draws.get(chosen, {}):
                expected_interval = np.quantile(bootstrap_draws[chosen][key], (0.025, 0.975))
                if not np.allclose(selected_metrics.get("error_reduction_cluster_ci95"), expected_interval):
                    errors.append(f"selected baseline interval for {key} does not match saved bootstrap draws")
    if not isinstance(paired_contrasts, Mapping) or set(paired_contrasts) != budget_keys:
        errors.append("paired policy contrasts are incomplete")
    else:
        for key, contrast in paired_contrasts.items():
            try:
                difference = float(contrast["error_reduction_difference"])
                interval = np.asarray(contrast["cluster_ci95"], dtype=np.float64)
            except (KeyError, TypeError, ValueError):
                errors.append(f"paired contrast for {key} is invalid")
                continue
            expected_difference = float(policies["selected_router"][key]["error_reduction"]) - float(
                policies["selected_baseline"][key]["error_reduction"]
            )
            if not np.isclose(difference, expected_difference) or interval.shape != (2,) or not np.isfinite(interval).all():
                errors.append(f"paired contrast for {key} is inconsistent")
            if contrast.get("bootstrap_method") != "video_cluster_resample_with_exact_budget_reselection":
                errors.append(f"paired contrast for {key} has an invalid bootstrap method")
            chosen_baseline = (
                baseline_selection.get(key, {}).get("policy")
                if isinstance(baseline_selection, Mapping)
                else None
            )
            if contrast.get("left_policy") != "selected_router" or contrast.get("right_policy") != chosen_baseline:
                errors.append(f"paired contrast policy labels for {key} are invalid")
            video_macro_difference = contrast.get("video_macro_difference")
            expected_video_macro_difference = float(
                policies["selected_router"][key]["error_reduction_video_macro"]
            ) - float(policies["selected_baseline"][key]["error_reduction_video_macro"])
            if (
                not finite_number(video_macro_difference)
                or abs(float(video_macro_difference)) > 1.0 + 1e-12
                or not np.isclose(float(video_macro_difference), expected_video_macro_difference)
            ):
                errors.append(f"paired video-macro contrast for {key} is invalid")
            try:
                contrast_draws = np.asarray(contrast["bootstrap_draws"], dtype=np.float64)
                expected_draws = bootstrap_draws["selected_router"][key] - bootstrap_draws[str(chosen_baseline)][key]
                if (
                    contrast_draws.shape != expected_draws.shape
                    or not np.isfinite(contrast_draws).all()
                    or not np.allclose(contrast_draws, expected_draws)
                    or not np.allclose(interval, np.quantile(contrast_draws, (0.025, 0.975)))
                ):
                    errors.append(f"paired bootstrap draws for {key} are inconsistent")
            except (KeyError, TypeError, ValueError):
                errors.append(f"paired bootstrap draws for {key} are invalid")
            group_statistics = contrast.get("video_group_statistics")
            if not isinstance(group_statistics, list) or not group_statistics:
                errors.append(f"paired group statistics for {key} are missing")
            else:
                try:
                    group_hashes = [entry["group_sha256"] for entry in group_statistics]
                    group_count = sum(int(entry["decision_count"]) for entry in group_statistics)
                    group_sum = sum(float(entry["error_reduction_difference_sum"]) for entry in group_statistics)
                    group_macro = np.mean(
                        [
                            float(entry["error_reduction_difference_sum"])
                            / int(entry["decision_count"])
                            for entry in group_statistics
                        ]
                    )
                    local_group_counts = {
                        str(entry["group_sha256"]): int(entry["decision_count"])
                        for entry in group_statistics
                    }
                    if (
                        len(set(group_hashes)) != len(group_hashes)
                        or set(group_hashes) != validation_group_hashes
                        or group_count != decision_count
                        or not np.isclose(group_sum / group_count, difference)
                        or not np.isclose(group_macro, float(video_macro_difference))
                        or (
                            canonical_group_counts is not None
                            and local_group_counts != canonical_group_counts
                        )
                    ):
                        errors.append(f"paired group statistics for {key} are inconsistent")
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    errors.append(f"paired group statistics for {key} are invalid")
            selective = contrast.get("coverage_0.90")
            if not isinstance(selective, Mapping):
                errors.append(f"paired selective contrast for {key} is missing")
            else:
                left_coverage = selective.get("left_coverage")
                right_coverage = selective.get("right_coverage")
                left_risk = selective.get("left_selective_risk")
                right_risk = selective.get("right_selective_risk")
                selective_difference = selective.get("selective_risk_difference")
                selective_interval = selective.get("cluster_ci95")
                try:
                    selective_draws = np.asarray(selective.get("bootstrap_draws"), dtype=np.float64)
                except (TypeError, ValueError):
                    selective_draws = np.asarray([], dtype=np.float64)
                expected_accepted = int(np.rint(0.9 * decision_count))
                selective_values_valid = (
                    selective.get("coverage_definition") == "exact_evaluation_count"
                    and np.isclose(float(selective.get("target_coverage", np.nan)), 0.9)
                    and selective.get("accepted_count") == expected_accepted
                    and valid_rate(left_coverage)
                    and valid_rate(right_coverage)
                    and np.isclose(float(left_coverage), expected_accepted / decision_count)
                    and np.isclose(float(right_coverage), expected_accepted / decision_count)
                    and valid_rate(left_risk, nullable=True)
                    and valid_rate(right_risk, nullable=True)
                    and finite_number(selective_difference)
                    and abs(float(selective_difference)) <= 1.0 + 1e-12
                    and isinstance(selective_interval, list)
                    and len(selective_interval) == 2
                    and all(finite_number(value) and abs(float(value)) <= 1.0 + 1e-12 for value in selective_interval)
                    and selective.get("bootstrap_method")
                    == "video_cluster_resample_with_exact_query_budget_and_exact_coverage_reselection"
                    and selective_draws.shape == (int(selective.get("bootstrap_repetitions", -1)),)
                    and np.isfinite(selective_draws).all()
                    and np.all(np.abs(selective_draws) <= 1.0 + 1e-12)
                    and np.allclose(
                        np.asarray(selective_interval, dtype=np.float64),
                        np.quantile(selective_draws, (0.025, 0.975)),
                    )
                )
                if selective_values_valid and not np.isclose(
                    float(selective_difference), float(right_risk) - float(left_risk)
                ):
                    selective_values_valid = False
                if not selective_values_valid:
                    errors.append(f"paired selective contrast for {key} is invalid")
                selective_groups = selective.get("video_group_statistics")
                if not isinstance(selective_groups, list) or not selective_groups:
                    errors.append(f"paired selective group statistics for {key} are missing")
                elif isinstance(group_statistics, list):
                    ordinary_group_counts = {
                        entry.get("group_sha256"): entry.get("decision_count")
                        for entry in group_statistics
                        if isinstance(entry, Mapping)
                    }
                    selective_hashes = [
                        entry.get("group_sha256")
                        for entry in selective_groups
                        if isinstance(entry, Mapping)
                    ]
                    if (
                        len(selective_hashes) != len(selective_groups)
                        or len(set(selective_hashes)) != len(selective_hashes)
                        or set(selective_hashes) != set(ordinary_group_counts)
                    ):
                        errors.append(f"paired selective group statistics for {key} are inconsistent")
                    else:
                        try:
                            left_count = 0
                            right_count = 0
                            left_error_sum = 0.0
                            right_error_sum = 0.0
                            for entry in selective_groups:
                                group_hash = entry["group_sha256"]
                                group_decisions = int(ordinary_group_counts[group_hash])
                                local_left_count = entry["left_accepted_count"]
                                local_right_count = entry["right_accepted_count"]
                                local_left_error = entry["left_accepted_error_sum"]
                                local_right_error = entry["right_accepted_error_sum"]
                                if (
                                    not valid_count(local_left_count, maximum=group_decisions)
                                    or not valid_count(local_right_count, maximum=group_decisions)
                                    or not finite_number(local_left_error)
                                    or not finite_number(local_right_error)
                                    or float(local_left_error) < -1e-12
                                    or float(local_left_error) > int(local_left_count) + 1e-12
                                    or float(local_right_error) < -1e-12
                                    or float(local_right_error) > int(local_right_count) + 1e-12
                                ):
                                    raise ValueError("invalid selective group sufficient statistic")
                                left_count += int(local_left_count)
                                right_count += int(local_right_count)
                                left_error_sum += float(local_left_error)
                                right_error_sum += float(local_right_error)
                            reconstructed = (
                                left_count > 0
                                and right_count > 0
                                and np.isclose(left_count / decision_count, float(left_coverage))
                                and np.isclose(right_count / decision_count, float(right_coverage))
                                and np.isclose(left_error_sum / left_count, float(left_risk))
                                and np.isclose(right_error_sum / right_count, float(right_risk))
                                and np.isclose(
                                    right_error_sum / right_count - left_error_sum / left_count,
                                    float(selective_difference),
                                )
                            )
                            if not reconstructed:
                                errors.append(f"paired selective group statistics for {key} do not reconstruct the contrast")
                        except (KeyError, TypeError, ValueError, ZeroDivisionError):
                            errors.append(f"paired selective group statistics for {key} are invalid")
                if expected_config is not None and selective.get("bootstrap_repetitions") != int(
                    expected_config["bootstrap_repetitions"]
                ):
                    errors.append(f"paired selective bootstrap count for {key} differs from configuration")
                if expected_config is not None:
                    budget_index = sorted(budget_keys, key=float).index(key)
                    expected_selective_seed = int(expected_config.get("bootstrap_seed", 20_270_917)) + 10_000 + budget_index
                    if selective.get("bootstrap_seed") != expected_selective_seed:
                        errors.append(f"paired selective bootstrap seed for {key} differs from configuration")
    curves = evaluation.get("selective_curves")
    selection_decision_count = None
    if isinstance(partitions, Mapping):
        selection_partition = partitions.get("calibration")
        if isinstance(selection_partition, Mapping) and isinstance(selection_partition.get("utterances"), int):
            selection_decision_count = 3 * int(selection_partition["utterances"])
    expected_coverage_keys = (
        {
            f"{float(value):.6f}"
            for value in expected_config.get("coverage_targets", (1.0, 0.9, 0.8))
        }
        if expected_config is not None
        else None
    )
    if not isinstance(curves, Mapping) or set(curves) != required_policies:
        errors.append("selective curves are incomplete")
    else:
        for policy, by_budget in curves.items():
            if not isinstance(by_budget, Mapping) or set(by_budget) != budget_keys:
                errors.append(f"{policy} selective-curve budget set is invalid")
                continue
            for key, by_coverage in by_budget.items():
                if not isinstance(by_coverage, Mapping):
                    errors.append(f"{policy}/{key} selective curve is missing")
                    continue
                if expected_coverage_keys is not None and set(by_coverage) != expected_coverage_keys:
                    errors.append(f"{policy}/{key} coverage targets differ from the configuration")
                for coverage_key, values in by_coverage.items():
                    if not isinstance(values, Mapping):
                        errors.append(f"{policy}/{key}/{coverage_key} selective result is invalid")
                        continue
                    accepted_count = values.get("accepted_count")
                    accepted_error_sum = values.get("accepted_error_sum")
                    selection_accepted_count = values.get("selection_accepted_count")
                    selection_accepted_error_sum = values.get("selection_accepted_error_sum")
                    if (
                        not finite_number(values.get("threshold"))
                        or not valid_rate(values.get("coverage"))
                        or not valid_rate(values.get("abstention_rate"))
                        or not np.isclose(float(values.get("coverage", -1)) + float(values.get("abstention_rate", -1)), 1.0)
                        or not valid_rate(values.get("selective_risk"), nullable=True)
                        or not valid_rate(values.get("selection_error_at_threshold"), nullable=True)
                        or not valid_count(accepted_count, maximum=decision_count)
                        or not finite_number(accepted_error_sum)
                        or float(accepted_error_sum) < -1e-12
                        or float(accepted_error_sum) > int(accepted_count) + 1e-12
                        or not np.isclose(float(values.get("coverage", -1)), int(accepted_count) / decision_count)
                        or (
                            int(accepted_count) == 0
                            and values.get("selective_risk") is not None
                        )
                        or (
                            int(accepted_count) > 0
                            and not np.isclose(
                                float(values.get("selective_risk", np.nan)),
                                float(accepted_error_sum) / int(accepted_count),
                            )
                        )
                        or selection_decision_count is None
                        or not valid_count(selection_accepted_count, maximum=selection_decision_count)
                        or not finite_number(selection_accepted_error_sum)
                        or float(selection_accepted_error_sum) < -1e-12
                        or float(selection_accepted_error_sum) > int(selection_accepted_count) + 1e-12
                        or (
                            int(selection_accepted_count) == 0
                            and values.get("selection_error_at_threshold") is not None
                        )
                        or (
                            int(selection_accepted_count) > 0
                            and not np.isclose(
                                float(values.get("selection_error_at_threshold", np.nan)),
                                float(selection_accepted_error_sum) / int(selection_accepted_count),
                            )
                        )
                    ):
                        errors.append(f"{policy}/{key}/{coverage_key} selective values are invalid")
                    if coverage_key == "1.000000" and (
                        not np.isclose(float(values.get("coverage", -1)), 1.0)
                        or not np.isclose(
                            float(values.get("selective_risk", np.nan)),
                            float(policies[policy][key]["final_error"]),
                        )
                    ):
                        errors.append(f"{policy}/{key} full-coverage risk is inconsistent")
        if isinstance(baseline_selection, Mapping):
            for key, selection in baseline_selection.items():
                chosen = selection.get("policy") if isinstance(selection, Mapping) else None
                if (
                    chosen in {"source_prior", "confidence"}
                    and key in curves.get("selected_baseline", {})
                    and curves["selected_baseline"][key] != curves.get(chosen, {}).get(key)
                ):
                    errors.append(f"selected-baseline selective curves for {key} do not exactly reuse {chosen}")
    contexts = evaluation.get("contexts")
    expected_contexts = (
        {"audio", "video", "text"}
        if mode == "singleton"
        else {"audio+video", "audio+text", "video+text"}
    )
    if not isinstance(contexts, Mapping) or set(contexts) != expected_contexts:
        errors.append("starting-modality context results are incomplete")
    else:
        context_decision_total = 0
        for name, context in contexts.items():
            try:
                context_decisions = context["decisions"]
                if not valid_count(context_decisions, maximum=decision_count) or int(context_decisions) <= 0:
                    raise ValueError("invalid context decision count")
                context_decision_total += int(context_decisions)
            except (KeyError, TypeError, ValueError):
                errors.append(f"{name} decision count is invalid")
                context_decisions = 0
            observed_modalities = set(name.split("+"))
            expected_candidates = [modality for modality in MODALITIES if modality not in observed_modalities]
            candidates = context.get("candidates") if isinstance(context, Mapping) else None
            diagnostic_labels = context.get("by_label") if isinstance(context, Mapping) else None
            try:
                if (
                    context.get("candidate_modalities") != expected_candidates
                    or not valid_rate(context.get("base_error"))
                    or not isinstance(candidates, Mapping)
                    or set(candidates) != set(expected_candidates)
                ):
                    raise ValueError("invalid context diagnostics")
                for candidate in candidates.values():
                    if (
                        not isinstance(candidate, Mapping)
                        or not finite_number(candidate.get("mean_realized_value"))
                        or abs(float(candidate["mean_realized_value"])) > 1.0 + 1e-12
                        or not valid_rate(candidate.get("positive_value_rate"))
                        or not valid_rate(candidate.get("negative_value_rate"))
                        or float(candidate["positive_value_rate"])
                        + float(candidate["negative_value_rate"]) > 1.0 + 1e-12
                        or not finite_number(candidate.get("value_mean_squared_error"))
                        or float(candidate["value_mean_squared_error"]) < 0.0
                    ):
                        raise ValueError("invalid candidate diagnostics")
                if not isinstance(diagnostic_labels, Mapping) or set(diagnostic_labels) != {"0", "1"}:
                    raise ValueError("invalid context diagnostic labels")
                diagnostic_count = 0
                diagnostic_base_errors = 0.0
                for label_summary in diagnostic_labels.values():
                    label_count = label_summary.get("decisions")
                    if (
                        not valid_count(label_count, maximum=int(context_decisions))
                        or int(label_count) <= 0
                        or not valid_rate(label_summary.get("base_error"))
                        or not finite_number(label_summary.get("mean_oracle_value"))
                        or abs(float(label_summary["mean_oracle_value"])) > 1.0 + 1e-12
                    ):
                        raise ValueError("invalid context diagnostic label metric")
                    diagnostic_count += int(label_count)
                    diagnostic_base_errors += int(label_count) * float(label_summary["base_error"])
                if (
                    diagnostic_count != int(context_decisions)
                    or not np.isclose(
                        diagnostic_base_errors / int(context_decisions), float(context["base_error"])
                    )
                ):
                    raise ValueError("context diagnostics do not reconstruct the context")
            except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError):
                errors.append(f"{name} diagnostics are invalid")
            outcomes = context.get("policy_outcomes") if isinstance(context, Mapping) else None
            if not isinstance(outcomes, Mapping) or set(outcomes) != required_policies:
                errors.append(f"{name} policy outcomes are incomplete")
                continue
            for policy, by_budget in outcomes.items():
                if not isinstance(by_budget, Mapping) or set(by_budget) != budget_keys:
                    errors.append(f"{name}/{policy} budget outcomes are incomplete")
                    continue
                for key, summary in by_budget.items():
                    if not isinstance(summary, Mapping):
                        errors.append(f"{name}/{policy}/{key} outcome is invalid")
                        continue
                    try:
                        decisions = summary["decisions"]
                        query_count = summary["query_count"]
                        query_rate = summary["query_rate"]
                        base_error = summary["base_error"]
                        final_error = summary["final_error"]
                        reduction = summary["error_reduction"]
                        if (
                            not valid_count(decisions, maximum=decision_count)
                            or int(decisions) != int(context_decisions)
                            or not valid_count(query_count, maximum=int(decisions))
                            or not valid_rate(query_rate)
                            or not np.isclose(float(query_rate), int(query_count) / int(decisions))
                            or not valid_rate(base_error)
                            or not valid_rate(final_error)
                            or not finite_number(reduction)
                            or abs(float(reduction)) > 1.0 + 1e-12
                            or not np.isclose(float(base_error) - float(final_error), float(reduction))
                            or not valid_rate(summary.get("base_macro_f1"))
                            or not valid_rate(summary.get("final_macro_f1"))
                            or not valid_rate(summary.get("harmful_query_rate"), nullable=True)
                            or not valid_rate(summary.get("useful_query_precision"), nullable=True)
                            or not valid_rate(summary.get("candidate_ranking_accuracy"), nullable=True)
                            or not valid_rate(
                                summary.get("queried_candidate_selection_accuracy"), nullable=True
                            )
                            or not valid_count(summary.get("candidate_ranking_evaluable"), maximum=int(decisions))
                            or not np.isclose(float(base_error), float(context.get("base_error", np.nan)))
                        ):
                            raise ValueError("invalid context outcome metric")
                        if mode == "pair" and (
                            summary.get("candidate_ranking_evaluable") != 0
                            or summary.get("candidate_ranking_accuracy") is not None
                            or summary.get("queried_candidate_selection_accuracy") is not None
                        ):
                            raise ValueError("pair context reports source selection")
                        target_counts = summary.get("query_target_counts")
                        if (
                            not isinstance(target_counts, Mapping)
                            or set(target_counts) != set(MODALITIES)
                            or any(not valid_count(value, maximum=int(query_count)) for value in target_counts.values())
                            or sum(int(value) for value in target_counts.values()) != int(query_count)
                        ):
                            raise ValueError("invalid context query-target counts")
                        target_outcomes = summary.get("query_target_outcomes")
                        if not isinstance(target_outcomes, Mapping) or set(target_outcomes) != set(MODALITIES):
                            raise ValueError("invalid context query-target outcomes")
                        target_value_sum = 0.0
                        target_harmful_sum = 0.0
                        for modality, target_summary in target_outcomes.items():
                            if not isinstance(target_summary, Mapping) or set(target_summary) != {
                                "query_count",
                                "realized_value_sum",
                                "harmful_query_count",
                                "mean_realized_value",
                                "harmful_query_rate",
                            }:
                                raise ValueError("invalid context target outcome fields")
                            target_count = target_summary.get("query_count")
                            value_sum = target_summary.get("realized_value_sum")
                            harmful_count = target_summary.get("harmful_query_count")
                            mean_value = target_summary.get("mean_realized_value")
                            harmful_rate = target_summary.get("harmful_query_rate")
                            if (
                                not valid_count(target_count, maximum=int(query_count))
                                or int(target_count) != int(target_counts[modality])
                                or not finite_number(value_sum)
                                or abs(float(value_sum)) > int(target_count) + 1e-12
                                or not valid_count(harmful_count, maximum=int(target_count))
                            ):
                                raise ValueError("invalid context target outcome count")
                            if int(target_count) == 0:
                                if not np.isclose(float(value_sum), 0.0) or int(harmful_count) != 0:
                                    raise ValueError("empty target outcome has nonzero sufficient statistics")
                                if any(
                                    value is not None
                                    for value in (mean_value, harmful_rate)
                                ):
                                    raise ValueError("empty target outcome is not null")
                                continue
                            if (
                                not finite_number(mean_value)
                                or abs(float(mean_value)) > 1.0 + 1e-12
                                or not valid_rate(harmful_rate)
                                or not np.isclose(
                                    float(mean_value), float(value_sum) / int(target_count)
                                )
                                or not np.isclose(
                                    float(harmful_rate), int(harmful_count) / int(target_count)
                                )
                            ):
                                raise ValueError("invalid context target outcome metric")
                            target_value_sum += float(value_sum)
                            target_harmful_sum += int(harmful_count)
                        if (
                            not np.isclose(target_value_sum / int(decisions), float(reduction))
                            or (
                                int(query_count) > 0
                                and not np.isclose(
                                    target_harmful_sum / int(query_count),
                                    float(summary["harmful_query_rate"]),
                                )
                            )
                        ):
                            raise ValueError("context target outcomes do not reconstruct outcome")
                        by_label = summary.get("by_label")
                        if not isinstance(by_label, Mapping) or set(by_label) != {"0", "1"}:
                            raise ValueError("invalid context label breakdown")
                        label_decisions = 0
                        label_queries = 0.0
                        label_base_errors = 0.0
                        label_final_errors = 0.0
                        label_reductions = 0.0
                        for label_summary in by_label.values():
                            label_count = label_summary.get("decisions")
                            label_query_rate = label_summary.get("query_rate")
                            label_base_error = label_summary.get("base_error")
                            label_final_error = label_summary.get("final_error")
                            label_reduction = label_summary.get("error_reduction")
                            if (
                                not valid_count(label_count, maximum=int(decisions))
                                or int(label_count) <= 0
                                or not valid_rate(label_query_rate)
                                or not valid_rate(label_base_error)
                                or not valid_rate(label_final_error)
                                or not finite_number(label_reduction)
                                or abs(float(label_reduction)) > 1.0 + 1e-12
                                or not np.isclose(
                                    float(label_base_error) - float(label_final_error),
                                    float(label_reduction),
                                )
                            ):
                                raise ValueError("invalid context label metric")
                            label_decisions += int(label_count)
                            label_queries += int(label_count) * float(label_query_rate)
                            label_base_errors += int(label_count) * float(label_base_error)
                            label_final_errors += int(label_count) * float(label_final_error)
                            label_reductions += int(label_count) * float(label_reduction)
                        if (
                            label_decisions != int(decisions)
                            or not np.isclose(label_queries, int(query_count))
                            or not np.isclose(label_base_errors / int(decisions), float(base_error))
                            or not np.isclose(label_final_errors / int(decisions), float(final_error))
                            or not np.isclose(label_reductions / int(decisions), float(reduction))
                        ):
                            raise ValueError("context label breakdown does not reconstruct outcome")
                    except (KeyError, TypeError, ValueError, ZeroDivisionError):
                        errors.append(f"{name}/{policy}/{key} outcome is invalid")
        if context_decision_total != decision_count:
            errors.append("starting-modality decision counts do not sum to the pooled count")
        for policy in required_policies:
            for key in budget_keys:
                try:
                    summaries = [contexts[name]["policy_outcomes"][policy][key] for name in contexts]
                    total = sum(int(summary["decisions"]) for summary in summaries)
                    query_total = sum(int(summary["query_count"]) for summary in summaries)
                    pooled = policies[policy][key]
                    weighted_base = sum(
                        float(summary["base_error"]) * int(summary["decisions"])
                        for summary in summaries
                    ) / total
                    weighted_final = sum(
                        float(summary["final_error"]) * int(summary["decisions"])
                        for summary in summaries
                    ) / total
                    weighted_reduction = sum(
                        float(summary["error_reduction"]) * int(summary["decisions"])
                        for summary in summaries
                    ) / total
                    target_total = sum(
                        sum(int(value) for value in summary["query_target_counts"].values())
                        for summary in summaries
                    )
                    harmful_total = sum(
                        sum(
                            int(target["harmful_query_count"])
                            for target in summary["query_target_outcomes"].values()
                        )
                        for summary in summaries
                    )
                    pooled_harmful = pooled["harmful_query_rate"]
                    harmful_consistent = (
                        pooled_harmful is None
                        if query_total == 0
                        else finite_number(pooled_harmful)
                        and np.isclose(harmful_total / query_total, float(pooled_harmful))
                    )
                    label_counts = {
                        label: sum(int(summary["by_label"][label]["decisions"]) for summary in summaries)
                        for label in ("0", "1")
                    }
                    base_label_errors = {
                        label: sum(
                            int(summary["by_label"][label]["decisions"])
                            * float(summary["by_label"][label]["base_error"])
                            for summary in summaries
                        )
                        for label in ("0", "1")
                    }
                    final_label_errors = {
                        label: sum(
                            int(summary["by_label"][label]["decisions"])
                            * float(summary["by_label"][label]["final_error"])
                            for summary in summaries
                        )
                        for label in ("0", "1")
                    }
                    reconstructed_base_macro_f1 = macro_f1_from_label_errors(
                        label_counts, base_label_errors
                    )
                    reconstructed_final_macro_f1 = macro_f1_from_label_errors(
                        label_counts, final_label_errors
                    )
                    if (
                        total != decision_count
                        or query_total != int(pooled["query_count"])
                        or target_total != query_total
                        or not harmful_consistent
                        or not np.isclose(weighted_base, float(pooled["base_error"]))
                        or not np.isclose(weighted_final, float(pooled["final_error"]))
                        or not np.isclose(weighted_reduction, float(pooled["error_reduction"]))
                        or not np.isclose(
                            reconstructed_base_macro_f1, float(pooled["base_macro_f1"])
                        )
                        or not np.isclose(
                            reconstructed_final_macro_f1, float(pooled["final_macro_f1"])
                        )
                    ):
                        errors.append(f"{policy}/{key} context outcomes do not reconstruct the pooled result")
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    errors.append(f"{policy}/{key} context outcomes are invalid")
        if isinstance(baseline_selection, Mapping):
            for key, selection in baseline_selection.items():
                chosen = selection.get("policy") if isinstance(selection, Mapping) else None
                if chosen not in {"source_prior", "confidence"}:
                    continue
                for name, context in contexts.items():
                    outcomes = context.get("policy_outcomes") if isinstance(context, Mapping) else None
                    if (
                        isinstance(outcomes, Mapping)
                        and key in outcomes.get("selected_baseline", {})
                        and outcomes["selected_baseline"][key] != outcomes.get(chosen, {}).get(key)
                    ):
                        errors.append(
                            f"{name} selected-baseline outcome for {key} does not exactly reuse {chosen}"
                        )
    if selected_router_family in ROUTER_FAMILIES:
        for key in budget_keys:
            if policies.get("selected_router", {}).get(key) != policies.get(selected_router_family, {}).get(key):
                errors.append(f"selected router for {key} does not exactly reuse {selected_router_family}")
            if isinstance(curves, Mapping) and curves.get("selected_router", {}).get(key) != curves.get(
                selected_router_family, {}
            ).get(key):
                errors.append(f"selected-router selective curve for {key} is not an exact alias")
            if key in bootstrap_draws.get("selected_router", {}) and key in bootstrap_draws.get(
                selected_router_family, {}
            ) and not np.array_equal(
                bootstrap_draws["selected_router"][key],
                bootstrap_draws[selected_router_family][key],
            ):
                errors.append(f"selected-router bootstrap draws for {key} are not an exact alias")
            if isinstance(contexts, Mapping):
                for name, context in contexts.items():
                    outcomes = context.get("policy_outcomes") if isinstance(context, Mapping) else None
                    if (
                        isinstance(outcomes, Mapping)
                        and outcomes.get("selected_router", {}).get(key)
                        != outcomes.get(selected_router_family, {}).get(key)
                    ):
                        errors.append(f"{name} selected-router outcome for {key} is not an exact alias")
    regression = evaluation.get("value_regression")
    calibration = evaluation.get("benefit_calibration")
    confidence_calibration = evaluation.get("task_confidence_calibration")
    if not isinstance(regression, Mapping) or any(
        not finite_number(regression.get(name)) or float(regression[name]) < 0
        for name in ("mean_squared_error", "mean_absolute_error")
    ):
        errors.append("value-regression metrics are invalid")
    if not isinstance(calibration, Mapping) or not valid_rate(calibration.get("brier")) or not finite_number(
        calibration.get("log_loss")
    ) or float(calibration.get("log_loss", -1)) < 0:
        errors.append("benefit-calibration metrics are invalid")
    if (
        not isinstance(confidence_calibration, Mapping)
        or confidence_calibration.get("scope") != "per_observation_mask_on_calibration_groups"
        or not valid_rate(confidence_calibration.get("base_brier"))
        or not valid_rate(confidence_calibration.get("candidate_brier"))
    ):
        errors.append("task-confidence calibration metrics are invalid")
    headroom = record.get("headroom_gate")
    if not isinstance(headroom, Mapping):
        errors.append("headroom gate is missing")
    else:
        try:
            headroom_key = f"{float(headroom['budget']):.6f}"
            observed = float(headroom["observed_oracle_error_reduction"])
            minimum = float(headroom["minimum_oracle_error_reduction"])
            interval = [float(value) for value in headroom["cluster_ci95"]]
            headroom_draws = np.asarray(headroom["bootstrap_draws"], dtype=np.float64)
            raw_passed = headroom["passed"]
            passed = bool(raw_passed)
            oracle_metrics = policies["oracle"][headroom_key]
            recomputed = observed >= minimum and interval[0] > 0.0
            saved_oracle_draws = bootstrap_draws["oracle"][headroom_key]
            if (
                not np.isclose(observed, float(oracle_metrics["error_reduction"]))
                or interval != [float(value) for value in oracle_metrics["error_reduction_cluster_ci95"]]
                or headroom_draws.shape != saved_oracle_draws.shape
                or not np.allclose(headroom_draws, saved_oracle_draws)
                or not np.allclose(interval, np.quantile(headroom_draws, (0.025, 0.975)))
                or not isinstance(raw_passed, (bool, np.bool_))
                or passed != recomputed
            ):
                errors.append("headroom gate is inconsistent")
            if expected_config is not None and (
                not np.isclose(float(headroom["budget"]), float(expected_config["headroom_budget"]))
                or not np.isclose(minimum, float(expected_config["minimum_oracle_error_reduction"]))
            ):
                errors.append("headroom gate differs from the configuration")
        except (KeyError, TypeError, ValueError):
            errors.append("headroom gate is invalid")
    task_head = record.get("task_head_gate")
    if not isinstance(task_head, Mapping):
        errors.append("task-head adequacy gate is missing")
    else:
        try:
            accuracy = float(task_head["full_avt_validation_accuracy"])
            accuracy_by_mask = task_head["validation_accuracy_by_mask"]
            reference_accuracy = float(task_head["reference_validation_accuracy"])
            maximum_gap = float(task_head["maximum_accuracy_gap"])
            minimum_mask_accuracy = float(task_head["minimum_mask_validation_accuracy"])
            reference_digest = task_head["reference_record_sha256"]
            passed = task_head["passed"]
            expected_mask_names = {_mask_name(mask) for mask in _observed_masks()}
            mask_record_valid = (
                isinstance(accuracy_by_mask, Mapping)
                and set(accuracy_by_mask) == expected_mask_names
                and all(valid_rate(value) for value in accuracy_by_mask.values())
            )
            observed_minimum_mask_accuracy = (
                min(float(value) for value in accuracy_by_mask.values())
                if mask_record_valid
                else float("nan")
            )
            mask_context_names = (
                ("audio", "video", "text")
                if mode == "singleton"
                else ("audio+video", "audio+text", "video+text")
            )
            context_mask_consistent = isinstance(contexts, Mapping) and all(
                name in contexts
                and np.isclose(
                    float(accuracy_by_mask.get(name, np.nan)),
                    1.0 - float(contexts[name]["base_error"]),
                )
                for name in mask_context_names
            )
            full_mask_consistent = True
            if mode == "pair":
                full_mask_consistent = (
                    "1.000000" in budget_keys
                    and np.isclose(
                        accuracy,
                        1.0 - float(policies["linear_value"]["1.000000"]["final_error"]),
                    )
                )
            recomputed = (
                accuracy >= reference_accuracy - maximum_gap
                and observed_minimum_mask_accuracy >= minimum_mask_accuracy
            )
            if (
                task_head.get("predictor_scope")
                != "separate_frozen_ridge_per_observation_mask_fit_on_task_groups"
                or not mask_record_valid
                or not context_mask_consistent
                or not full_mask_consistent
                or (
                    mask_record_valid
                    and not np.isclose(float(accuracy_by_mask["audio+video+text"]), accuracy)
                )
                or not valid_rate(accuracy)
                or not valid_rate(reference_accuracy)
                or not finite_number(maximum_gap)
                or maximum_gap < 0.0
                or maximum_gap > 1.0
                or not valid_rate(minimum_mask_accuracy)
                or not isinstance(reference_digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", reference_digest) is None
                or not isinstance(passed, (bool, np.bool_))
                or bool(passed) != recomputed
            ):
                errors.append("task-head adequacy gate is inconsistent")
            if expected_config is not None and (
                task_head.get("reference_name") != expected_config["task_head_reference_name"]
                or not np.isclose(
                    reference_accuracy, float(expected_config["reference_full_avt_validation_accuracy"])
                )
                or reference_digest != expected_config["task_head_reference_sha256"]
                or not np.isclose(maximum_gap, float(expected_config["maximum_task_head_accuracy_gap"]))
                or not np.isclose(
                    minimum_mask_accuracy,
                    float(expected_config["minimum_mask_validation_accuracy"]),
                )
            ):
                errors.append("task-head adequacy gate differs from the configuration")
        except (KeyError, TypeError, ValueError):
            errors.append("task-head adequacy gate is invalid")
    saved_validation = record.get("validation")
    if not isinstance(saved_validation, Mapping) or saved_validation.get("ok") is not True:
        errors.append("saved validation status is not successful")
    return errors
