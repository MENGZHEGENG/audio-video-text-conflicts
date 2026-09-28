"""Configuration runner used by the command-line entrypoint."""

from __future__ import annotations

from datetime import datetime, timezone
import platform
from typing import Any

import numpy as np

from .core import (
    HEURISTIC_METHODS,
    LEARNED_METHODS,
    Dataset,
    evaluate_actions,
    fit_and_predict_acquisition_torch,
    fit_and_predict_torch,
    generate_dataset,
    policy_actions,
    torch_status,
)
from .mosei import load_mosei_bundle, load_mosei_cache
from .multibench import load_descriptor_cache, load_mmsa_bundle
from .temporal import (
    TEMPORAL_LEARNED_METHODS,
    TEMPORAL_MECHANISMS,
    TEMPORAL_PROTOCOL,
    TemporalDataset,
    fit_and_predict_temporal_torch,
    generate_temporal_dataset,
    temporal_summary,
)


_DESCRIPTOR_DATASETS = {"mosi", "sims", "sims_v2", "meld", "ur_funny", "descriptor"}


def _validate_config(config: dict[str, Any]) -> None:
    required = {"seed"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing config fields: {missing}")
    dataset = str(config.get("dataset", "synthetic")).lower()
    if dataset == "synthetic":
        required = {"train_size", "eval_size", "train_mechanisms", "seen_mechanisms", "unseen_mechanisms"}
    elif dataset == "synthetic_temporal":
        required = {
            "train_size",
            "eval_size",
            "train_mechanisms",
            "seen_mechanisms",
            "unseen_mechanisms",
            "seq_len",
        }
    elif dataset == "mosei":
        path_fields = {"modality_paths", "labels_path", "splits_path"}
        if "cache_path" not in config and not path_fields.issubset(config):
            raise ValueError("mosei config must provide cache_path or all CSD path fields")
        return
    elif dataset in _DESCRIPTOR_DATASETS:
        if "cache_path" not in config and "feature_path" not in config:
            raise ValueError(f"{dataset} config must provide cache_path or feature_path")
        return
    else:
        supported = ", ".join(sorted({"synthetic", "synthetic_temporal", "mosei", *_DESCRIPTOR_DATASETS}))
        raise ValueError(f"dataset must be one of: {supported}")
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"missing {dataset} config fields: {missing}")
    if dataset == "synthetic_temporal" and int(config["seq_len"]) < 2:
        raise ValueError("seq_len must be at least 2 for synthetic_temporal")


def _heuristic_record(name: str, split: Dataset, config: dict[str, Any]) -> dict[str, Any]:
    output = policy_actions(name, split.x)
    evaluated = evaluate_actions(
        output,
        split,
        query_cost=float(config.get("query_cost", 0.10)),
        abstain_cost=float(config.get("abstain_cost", 0.20)),
    )
    return {"status": "verified", **evaluated}


def _subset(dataset: Dataset, indices: np.ndarray) -> Dataset:
    return Dataset(
        x=dataset.x[indices],
        y=dataset.y[indices],
        ambiguous=dataset.ambiguous[indices],
        mechanism=dataset.mechanism[indices],
    )


def _acquisition_calibration(
    train: Dataset,
    *,
    seed: int,
    dataset_name: str,
    config: dict[str, Any],
    strength: float = 1.0,
    noise: float = 0.22,
) -> tuple[Dataset, Dataset, str]:
    """Return disjoint fit/calibration data without using evaluation splits."""

    if dataset_name == "synthetic":
        calibration = generate_dataset(
            seed + 300_003,
            int(config.get("calibration_size", config.get("eval_size", len(train.y)))),
            config["train_mechanisms"],
            strength=strength,
            noise=noise,
        )
        return train, calibration, "independent_train_mechanisms"

    if dataset_name == "synthetic_temporal":
        calibration_raw = generate_temporal_dataset(
            seed + 300_003,
            int(config.get("calibration_size", config.get("eval_size", len(train.y)))),
            config["train_mechanisms"],
            seq_len=int(config["seq_len"]),
            strength=strength,
            noise=noise,
            positive_rate=float(config.get("positive_rate", 0.5)),
            mechanism_probs=config.get("train_mechanism_probs"),
        )
        return train, temporal_summary(calibration_raw), "independent_train_mechanisms"

    fraction = float(config.get("acquisition_calibration_fraction", 0.20))
    if not 0.0 < fraction < 1.0:
        raise ValueError("acquisition_calibration_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed + 300_003)
    indices = rng.permutation(len(train.y))
    calibration_size = max(1, int(round(len(indices) * fraction)))
    calibration_indices = indices[:calibration_size]
    fit_indices = indices[calibration_size:]
    if len(fit_indices) == 0:
        raise ValueError("acquisition calibration split leaves no fit rows")
    return _subset(train, fit_indices), _subset(train, calibration_indices), "held_out_train_rows"


def _split_record(
    split: Dataset,
    config: dict[str, Any],
    heuristic_records: dict[str, dict[str, Any]],
    learned_records: dict[str, dict[str, Any]],
    *,
    temporal_split: TemporalDataset | None = None,
    temporal_records: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    methods: dict[str, dict[str, Any]] = {}
    methods.update({name: heuristic_records[name] for name in HEURISTIC_METHODS})
    for name, record in learned_records.items():
        output = record.get("outputs", {}).get(id(split))
        if output is None:
            methods[name] = {k: v for k, v in record.items() if k != "outputs"}
            continue
        evaluated = evaluate_actions(
            output,
            split,
            query_cost=float(config.get("query_cost", 0.10)),
            abstain_cost=float(config.get("abstain_cost", 0.20)),
        )
        methods[name] = {k: v for k, v in record.items() if k != "outputs"}
        methods[name].update(evaluated)
    for name, record in (temporal_records or {}).items():
        output = record.get("outputs", {}).get(id(temporal_split)) if temporal_split is not None else None
        if output is None:
            methods[name] = {k: v for k, v in record.items() if k != "outputs"}
            continue
        evaluated = evaluate_actions(
            output,
            split,
            query_cost=float(config.get("query_cost", 0.10)),
            abstain_cost=float(config.get("abstain_cost", 0.20)),
        )
        methods[name] = {k: v for k, v in record.items() if k != "outputs"}
        methods[name].update(evaluated)
    return {"n": int(len(split.y)), "mechanisms": sorted(set(split.mechanism.tolist())), "methods": methods}


def run_from_config(config: dict[str, Any], *, device_preference: str = "auto") -> dict[str, Any]:
    """Run all configured methods and return a JSON-serializable result record."""

    _validate_config(config)
    seed = int(config["seed"])
    dataset_name = str(config.get("dataset", "synthetic")).lower()
    dataset_metadata: dict[str, Any] = {"dataset": dataset_name}
    sample_ids: dict[str, tuple[str, ...]] = {}
    temporal_splits: dict[str, TemporalDataset] | None = None
    temporal_train: TemporalDataset | None = None
    if dataset_name == "mosei":
        if config.get("cache_path"):
            bundle = load_mosei_cache(config["cache_path"])
        else:
            bundle = load_mosei_bundle(
                config["modality_paths"],
                config["labels_path"],
                config["splits_path"],
                sentiment_mode=str(config.get("sentiment_mode", "nonnegative")),
                max_samples_per_split=config.get("max_samples_per_split"),
            )
        train = bundle.splits["train"]
        # The official validation and test folds become seen and unseen
        # evaluation respectively; training never contributes evaluation rows.
        splits = {"seen": bundle.splits["valid"], "unseen": bundle.splits["test"]}
        sample_ids = {"seen": bundle.sample_ids["valid"], "unseen": bundle.sample_ids["test"]}
        dataset_metadata.update(bundle.metadata)
        strength = 1.0
        noise = 0.22
    elif dataset_name in _DESCRIPTOR_DATASETS:
        if config.get("cache_path"):
            bundle = load_descriptor_cache(config["cache_path"])
        else:
            bundle = load_mmsa_bundle(
                config["feature_path"],
                dataset_name=str(config.get("benchmark_name", dataset_name)),
                sentiment_mode=str(config.get("sentiment_mode", "nonnegative")),
            )
        train = bundle.splits["train"]
        splits = {"seen": bundle.splits["valid"], "unseen": bundle.splits["test"]}
        sample_ids = {"seen": bundle.sample_ids["valid"], "unseen": bundle.sample_ids["test"]}
        dataset_metadata.update(bundle.metadata)
        strength = 1.0
        noise = 0.22
    elif dataset_name == "synthetic_temporal":
        strength = float(config.get("strength", 1.0))
        noise = float(config.get("noise", 0.22))
        temporal_kwargs = {
            "seq_len": int(config["seq_len"]),
            "strength": strength,
            "noise": noise,
            "positive_rate": float(config.get("positive_rate", 0.5)),
            "mechanism_probs": config.get("train_mechanism_probs"),
        }
        temporal_train = generate_temporal_dataset(
            seed, int(config["train_size"]), config["train_mechanisms"], **temporal_kwargs
        )
        seen_raw = generate_temporal_dataset(
            seed + 100_003,
            int(config["eval_size"]),
            config["seen_mechanisms"],
            **{**temporal_kwargs, "mechanism_probs": config.get("seen_mechanism_probs")},
        )
        unseen_raw = generate_temporal_dataset(
            seed + 200_003,
            int(config["eval_size"]),
            config["unseen_mechanisms"],
            **{**temporal_kwargs, "mechanism_probs": config.get("unseen_mechanism_probs")},
        )
        temporal_splits = {"seen": seen_raw, "unseen": unseen_raw}
        train = temporal_summary(temporal_train)
        splits = {name: temporal_summary(raw) for name, raw in temporal_splits.items()}
        dataset_metadata["temporal"] = {
            "protocol": TEMPORAL_PROTOCOL,
            "seq_len": int(config["seq_len"]),
            "channels": 3,
            "mechanisms": list(TEMPORAL_MECHANISMS),
            "summary": "mean_over_time_for_scalar_controls",
            "marker": "zero_mean_two_harmonic",
            "swap": "opposite_marker_with_nonzero_cyclic_lag",
            "segment_invert": "contiguous_opposite_marker_interval",
            "occlusion": "contiguous_noise_interval",
            "drift": "zero_area_channel_local_linear_drift",
            "sequence_methods": list(TEMPORAL_LEARNED_METHODS),
        }
    else:
        strength = float(config.get("strength", 1.0))
        noise = float(config.get("noise", 0.22))
        train = generate_dataset(seed, int(config["train_size"]), config["train_mechanisms"], strength=strength, noise=noise)
        seen = generate_dataset(seed + 100_003, int(config["eval_size"]), config["seen_mechanisms"], strength=strength, noise=noise)
        unseen = generate_dataset(seed + 200_003, int(config["eval_size"]), config["unseen_mechanisms"], strength=strength, noise=noise)
        splits = {"seen": seen, "unseen": unseen}

    acquisition_train, acquisition_calibration, acquisition_calibration_source = _acquisition_calibration(
        train,
        seed=seed,
        dataset_name=dataset_name,
        config=config,
        strength=strength,
        noise=noise,
    )
    dataset_metadata["acquisition_calibration"] = {
        "source": acquisition_calibration_source,
        "fit_size": int(len(acquisition_train.y)),
        "calibration_size": int(len(acquisition_calibration.y)),
    }

    heuristics_by_split: dict[str, dict[str, dict[str, Any]]] = {}
    for split_name, split in splits.items():
        heuristics_by_split[split_name] = {name: _heuristic_record(name, split, config) for name in HEURISTIC_METHODS}

    torch_info = torch_status()
    requested = device_preference
    if requested == "auto":
        device = "cuda" if torch_info.get("cuda_available") else "cpu"
    elif requested == "cuda" and not torch_info.get("cuda_available"):
        device = "cpu"
    else:
        device = requested

    learned_by_method: dict[str, dict[str, Any]] = {}
    for name in LEARNED_METHODS:
        if not torch_info.get("available"):
            learned_by_method[name] = {"status": "skipped", "reason": torch_info.get("reason", "PyTorch unavailable")}
            continue
        try:
            if name == "learned_acquisition":
                outputs, fit_record = fit_and_predict_acquisition_torch(
                    acquisition_train,
                    acquisition_calibration,
                    splits,
                    seed=seed,
                    epochs=int(config.get("acquisition_epochs", config.get("epochs", 25))),
                    batch_size=int(config.get("batch_size", 64)),
                    learning_rate=float(config.get("acquisition_learning_rate", config.get("learning_rate", 0.002))),
                    device=device,
                    query_cost=float(config.get("query_cost", 0.10)),
                    abstain_cost=float(config.get("abstain_cost", 0.20)),
                    initial_threshold=float(config.get("initial_threshold", 0.35)),
                    threshold_grid=config.get("acquisition_thresholds", (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)),
                    torch_threads=int(config.get("torch_threads", 2)),
                )
            else:
                outputs, fit_record = fit_and_predict_torch(
                    name,
                    train,
                    splits,
                    seed=seed,
                    epochs=int(config.get("epochs", 25)),
                    batch_size=int(config.get("batch_size", 64)),
                    learning_rate=float(config.get("learning_rate", 0.002)),
                    device=device,
                    torch_threads=int(config.get("torch_threads", 2)),
                )
            # Object IDs are local-only keys consumed before serialization.
            learned_by_method[name] = {**fit_record, "outputs": {id(splits[k]): outputs[k] for k in outputs}}
        except Exception as exc:  # pragma: no cover - device/library dependent
            learned_by_method[name] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}

    temporal_by_method: dict[str, dict[str, Any]] = {}
    if temporal_splits is not None and temporal_train is not None:
        for name in TEMPORAL_LEARNED_METHODS:
            if not torch_info.get("available"):
                temporal_by_method[name] = {
                    "status": "skipped",
                    "reason": torch_info.get("reason", "PyTorch unavailable"),
                }
                continue
            try:
                outputs, fit_record = fit_and_predict_temporal_torch(
                    name,
                    temporal_train,
                    temporal_splits,
                    seed=seed,
                    max_steps=int(config.get("temporal_max_steps", config.get("max_steps", 200))),
                    batch_size=int(config.get("temporal_batch_size", config.get("batch_size", 64))),
                    learning_rate=float(
                        config.get("temporal_learning_rate", config.get("learning_rate", 0.002))
                    ),
                    width=int(config.get("temporal_width", 96)),
                    depth=int(config.get("temporal_depth", 4)),
                    device=device,
                    torch_threads=int(config.get("torch_threads", 2)),
                    eval_batch_size=int(config.get("temporal_eval_batch_size", 1024)),
                )
                temporal_by_method[name] = {
                    **fit_record,
                    "outputs": {id(temporal_splits[k]): outputs[k] for k in outputs},
                }
            except Exception as exc:  # pragma: no cover - device/library dependent
                temporal_by_method[name] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}

    split_records = {
        split_name: _split_record(
            split,
            config,
            heuristics_by_split[split_name],
            learned_by_method,
            temporal_split=temporal_splits[split_name] if temporal_splits is not None else None,
            temporal_records=temporal_by_method,
        )
        for split_name, split in splits.items()
    }
    method_statuses = {
        split_name: {name: record.get("status") for name, record in record_data["methods"].items()}
        for split_name, record_data in split_records.items()
    }
    base_verified = all(method_statuses[s][name] == "verified" for s in method_statuses for name in HEURISTIC_METHODS)
    no_failed_methods = all(status != "failed" for statuses in method_statuses.values() for status in statuses.values())
    learned_verified = all(
        method_statuses[s][name] == "verified" for s in method_statuses for name in LEARNED_METHODS
    )
    temporal_verified = not temporal_splits or all(
        method_statuses[s][name] == "verified" for s in method_statuses for name in TEMPORAL_LEARNED_METHODS
    )

    return {
        "schema": "conflictbench.run.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "dataset": dataset_metadata,
        "config": config,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "torch": torch_info,
            "requested_device": requested,
            "used_device": device,
        },
        "splits": {
            split_name: {
                **record,
                **({"sample_ids": list(sample_ids[split_name])} if split_name in sample_ids else {}),
            }
            for split_name, record in split_records.items()
        },
        "validation": {
            "ok": bool(base_verified and no_failed_methods and temporal_verified),
            "base_methods_verified": bool(base_verified),
            "learned_methods_verified": bool(learned_verified),
            "temporal_methods_verified": bool(temporal_verified),
            "learned_methods_skipped": sorted(name for name in LEARNED_METHODS if any(method_statuses[s][name] == "skipped" for s in method_statuses)),
            "temporal_methods_skipped": sorted(
                name
                for name in TEMPORAL_LEARNED_METHODS
                if any(method_statuses[s].get(name) == "skipped" for s in method_statuses)
            ),
        },
    }
