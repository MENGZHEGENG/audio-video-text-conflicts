from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from conflictbench.mosei_raw import (
    RAW_CACHE_SCHEMA,
    RawMoseiBundle,
    RawMoseiSplit,
    build_raw_mosei_bundle,
    load_raw_mosei_split,
    save_raw_mosei_cache,
)
from conflictbench.mosei_raw_value_study import (
    RAW_RESULT_SCHEMA,
    RAW_STUDY_SCHEMA,
    implementation_sha256,
    prepare_projected_data,
    run_raw_value_study,
    validate_raw_value_result,
)
from conflictbench.mosei_value_study import ROUTER_FAMILIES, _router_features

MODALITIES = ("audio", "video", "text")
ROOT = Path(__file__).parents[1]
RAW_CACHE_SCRIPT = ROOT / "scripts" / "cache_mosei_raw_descriptors.py"
RAW_RUN_SCRIPT = ROOT / "scripts" / "run_mosei_raw_value_study.py"


def _load_script(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split(prefix: str, group_count: int, *, offset: float = 0.0) -> RawMoseiSplit:
    groups = []
    sample_ids = []
    labels = []
    for group_index in range(group_count):
        for label in (0, 1):
            group = f"{prefix}-{group_index:02d}"
            groups.append(group)
            sample_ids.append(f"{group}[{label}]")
            labels.append(label)
    y = np.asarray(labels, dtype=np.int64)
    sign = 2.0 * y - 1.0
    base = np.arange(len(y), dtype=np.float64) * 0.01 + offset
    descriptors = {
        "audio": np.column_stack((sign + base, base + 0.2)),
        "video": np.column_stack((0.8 * sign + base, base - 0.1, sign * 0.2)),
        "text": np.column_stack((1.2 * sign + base, base + 0.4, sign * 0.3, base - 0.3)),
    }
    return RawMoseiSplit(
        descriptors={name: values.astype(np.float32) for name, values in descriptors.items()},
        y=y,
        sample_ids=tuple(sample_ids),
        groups=np.asarray(groups, dtype="U"),
    )


def _bundle() -> RawMoseiBundle:
    return RawMoseiBundle(
        splits={"train": _split("train", 12), "valid": _split("valid", 6, offset=0.3)},
        metadata={
            "dataset": "CMU-MOSEI",
            "sentiment_mode": "nonnegative",
            "alignment": "positive_interval_overlap_mean",
            "pooling": "finite_frame_mean",
            "supervised_transform": "none",
            "source_provenance": {
                "audio": {
                    "name": "audio.csd",
                    "source_file_sha256": "1" * 64,
                    "permitted_content_sha256": "a" * 64,
                    "content_scope": "official_train_and_validation_pooled_values",
                    "bytes": 10,
                },
                "video": {
                    "name": "video.csd",
                    "source_file_sha256": "2" * 64,
                    "permitted_content_sha256": "b" * 64,
                    "content_scope": "official_train_and_validation_pooled_values",
                    "bytes": 20,
                },
                "text": {
                    "name": "text.csd",
                    "source_file_sha256": "3" * 64,
                    "permitted_content_sha256": "c" * 64,
                    "content_scope": "official_train_and_validation_pooled_values",
                    "bytes": 30,
                },
                "labels": {
                    "name": "labels.csd",
                    "source_file_sha256": "4" * 64,
                    "permitted_content_sha256": "d" * 64,
                    "content_scope": "official_train_and_validation_labels",
                    "bytes": 40,
                },
                "splits": {
                    "name": "splits.json",
                    "source_file_sha256": "5" * 64,
                    "permitted_content_sha256": "e" * 64,
                    "content_scope": "official_train_and_validation_group_ids",
                    "bytes": 50,
                },
            },
            "loaded_split_counts": {"train": 24, "valid": 12},
            "official_fold_counts": {"train": 12, "valid": 6, "test": 2},
            "missing_permitted_fold_ids": {"train": [], "valid": []},
            "feature_dimensions": {"audio": 2, "video": 3, "text": 4},
            "official_test_policy": "excluded_from_cache_and_value_study",
        },
    )


def _config() -> dict:
    return {
        "schema": RAW_STUDY_SCHEMA,
        "dataset": "CMU-MOSEI",
        "input_representation": "utterance-aligned pooled released computational-sequence descriptors",
        "evidence_status": "exploratory_taskfit_reprojection",
        "conflict_definition": "none_unmodified_official_mosei_sentiment",
        "projection": {
            "fit_scope": "official_train_task_fit_groups",
            "standardization": "per_modality_task_fit_zscore",
            "supervised_projection": "task_fit_standardized_mean_difference",
            "minimum_scale": 1e-8,
        },
        "expected_split_counts": {"train": 24, "valid": 12},
        "train_group_fractions": [0.5, 0.25, 0.25],
        "seeds": [11, 23],
        "budgets": [0.0, 0.5, 1.0],
        "coverage_targets": [1.0, 0.9],
        "ridge": 0.01,
        "benefit_ridge": 0.1,
        "confidence_ridge": 0.1,
        "bootstrap_repetitions": 8,
        "bootstrap_seed": 20270917,
        "headroom_budget": 0.5,
        "primary_budget": 0.5,
        "minimum_oracle_error_reduction": 0.0,
        "minimum_primary_error_reduction_gain": 0.0,
        "minimum_video_macro_gain_fraction": 0.0,
        "minimum_nonnegative_start_contexts": 0,
        "maximum_start_context_harm": 1.0,
        "task_head_reference_name": "raw-lane test reference",
        "reference_full_avt_validation_accuracy": 0.5,
        "task_head_reference_sha256": "0" * 64,
        "maximum_task_head_accuracy_gap": 0.5,
        "minimum_mask_validation_accuracy": 0.0,
        "router_ladder": {
            "inner_group_folds": 3,
            "selection_budget": 0.5,
            "selection_metric": "mean_held_out_error_reduction",
            "family_tie_order": list(ROUTER_FAMILIES),
            "scikit_learn_version": "1.9.0",
            "hist_gbt_grid": [
                {
                    "max_leaf_nodes": 7,
                    "learning_rate": 0.05,
                    "max_iter": 5,
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
                    "max_iter": 5,
                    "early_stopping": False,
                }
            ],
        },
    }


def _write_raw_cache(path: Path, bundle: RawMoseiBundle | None = None) -> Path:
    return save_raw_mosei_cache(_bundle() if bundle is None else bundle, path)


def _config_for_cache(path: Path) -> dict:
    config = _config()
    config["expected_cache_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return config


def test_implementation_digest_binds_imported_mosei_loader(monkeypatch):
    baseline = implementation_sha256()
    original_read_bytes = Path.read_bytes
    changed_dependency_reads: list[Path] = []

    def read_changed_dependency(path: Path) -> bytes:
        payload = original_read_bytes(path)
        if path.name == "mosei.py":
            changed_dependency_reads.append(path)
            return payload + b"\n# simulated loader change\n"
        return payload

    monkeypatch.setattr(Path, "read_bytes", read_changed_dependency)

    changed = implementation_sha256()

    assert changed_dependency_reads
    assert changed != baseline


def test_raw_cache_round_trip_has_unsupervised_schema_and_safe_provenance(tmp_path):
    path = _write_raw_cache(tmp_path / "raw.npz")
    train, metadata = load_raw_mosei_split(path, "train")

    assert metadata["cache_schema"] == RAW_CACHE_SCHEMA
    assert metadata["supervised_transform"] == "none"
    assert set(metadata["source_provenance"]) == {"audio", "video", "text", "labels", "splits"}
    assert str(tmp_path) not in json.dumps(metadata)
    assert all(
        set(entry)
        == {
            "name",
            "source_file_sha256",
            "permitted_content_sha256",
            "content_scope",
            "bytes",
        }
        for entry in metadata["source_provenance"].values()
    )
    assert train.sample_ids == _bundle().splits["train"].sample_ids
    for modality in MODALITIES:
        np.testing.assert_allclose(
            train.descriptors[modality], _bundle().splits["train"].descriptors[modality]
        )
    with np.load(path, allow_pickle=False) as archive:
        assert not any(name.startswith("test_") for name in archive.files)


def test_raw_cache_loader_rejects_forged_source_provenance(tmp_path):
    path = _write_raw_cache(tmp_path / "raw.npz")
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(payload["metadata_json"].reshape(()).item()))
    metadata["source_provenance"]["audio"]["permitted_content_sha256"] = "z" * 64
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError, match="source provenance"):
        load_raw_mosei_split(path, "train")


def test_raw_cache_loader_rejects_metadata_count_drift(tmp_path):
    path = _write_raw_cache(tmp_path / "raw.npz")
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(payload["metadata_json"].reshape(()).item()))
    metadata["loaded_split_counts"]["train"] += 1
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **payload)

    with pytest.raises(ValueError, match="split count"):
        load_raw_mosei_split(path, "train")


def test_builder_pools_only_train_and_validation_and_skips_test_arrays(tmp_path):
    h5py = pytest.importorskip("h5py")

    def write_features(path: Path, values: dict[str, list[list[float]]]) -> None:
        with h5py.File(path, "w") as handle:
            data = handle.create_group("data")
            for key, frames in values.items():
                entry = data.create_group(key)
                entry.create_dataset("features", data=np.asarray(frames, dtype=np.float32))

    paths = {}
    frames = {
        "train-a[0]": [[1.0, 2.0], [3.0, 4.0]],
        "train-b[0]": [[-1.0, 0.0], [-3.0, -2.0]],
        "valid-a[0]": [[2.0, 1.0], [4.0, 3.0]],
    }
    for modality in MODALITIES:
        path = tmp_path / f"{modality}.csd"
        write_features(path, frames)
        # A test entry with no feature dataset must be skipped before access.
        with h5py.File(path, "a") as handle:
            handle["data"].create_group("test-a[0]")
        paths[modality] = path
    labels = tmp_path / "labels.csd"
    write_features(
        labels,
        {
            "train-a[0]": [[1.0]],
            "train-b[0]": [[-1.0]],
            "valid-a[0]": [[1.0]],
        },
    )
    with h5py.File(labels, "a") as handle:
        handle["data"].create_group("test-a[0]")
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps({"train": ["train-a", "train-b"], "valid": ["valid-a"], "test": ["test-a"]}),
        encoding="utf-8",
    )

    bundle = build_raw_mosei_bundle(paths, labels, splits)

    assert set(bundle.splits) == {"train", "valid"}
    np.testing.assert_allclose(bundle.splits["train"].descriptors["audio"][0], [2.0, 3.0])
    assert bundle.metadata["loaded_split_counts"] == {"train": 2, "valid": 1}
    assert bundle.metadata["official_fold_counts"] == {"train": 2, "valid": 1, "test": 1}
    assert all("path" not in entry for entry in bundle.metadata["source_provenance"].values())


def test_builder_pools_official_interval_layout_by_utterance_window(tmp_path):
    h5py = pytest.importorskip("h5py")

    def write_entries(path: Path, entries: dict[str, tuple[list, list]]) -> None:
        with h5py.File(path, "w") as handle:
            data = handle.create_group("data")
            for key, (features, intervals) in entries.items():
                entry = data.create_group(key)
                entry.create_dataset("features", data=np.asarray(features, dtype=np.float32))
                entry.create_dataset("intervals", data=np.asarray(intervals, dtype=np.float32))

    labels = tmp_path / "labels.csd"
    write_entries(
        labels,
        {
            "train-video": ([[-1.0], [1.0]], [[0.0, 1.0], [1.0, 2.0]]),
            "valid-video": ([[1.0]], [[0.0, 1.0]]),
            "test-video": ([[1.0]], [[0.0, 1.0]]),
        },
    )
    modality_paths = {}
    feature_entries = {
        "train-video": (
            [[1.0, float("nan")], [3.0, 2.0], [10.0, 4.0], [14.0, 8.0]],
            [[0.0, 0.5], [0.5, 1.0], [1.0, 1.5], [1.5, 2.0]],
        ),
        "valid-video": ([[5.0, 7.0]], [[0.0, 1.0]]),
        # A malformed official-test group proves it is filtered before dataset access.
        "test-video": ([[99.0, 99.0]], [[0.0, 0.0]]),
    }
    for modality in MODALITIES:
        path = tmp_path / f"{modality}.csd"
        write_entries(path, feature_entries)
        modality_paths[modality] = path
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps(
            {"train": ["train-video"], "valid": ["valid-video"], "test": ["test-video"]}
        ),
        encoding="utf-8",
    )

    bundle = build_raw_mosei_bundle(modality_paths, labels, splits)

    assert bundle.metadata["alignment"] == "positive_interval_overlap_mean"
    assert bundle.splits["train"].sample_ids == ("train-video[0]", "train-video[1]")
    assert bundle.splits["train"].groups.tolist() == ["train-video", "train-video"]
    for modality in MODALITIES:
        np.testing.assert_allclose(
            bundle.splits["train"].descriptors[modality],
            [[2.0, 2.0], [12.0, 6.0]],
        )


@pytest.mark.parametrize(
    "left,right",
    [("train", "valid"), ("train", "test"), ("valid", "test")],
)
def test_builder_rejects_overlapping_official_fold_groups(tmp_path, left, right):
    split_values = {"train": ["train-a"], "valid": ["valid-a"], "test": ["test-a"]}
    split_values[left].append("shared-video")
    split_values[right].append("shared-video")
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps(split_values), encoding="utf-8")

    with pytest.raises(ValueError, match="overlap"):
        build_raw_mosei_bundle(
            {name: tmp_path / f"{name}.csd" for name in MODALITIES},
            tmp_path / "labels.csd",
            splits,
        )


def test_builder_rejects_duplicate_official_fold_groups(tmp_path):
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps(
            {
                "train": ["train-a", "train-a"],
                "valid": ["valid-a"],
                "test": ["test-a"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        build_raw_mosei_bundle(
            {name: tmp_path / f"{name}.csd" for name in MODALITIES},
            tmp_path / "labels.csd",
            splits,
        )


def test_projection_fit_ignores_router_and_calibration_values(tmp_path):
    first_path = _write_raw_cache(tmp_path / "first.npz")
    first = prepare_projected_data(first_path, _config_for_cache(first_path), seed=11)
    task = set(first.partition_indices["task_fit"].tolist())

    changed = _bundle()
    changed_train = changed.splits["train"]
    outside = np.asarray([index not in task for index in range(len(changed_train.y))])
    changed_descriptors = {
        modality: values.copy() for modality, values in changed_train.descriptors.items()
    }
    for values in changed_descriptors.values():
        values[outside] += 10_000.0
    changed_y = changed_train.y.copy()
    changed_y[outside] = 1 - changed_y[outside]
    changed_bundle = RawMoseiBundle(
        splits={
            "train": RawMoseiSplit(
                descriptors=changed_descriptors,
                y=changed_y,
                sample_ids=changed_train.sample_ids,
                groups=changed_train.groups,
            ),
            "valid": changed.splits["valid"],
        },
        metadata=changed.metadata,
    )
    second_path = _write_raw_cache(tmp_path / "second.npz", changed_bundle)
    second = prepare_projected_data(second_path, _config_for_cache(second_path), seed=11)

    assert first.projection_record == second.projection_record
    for modality in MODALITIES:
        np.testing.assert_allclose(
            first.projected_train.x[list(task), MODALITIES.index(modality)],
            second.projected_train.x[list(task), MODALITIES.index(modality)],
        )


def test_validation_perturbation_cannot_change_projection_or_router_fit(tmp_path):
    first_path = _write_raw_cache(tmp_path / "first.npz")
    changed = _bundle()
    valid = changed.splits["valid"]
    changed_valid = RawMoseiSplit(
        descriptors={name: values * -500.0 for name, values in valid.descriptors.items()},
        y=1 - valid.y,
        sample_ids=valid.sample_ids,
        groups=valid.groups,
    )
    second_path = _write_raw_cache(
        tmp_path / "second.npz",
        RawMoseiBundle(
            splits={"train": changed.splits["train"], "valid": changed_valid},
            metadata=changed.metadata,
        ),
    )

    first = run_raw_value_study(
        first_path, _config_for_cache(first_path), mode="singleton", seed=11
    )
    second = run_raw_value_study(
        second_path, _config_for_cache(second_path), mode="singleton", seed=11
    )

    assert first["projection_fit"] == second["projection_fit"]
    assert (
        first["router_ladder"]["training_attestation_sha256"]
        == second["router_ladder"]["training_attestation_sha256"]
    )


def test_raw_value_study_records_group_isolation_and_reuses_router_families(tmp_path):
    cache = _write_raw_cache(tmp_path / "raw.npz")
    config = _config_for_cache(cache)
    result = run_raw_value_study(cache, config, mode="singleton", seed=23)

    assert result["schema"] == RAW_RESULT_SCHEMA
    assert result["study_schema"] == RAW_STUDY_SCHEMA
    assert result["data_access"]["loaded_splits"] == ["train", "valid"]
    assert result["data_access"]["test_opened"] is False
    assert result["group_partitions"]["pairwise_disjoint"] is True
    assert set(result["router_ladder"]["families"]) == set(ROUTER_FAMILIES)
    assert result["projection_fit"]["fit_partition"] == "task_fit"
    assert (
        result["projection_fit"]["fit_group_digest"]
        == result["group_partitions"]["task_fit"]["group_digest"]
    )
    assert validate_raw_value_result(result, expected_config=config) == []


def test_raw_result_validator_rejects_forged_projection_state(tmp_path):
    cache = _write_raw_cache(tmp_path / "raw.npz")
    config = _config_for_cache(cache)
    result = run_raw_value_study(cache, config, mode="singleton", seed=23)
    result["projection_fit"]["modalities"]["audio"]["mean_sha256"] = "bad"
    payload = dict(result["projection_fit"])
    payload.pop("attestation_sha256")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    result["projection_fit"]["attestation_sha256"] = hashlib.sha256(encoded).hexdigest()

    assert "projection modality record is invalid" in validate_raw_value_result(
        result, expected_config=config
    )


def test_raw_result_validator_rejects_forged_source_provenance(tmp_path):
    cache = _write_raw_cache(tmp_path / "raw.npz")
    config = _config_for_cache(cache)
    result = run_raw_value_study(cache, config, mode="singleton", seed=23)
    result["data_access"]["source_provenance"]["audio"]["permitted_content_sha256"] = "z" * 64

    assert "raw source provenance contains an invalid entry" in validate_raw_value_result(
        result, expected_config=config
    )


def test_hidden_descriptor_values_cannot_change_prequery_router_features():
    scores = np.asarray([[0.4, 7.0, -2.0], [-0.2, -3.0, 8.0]], dtype=np.float64)
    observed = np.asarray([True, False, False])
    changed = scores.copy()
    changed[:, 1:] = [[1e12, np.nan], [-1e12, np.nan]]
    base_score = np.asarray([0.3, -0.4])

    np.testing.assert_allclose(
        _router_features(scores, observed, 1, base_score),
        _router_features(changed, observed, 1, base_score),
    )


def test_raw_lane_rejects_injected_test_arrays_without_opening_them(tmp_path):
    cache = _write_raw_cache(tmp_path / "raw.npz")
    with np.load(cache, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    inaccessible = np.asarray([{"must_not_open": True}], dtype=object)
    payload.update({"test_audio": inaccessible, "test_y": inaccessible})
    np.savez_compressed(cache, **payload)

    with pytest.raises(ValueError, match="unexpected fields"):
        run_raw_value_study(cache, _config_for_cache(cache), mode="pair", seed=11)


def test_raw_lane_requires_locked_cache_and_declared_seed(tmp_path):
    cache = _write_raw_cache(tmp_path / "raw.npz")
    unlocked = _config()
    with pytest.raises(ValueError, match="lock the generated raw cache"):
        run_raw_value_study(cache, unlocked, mode="singleton", seed=11)

    locked = _config_for_cache(cache)
    with pytest.raises(ValueError, match="seed is not in"):
        run_raw_value_study(cache, locked, mode="singleton", seed=999)


def test_raw_value_runner_cli_writes_a_valid_exclusive_result(tmp_path):
    cache = _write_raw_cache(tmp_path / "raw.npz")
    config = _config_for_cache(cache)
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "result.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    module = _load_script(RAW_RUN_SCRIPT)

    exit_code = module.main(
        [
            "--config",
            str(config_path),
            "--cache",
            str(cache),
            "--mode",
            "singleton",
            "--seed",
            "11",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["schema"] == RAW_RESULT_SCHEMA
    assert record["validation"] == {"ok": True, "errors": []}


def test_raw_cache_cli_builds_the_distinct_schema(tmp_path):
    h5py = pytest.importorskip("h5py")

    def write_csd(path: Path, rows: dict[str, list[float]]) -> None:
        with h5py.File(path, "w") as handle:
            data = handle.create_group("data")
            for key, values in rows.items():
                entry = data.create_group(key)
                entry.create_dataset("features", data=np.asarray(values, dtype=np.float32))

    rows = {
        "train-a[0]": [1.0, 2.0],
        "train-b[0]": [-1.0, 0.0],
        "valid-a[0]": [2.0, 1.0],
    }
    sources = {}
    for modality in MODALITIES:
        sources[modality] = tmp_path / f"{modality}.csd"
        write_csd(sources[modality], rows)
    labels = tmp_path / "labels.csd"
    write_csd(labels, {key: [1.0 if values[0] > 0 else -1.0] for key, values in rows.items()})
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps({"train": ["train-a", "train-b"], "valid": ["valid-a"], "test": ["test-a"]}),
        encoding="utf-8",
    )
    output = tmp_path / "raw.npz"
    module = _load_script(RAW_CACHE_SCRIPT)

    exit_code = module.main(
        [
            "--audio",
            str(sources["audio"]),
            "--video",
            str(sources["video"]),
            "--text",
            str(sources["text"]),
            "--labels",
            str(labels),
            "--splits",
            str(splits),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    _, metadata = load_raw_mosei_split(output, "train")
    assert metadata["cache_schema"] == RAW_CACHE_SCHEMA
