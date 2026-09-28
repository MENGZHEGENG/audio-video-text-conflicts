import numpy as np
import pytest

from conflictbench.core import (
    ActionOutput,
    Dataset,
    acquisition_actions,
    acquisition_feature_matrix,
    acquisition_training_targets,
    evaluate_actions,
    generate_dataset,
    policy_actions,
)
from conflictbench.runner import run_from_config
from conflictbench.mosei import load_mosei_bundle, load_mosei_cache, read_csd, save_mosei_cache
from conflictbench.multibench import load_descriptor_cache, load_mmsa_bundle, save_descriptor_cache


def test_generator_is_deterministic_and_finite():
    a = generate_dataset(7, 80, ["clean", "invert", "ambiguity"])
    b = generate_dataset(7, 80, ["clean", "invert", "ambiguity"])
    assert np.array_equal(a.x, b.x)
    assert np.array_equal(a.y, b.y)
    assert np.isfinite(a.x).all()


def test_active_policy_requests_on_conflict_and_can_abstain():
    values = np.array([[1.0, -1.0, 1.0], [1.0, -1.0, 0.0]], dtype=np.float32)
    output = policy_actions("active_diagnostic", values)
    assert output.query.tolist() == [True, True]
    assert output.action.tolist() == [1, 2]


def test_learned_acquisition_uses_pair_for_nonqueries_and_full_scores_after_query():
    values = np.array(
        [[1.0, 1.0, -1.0], [1.0, -1.0, 1.0], [1.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    output = acquisition_actions(values, np.array([0.1, 0.9, 0.9]), threshold=0.5)
    assert output.query.tolist() == [False, True, True]
    assert output.action.tolist() == [1, 1, 2]


def test_acquisition_targets_are_binary_and_observation_shape_is_unchanged():
    dataset = generate_dataset(17, 120, ["clean", "invert", "dropout", "ambiguity"])
    targets = acquisition_training_targets(dataset)
    assert targets.shape == (120,)
    assert set(np.unique(targets)).issubset({0.0, 1.0})
    assert dataset.x.shape == (120, 3)


def test_acquisition_features_are_invariant_to_unavailable_text():
    dataset = Dataset(
        x=np.array([[0.3, -0.7, 2.0], [-0.5, 0.4, -3.0]], dtype=np.float32),
        y=np.array([1, 0], dtype=np.int64),
        ambiguous=np.array([False, False]),
        mechanism=np.array(["clean", "clean"]),
    )
    changed_text = Dataset(
        x=np.array([[0.3, -0.7, -99.0], [-0.5, 0.4, 99.0]], dtype=np.float32),
        y=dataset.y,
        ambiguous=dataset.ambiguous,
        mechanism=dataset.mechanism,
    )
    np.testing.assert_array_equal(acquisition_feature_matrix(dataset), acquisition_feature_matrix(changed_text))
    np.testing.assert_array_equal(
        acquisition_feature_matrix(dataset),
        np.array([[0.3, -0.7, 0.3, 0.7], [-0.5, 0.4, 0.5, 0.4]], dtype=np.float32),
    )


def test_metrics_charge_unsafe_abstention_and_queries():
    dataset = Dataset(
        x=np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 0.0]], dtype=np.float32),
        y=np.array([1, 0], dtype=np.int64),
        ambiguous=np.array([False, True]),
        mechanism=np.array(["clean", "ambiguity"]),
    )
    output = policy_actions("active_diagnostic", dataset.x)
    result = evaluate_actions(output, dataset)
    assert result["metrics"]["coverage"] == 0.5
    assert result["metrics"]["ambiguous_abstain_rate"] == 1.0


def test_utility_matches_signed_correctness_definition():
    dataset = Dataset(
        x=np.zeros((3, 3), dtype=np.float32),
        y=np.array([1, 0, 1], dtype=np.int64),
        ambiguous=np.array([False, False, True]),
        mechanism=np.array(["clean", "clean", "ambiguity"]),
    )
    output = ActionOutput(
        action=np.array([1, 2, 0], dtype=np.int64),
        query=np.array([False, True, True]),
    )
    result = evaluate_actions(output, dataset)
    # +1 correct, -1 incorrect, then subtract query and unsafe-abstention costs.
    np.testing.assert_allclose(result["metrics"]["utility"], (1.0 - 1.3 - 1.1) / 3.0)


def test_metrics_encode_empty_subgroups_as_null():
    dataset = Dataset(
        x=np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        y=np.array([1], dtype=np.int64),
        ambiguous=np.array([False]),
        mechanism=np.array(["clean"]),
    )
    result = evaluate_actions(policy_actions("majority", dataset.x), dataset)
    assert result["metrics"]["ambiguous_abstain_rate"] is None


def test_runner_has_required_splits_and_methods():
    config = {
        "train_size": 60,
        "eval_size": 80,
        "train_mechanisms": ["clean", "invert"],
        "seen_mechanisms": ["clean", "invert"],
        "unseen_mechanisms": ["clean", "ambiguity"],
        "seed": 3,
        "epochs": 2,
        "batch_size": 16,
    }
    result = run_from_config(config)
    assert result["validation"]["ok"]
    assert set(result["splits"]) == {"seen", "unseen"}
    for split in result["splits"].values():
        assert {
            "majority",
            "weighted",
            "median",
            "active_diagnostic",
            "mlp",
            "gated",
            "learned_acquisition",
        } == set(split["methods"])


def test_mosei_loader_uses_train_projection_and_official_fold_aliases(tmp_path):
    h5py = pytest.importorskip("h5py")

    def write_csd(path, rows):
        with h5py.File(path, "w") as handle:
            group = handle.create_group("data")
            for key, value in rows.items():
                entry = group.create_group(key)
                entry.create_dataset("features", data=np.asarray(value))

    ids = ["train_a[0]", "train_b[0]", "valid_a[0]", "test_a[0]"]
    values = {
        ids[0]: [2.0, 0.0],
        ids[1]: [-2.0, 0.0],
        ids[2]: [1.0, 0.0],
        ids[3]: [-1.0, 0.0],
    }
    modality_paths = {}
    for modality, offset in zip(("audio", "video", "text"), (0.0, 0.2, -0.2)):
        path = tmp_path / f"{modality}.csd"
        write_csd(path, {key: np.asarray(value) + offset for key, value in values.items()})
        modality_paths[modality] = path
    labels_path = tmp_path / "labels.csd"
    write_csd(labels_path, {ids[0]: [1.0], ids[1]: [-1.0], ids[2]: [1.0], ids[3]: [-1.0]})
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(
        '{"training": ["train_a", "train_b"], "dev": ["valid_a"], '
        '"testing": ["test_a", "test_missing"]}\n',
        encoding="utf-8",
    )

    bundle = load_mosei_bundle(modality_paths, labels_path, splits_path)
    assert bundle.sample_ids["train"] == ("train_a[0]", "train_b[0]")
    assert bundle.sample_ids["valid"] == ("valid_a[0]",)
    assert bundle.sample_ids["test"] == ("test_a[0]",)
    assert bundle.splits["valid"].x.shape == (1, 3)
    assert bundle.metadata["projection"] == "train_standardized_mean_difference"
    assert bundle.metadata["official_fold_counts"] == {"train": 2, "valid": 1, "test": 2}
    assert bundle.metadata["observed_split_counts"] == {"train": 2, "valid": 1, "test": 1}
    assert bundle.metadata["missing_official_fold_ids"] == {
        "train": [],
        "valid": [],
        "test": ["test_missing"],
    }


def test_mosei_loader_reads_nested_official_labels_group(tmp_path):
    h5py = pytest.importorskip("h5py")

    def write_nested_labels(path, rows):
        with h5py.File(path, "w") as handle:
            root = handle.create_group("All Labels")
            group = root.create_group("data")
            root.create_group("metadata").create_dataset("description", data=np.asarray([b"fixture"]))
            for key, value in rows.items():
                entry = group.create_group(key)
                entry.create_dataset("features", data=np.asarray(value))

    ids = ["train_a[0]", "train_b[0]", "valid_a[0]", "test_a[0]"]
    rows = {ids[0]: [1.0], ids[1]: [-1.0], ids[2]: [1.0], ids[3]: [-1.0]}
    labels_path = tmp_path / "nested-labels.csd"
    write_nested_labels(labels_path, rows)

    loaded = read_csd(labels_path, reduce_sequence=False)
    assert set(loaded) == set(ids)
    np.testing.assert_array_equal(loaded[ids[0]], np.asarray([1.0]))


def test_mosei_runner_maps_validation_and_test_to_seen_and_unseen(tmp_path):
    h5py = pytest.importorskip("h5py")

    def write_csd(path, rows):
        with h5py.File(path, "w") as handle:
            group = handle.create_group("data")
            for key, value in rows.items():
                entry = group.create_group(key)
                entry.create_dataset("features", data=np.asarray(value))

    rows = {"tr0[0]": [1.0, 0.0], "tr1[0]": [-1.0, 0.0], "va0[0]": [1.0, 0.0], "te0[0]": [-1.0, 0.0]}
    paths = {}
    for modality in ("audio", "video", "text"):
        paths[modality] = tmp_path / f"{modality}.csd"
        write_csd(paths[modality], rows)
    labels = tmp_path / "labels.csd"
    write_csd(labels, {key: [1.0 if value[0] > 0 else -1.0] for key, value in rows.items()})
    folds = tmp_path / "folds.json"
    folds.write_text('{"train": ["tr0", "tr1"], "valid": ["va0"], "test": ["te0"]}\n', encoding="utf-8")
    result = __import__("conflictbench.runner", fromlist=["run_from_config"]).run_from_config(
        {
            "dataset": "mosei",
            "modality_paths": {key: str(value) for key, value in paths.items()},
            "labels_path": str(labels),
            "splits_path": str(folds),
            "seed": 9,
            "epochs": 1,
            "batch_size": 2,
        },
        device_preference="cpu",
    )
    assert result["dataset"]["dataset"] == "CMU-MOSEI"
    assert result["splits"]["seen"]["sample_ids"] == ["va0[0]"]
    assert result["splits"]["unseen"]["sample_ids"] == ["te0[0]"]


def test_mosei_cache_round_trip_preserves_scores_ids_and_metadata(tmp_path):
    h5py = pytest.importorskip("h5py")

    def write_csd(path, rows):
        with h5py.File(path, "w") as handle:
            group = handle.create_group("data")
            for key, value in rows.items():
                entry = group.create_group(key)
                entry.create_dataset("features", data=np.asarray(value))

    rows = {"tr0[0]": [1.0, 0.0], "tr1[0]": [-1.0, 0.0], "va0[0]": [1.0, 0.0], "te0[0]": [-1.0, 0.0]}
    paths = {}
    for modality in ("audio", "video", "text"):
        paths[modality] = tmp_path / f"{modality}.csd"
        write_csd(paths[modality], rows)
    labels = tmp_path / "labels.csd"
    write_csd(labels, {key: [1.0 if value[0] > 0 else -1.0] for key, value in rows.items()})
    folds = tmp_path / "folds.json"
    folds.write_text('{"train": ["tr0", "tr1"], "valid": ["va0"], "test": ["te0"]}\n', encoding="utf-8")

    original = load_mosei_bundle(paths, labels, folds)
    cache_path = save_mosei_cache(original, tmp_path / "scores.npz")
    restored = load_mosei_cache(cache_path)
    assert restored.metadata["cache_schema"] == "conflictbench.mosei-cache.v2"
    for split_name in ("train", "valid", "test"):
        np.testing.assert_allclose(restored.splits[split_name].x, original.splits[split_name].x)
        np.testing.assert_array_equal(restored.splits[split_name].y, original.splits[split_name].y)
        assert restored.sample_ids[split_name] == original.sample_ids[split_name]


def test_mmsa_loader_pools_lengths_filters_zero_and_round_trips_cache(tmp_path):
    rows = {}
    for split_name, labels in (("train", [1.0, -1.0, 0.0]), ("valid", [1.0, -1.0]), ("test", [-1.0, 1.0])):
        n = len(labels)
        rows[split_name] = {
            "id": [f"{split_name}-{i}" for i in range(n)],
            "audio": [np.asarray([[label, 2.0], [label * 2, 4.0], [99.0, 99.0]]) for label in labels],
            "vision": [np.asarray([[label + 1.0], [label + 2.0], [88.0]]) for label in labels],
            "text": [np.asarray([[label - 1.0], [label - 2.0], [77.0]]) for label in labels],
            "audio_lengths": [2] * n,
            "vision_lengths": [2] * n,
            "text_lengths": [2] * n,
            "regression_labels": labels,
        }
    source = tmp_path / "mmsa.pkl"
    import pickle
    with source.open("wb") as handle:
        pickle.dump(rows, handle)

    bundle = load_mmsa_bundle(source, dataset_name="CMU-MOSI", sentiment_mode="positive")
    assert bundle.sample_ids["train"] == ("train-0", "train-1")
    assert bundle.splits["train"].x.shape == (2, 3)
    assert bundle.metadata["projection"] == "train_standardized_mean_difference"
    assert bundle.metadata["feature_dimensions"] == {"audio": 2, "video": 1, "text": 1}
    cache = save_descriptor_cache(bundle, tmp_path / "scores.npz")
    restored = load_descriptor_cache(cache)
    for split_name in ("train", "valid", "test"):
        np.testing.assert_allclose(restored.splits[split_name].x, bundle.splits[split_name].x)
        np.testing.assert_array_equal(restored.splits[split_name].y, bundle.splits[split_name].y)
        assert restored.sample_ids[split_name] == bundle.sample_ids[split_name]


def test_runner_accepts_descriptor_cache(tmp_path):
    from conflictbench.multibench import DescriptorBundle

    splits = {}
    ids = {}
    for name, seed in (("train", 1), ("valid", 2), ("test", 3)):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(12, 3)).astype(np.float32)
        y = (x[:, 0] >= 0).astype(np.int64)
        splits[name] = Dataset(x=x, y=y, ambiguous=np.zeros(12, dtype=bool), mechanism=np.full(12, "CMU-MOSI", dtype="U16"))
        ids[name] = tuple(f"{name}-{i}" for i in range(12))
    cache = save_descriptor_cache(DescriptorBundle(splits, ids, {"dataset": "CMU-MOSI"}), tmp_path / "cache.npz")
    result = run_from_config({"dataset": "mosi", "cache_path": str(cache), "seed": 4, "epochs": 1, "batch_size": 4}, device_preference="cpu")
    assert result["validation"]["ok"]
    assert result["dataset"]["dataset"] == "CMU-MOSI"


def test_mmsa_loader_rejects_cross_split_dimension_drift(tmp_path):
    import pickle

    rows = {}
    for split_name in ("train", "valid", "test"):
        n = 2
        rows[split_name] = {
            "id": [f"{split_name}-{i}" for i in range(n)],
            "audio": [np.ones((2, 2)) for _ in range(n)],
            "vision": [np.ones((2, 1)) for _ in range(n)],
            "text": [np.ones((2, 1)) for _ in range(n)],
            "audio_lengths": [2] * n,
            "vision_lengths": [2] * n,
            "text_lengths": [2] * n,
            "regression_labels": [1.0, -1.0],
        }
    rows["test"]["vision"] = [np.ones((2, 2)) for _ in range(2)]
    source = tmp_path / "dimension-drift.pkl"
    with source.open("wb") as handle:
        pickle.dump(rows, handle)
    with pytest.raises(ValueError, match="dimensions in test do not match train"):
        load_mmsa_bundle(source, dataset_name="CMU-MOSI")


def test_mmsa_loader_rejects_all_nonfinite_sequence(tmp_path):
    import pickle

    rows = {}
    for split_name in ("train", "valid", "test"):
        rows[split_name] = {
            "id": [f"{split_name}-0", f"{split_name}-1"],
            "audio": [np.ones((2, 2)), np.full((2, 2), np.nan)],
            "vision": [np.ones((2, 1)), np.ones((2, 1))],
            "text": [np.ones((2, 1)), np.ones((2, 1))],
            "audio_lengths": [2, 2],
            "vision_lengths": [2, 2],
            "text_lengths": [2, 2],
            "regression_labels": [1.0, -1.0],
        }
    source = tmp_path / "nonfinite.pkl"
    with source.open("wb") as handle:
        pickle.dump(rows, handle)
    with pytest.raises(ValueError, match="no finite values"):
        load_mmsa_bundle(source, dataset_name="CMU-MOSI")
