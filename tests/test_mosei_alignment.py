import json

import numpy as np
import pytest
from conflictbench import mosei
from conflictbench.mosei import load_mosei_bundle, load_mosei_cache


def test_video_sequences_are_aligned_to_each_label_interval(tmp_path):
    h5py = pytest.importorskip("h5py")
    paths = {}
    for modality in ("audio", "video", "text", "labels"):
        path = tmp_path / f"{modality}.csd"
        paths[modality] = path
        with h5py.File(path, "w") as handle:
            group = handle.create_group("data")
            for key in ("train", "valid", "test"):
                entry = group.create_group(key)
                if modality == "labels":
                    features = [[-1.0, 0.0], [1.0, 0.0]]
                    intervals = [[0.0, 1.0], [2.0, 3.0]]
                else:
                    features = [[-2.0], [99.0], [2.0]]
                    intervals = [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
                entry.create_dataset("features", data=features)
                entry.create_dataset("intervals", data=intervals)
    folds = tmp_path / "folds.json"
    folds.write_text(json.dumps({"train": ["train"], "valid": ["valid"], "test": ["test"]}))
    bundle = load_mosei_bundle(
        {name: paths[name] for name in ("audio", "video", "text")}, paths["labels"], folds
    )
    assert bundle.sample_ids["train"] == ("train[0]", "train[1]")
    np.testing.assert_array_equal(bundle.splits["train"].y, [0, 1])
    np.testing.assert_allclose(bundle.splits["valid"].x, [[-1] * 3, [1] * 3])
    assert bundle.metadata["alignment"] == "positive_interval_overlap_mean"
    with h5py.File(paths["labels"], "r+") as handle:
        handle["data/test/features"][...] = [[1.0, 0.0], [-1.0, 0.0]]
    changed = load_mosei_bundle(
        {name: paths[name] for name in ("audio", "video", "text")}, paths["labels"], folds
    )
    np.testing.assert_array_equal(changed.splits["train"].x, bundle.splits["train"].x)
    np.testing.assert_array_equal(changed.splits["test"].x, bundle.splits["test"].x)


def test_rejects_legacy_video_averaged_cache(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, cache_schema=np.asarray("conflictbench.mosei-cache.v1"))
    with pytest.raises(ValueError, match="unsupported MOSEI cache schema"):
        load_mosei_cache(path)


def test_split_file_rejects_multiple_aliases_for_one_canonical_split(tmp_path):
    folds = tmp_path / "folds.json"
    folds.write_text(
        json.dumps(
            {
                "train": ["train"],
                "valid": ["valid-a"],
                "validation": ["valid-b"],
                "test": ["test"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="multiple MOSEI split entries map to 'valid'"):
        mosei._read_split_file(folds)
