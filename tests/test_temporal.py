import numpy as np
import pytest

from conflictbench.core import ActionOutput
from conflictbench.runner import run_from_config
from conflictbench.temporal import (
    TEMPORAL_MECHANISMS,
    generate_temporal_dataset,
    temporal_summary,
    temporal_linear_actions,
    temporal_template_actions,
    transform_temporal_dataset,
)


def test_temporal_generator_is_deterministic_and_preserves_sequence_shape():
    kwargs = {
        "seq_len": 20,
        "mechanism_probs": {"clean": 0.5, "burst": 0.3, "ambiguity": 0.2},
    }
    first = generate_temporal_dataset(11, 40, ["clean", "burst", "ambiguity"], **kwargs)
    second = generate_temporal_dataset(11, 40, ["clean", "burst", "ambiguity"], **kwargs)
    np.testing.assert_array_equal(first.x, second.x)
    np.testing.assert_array_equal(first.y, second.y)
    np.testing.assert_array_equal(first.ambiguous, second.ambiguous)
    assert first.x.shape == (40, 3, 20)
    assert np.isfinite(first.x).all()
    assert np.any(first.ambiguous)


def test_temporal_summary_is_the_scalar_control_view():
    temporal = generate_temporal_dataset(3, 12, ["clean"], seq_len=8)
    summary = temporal_summary(temporal)
    assert summary.x.shape == (12, 3)
    np.testing.assert_allclose(summary.x, temporal.x.mean(axis=2), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(summary.y, temporal.y)


def test_zero_mean_marker_removes_label_signal_from_scalar_control_view():
    temporal = generate_temporal_dataset(
        19,
        64,
        ["clean"],
        seq_len=32,
        noise=0.0,
        mechanism_probs={"clean": 1.0},
    )
    # The marker itself has no DC component, so a time-mean control cannot
    # recover the hidden sign in the noiseless case.
    assert float(np.max(np.abs(temporal_summary(temporal).x))) < 2e-6


def test_temporal_diagnostic_transforms_are_deterministic_and_preserve_shape():
    dataset = generate_temporal_dataset(29, 16, ["clean"], seq_len=16, noise=0.0)
    shared_a = transform_temporal_dataset(dataset, mode="shared_permutation", seed=7)
    shared_b = transform_temporal_dataset(dataset, mode="shared_permutation", seed=7)
    shifted = transform_temporal_dataset(dataset, mode="per_example_circular_shift", seed=7)
    np.testing.assert_array_equal(shared_a.x, shared_b.x)
    assert shared_a.x.shape == dataset.x.shape
    np.testing.assert_allclose(np.sort(shifted.x, axis=2), np.sort(dataset.x, axis=2))


def test_train_only_templates_decode_clean_noiseless_traces():
    train = generate_temporal_dataset(31, 120, ["clean"], seq_len=32, noise=0.0)
    evaluation = generate_temporal_dataset(37, 80, ["clean"], seq_len=32, noise=0.0)
    output = temporal_template_actions(train, evaluation)
    np.testing.assert_array_equal(output.action, evaluation.y)
    assert not output.query.any()


def test_full_trace_linear_control_decodes_clean_noiseless_traces():
    train = generate_temporal_dataset(41, 160, ["clean"], seq_len=32, noise=0.0)
    evaluation = generate_temporal_dataset(43, 80, ["clean"], seq_len=32, noise=0.0)
    output = temporal_linear_actions(train, evaluation)
    np.testing.assert_array_equal(output.action, evaluation.y)
    assert not output.query.any()


def test_temporal_mechanisms_are_distinct_and_finite():
    assert {"segment_invert", "occlusion", "drift"}.issubset(TEMPORAL_MECHANISMS)
    inverted = generate_temporal_dataset(23, 80, ["invert"], seq_len=32, noise=0.0)
    swapped = generate_temporal_dataset(23, 80, ["swap"], seq_len=32, noise=0.0)
    segment = generate_temporal_dataset(23, 80, ["segment_invert"], seq_len=32, noise=0.0)
    occluded = generate_temporal_dataset(23, 80, ["occlusion"], seq_len=32, noise=0.0)
    drifted = generate_temporal_dataset(23, 80, ["drift"], seq_len=32, noise=0.0)
    assert not np.array_equal(inverted.x, swapped.x)
    assert not np.array_equal(inverted.x, segment.x)
    assert not np.array_equal(segment.x, occluded.x)
    assert np.isfinite(np.stack([swapped.x, segment.x, occluded.x, drifted.x])).all()
    np.testing.assert_allclose(drifted.x.mean(axis=2), 0.0, atol=2e-6)


def test_temporal_runner_requires_sequence_methods(monkeypatch):
    import conflictbench.runner as runner

    monkeypatch.setattr(
        runner,
        "torch_status",
        lambda: {"available": True, "cuda_available": False, "version": "test"},
    )

    def scalar_fit(_method, _train, evaluation_splits, **_kwargs):
        outputs = {
            name: ActionOutput(np.zeros(len(split.y), dtype=np.int64), np.zeros(len(split.y), dtype=bool))
            for name, split in evaluation_splits.items()
        }
        return outputs, {"status": "verified"}

    def temporal_fit(_method, _train, evaluation_splits, **_kwargs):
        outputs = {
            name: ActionOutput(np.zeros(len(split.y), dtype=np.int64), np.zeros(len(split.y), dtype=bool))
            for name, split in evaluation_splits.items()
        }
        return outputs, {"status": "verified"}

    monkeypatch.setattr(runner, "fit_and_predict_torch", scalar_fit)
    monkeypatch.setattr(runner, "fit_and_predict_acquisition_torch", scalar_fit)
    monkeypatch.setattr(runner, "fit_and_predict_temporal_torch", temporal_fit)

    result = run_from_config(
        {
            "dataset": "synthetic_temporal",
            "train_size": 24,
            "eval_size": 16,
            "train_mechanisms": ["clean", "invert"],
            "seen_mechanisms": ["clean", "invert"],
            "unseen_mechanisms": ["ambiguity", "burst"],
            "seq_len": 12,
            "seed": 5,
        },
        device_preference="cpu",
    )
    assert result["validation"]["ok"]
    assert result["validation"]["temporal_methods_verified"]
    for split in result["splits"].values():
        assert {"temporal_fusion", "temporal_gated"}.issubset(split["methods"])
        assert split["methods"]["temporal_fusion"]["status"] == "verified"
    assert result["dataset"]["temporal"]["summary"] == "mean_over_time_for_scalar_controls"


def test_temporal_config_requires_a_valid_sequence_length():
    with pytest.raises(ValueError, match="seq_len"):
        run_from_config(
            {
                "dataset": "synthetic_temporal",
                "train_size": 4,
                "eval_size": 4,
                "train_mechanisms": ["clean"],
                "seen_mechanisms": ["clean"],
                "unseen_mechanisms": ["clean"],
                "seq_len": 1,
                "seed": 0,
            }
        )
