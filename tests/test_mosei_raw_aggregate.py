from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conflictbench.mosei_raw_value_study import run_raw_value_study

ROOT = Path(__file__).parents[1]
AGGREGATE_SCRIPT = ROOT / "scripts" / "aggregate_mosei_raw_value.py"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_mosei_raw_value.py"
RAW_TEST_MODULE = Path(__file__).with_name("test_mosei_raw_value_study.py")
LOCKED_SEEDS = (11, 23, 37, 41, 53)


def _load_raw_test_helpers():
    spec = importlib.util.spec_from_file_location("raw_value_test_helpers", RAW_TEST_MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def raw_campaign(tmp_path_factory):
    helpers = _load_raw_test_helpers()
    directory = tmp_path_factory.mktemp("raw-aggregate")
    cache = helpers._write_raw_cache(directory / "raw.npz")
    cache_sha256 = hashlib.sha256(cache.read_bytes()).hexdigest()
    config = helpers._config()
    config.update(
        {
            "seeds": list(LOCKED_SEEDS),
            "expected_cache_sha256": cache_sha256,
            "evidence_status": "exploratory_taskfit_reprojection",
            # Force a valid negative scientific outcome for the regression test.
            "minimum_oracle_error_reduction": 1.0,
            "minimum_primary_error_reduction_gain": 1.0,
            "minimum_mask_validation_accuracy": 1.0,
        }
    )
    config_path = directory / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    paths: dict[str, list[Path]] = {"singleton": [], "pair": []}
    for mode, mode_paths in paths.items():
        for seed in LOCKED_SEEDS:
            record = run_raw_value_study(cache, config, mode=mode, seed=seed)
            path = directory / f"{mode}-{seed}.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            mode_paths.append(path)
    return directory, config_path, paths


def _run_aggregate(config: Path, singleton: list[Path], pair: list[Path], output: Path):
    return subprocess.run(
        [
            sys.executable,
            str(AGGREGATE_SCRIPT),
            "--config",
            str(config),
            "--singleton",
            *(str(path) for path in singleton),
            "--pair",
            *(str(path) for path in pair),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _mutated_copy(source: Path, destination: Path, mutate) -> Path:
    record = json.loads(source.read_text(encoding="utf-8"))
    mutate(record)
    destination.write_text(json.dumps(record), encoding="utf-8")
    return destination


def test_raw_aggregate_accepts_structurally_valid_negative_result(raw_campaign):
    directory, config, paths = raw_campaign
    output = directory / "aggregate.json"

    completed = _run_aggregate(config, paths["singleton"], paths["pair"], output)

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["schema"] == "conflictbench.mosei-raw-value-aggregate.v1"
    assert record["seeds"] == list(LOCKED_SEEDS)
    assert record["validation"] == {"ok": True, "errors": []}
    assert record["scientific_result"]["status"] == "negative"
    assert record["scientific_result"]["passed"] is False
    assert record["evidence_status"] == "exploratory_taskfit_reprojection"
    assert set(record["projection_attestations_by_seed"]) == {
        str(seed) for seed in LOCKED_SEEDS
    }
    for mode in ("singleton", "pair"):
        for contrast in record["modes"][mode]["pooled_video_group_contrasts"].values():
            assert contrast["bootstrap_method"] == (
                "aligned_video_cluster_resample_with_policy_reselection_then_seed_mean"
            )


def test_raw_aggregate_separates_multiseed_online_fixed_threshold_gate(raw_campaign):
    directory, config_path, paths = raw_campaign
    output = directory / "online-fixed-threshold-aggregate.json"

    completed = _run_aggregate(config_path, paths["singleton"], paths["pair"], output)

    assert completed.returncode == 0, completed.stderr
    aggregate = json.loads(output.read_text(encoding="utf-8"))
    assert aggregate["scientific_result"]["evaluation_protocol"] == (
        "offline_exact_cohort_query_ranking_with_budget_reselection"
    )
    offline = aggregate["scientific_result"]["gates"]["selected_router_selection"]
    assert offline["comparison"] == (
        "offline_selected_router_minus_calibration_selected_baseline"
    )

    singleton_records = [
        json.loads(path.read_text(encoding="utf-8")) for path in paths["singleton"]
    ]
    summary = aggregate["modes"]["singleton"]["online_query_policy"]
    expected_seed_keys = {str(seed) for seed in LOCKED_SEEDS}
    assert summary["fit_split"] == "calibration"
    assert summary["evaluation_batch_access"] == "independent_per_example"
    for field in (
        "selected_family_by_seed",
        "router_training_attestation_sha256_by_seed",
        "calibration_priority_sha256_by_seed",
        "evaluation_priority_sha256_by_seed",
        "evaluation_choice_sha256_by_seed",
        "evaluation_attestation_sha256_by_seed",
    ):
        assert set(summary[field]) == expected_seed_keys

    budget = "0.500000"
    budget_summary = summary["by_target_budget"][budget]
    assert set(budget_summary["fixed_threshold_by_seed"]) == expected_seed_keys
    assert set(budget_summary["query_on_equal_by_seed"]) == expected_seed_keys
    expected_draws = np.asarray(
        [
            record["bootstrap"]["policy_error_reduction_draws"]
            ["online_selected_router"][budget]
            for record in singleton_records
        ],
        dtype=np.float64,
    )
    paired = budget_summary["paired_error_reduction_vs_no_query"]
    np.testing.assert_allclose(
        paired["cluster_ci95"],
        np.quantile(np.mean(expected_draws, axis=0), (0.025, 0.975)),
    )
    assert paired["mean"] == pytest.approx(
        np.mean(
            [
                record["evaluation"]["policies"]["online_selected_router"][budget][
                    "error_reduction"
                ]
                for record in singleton_records
            ]
        )
    )
    assert paired["bootstrap_method"] == (
        "aligned_video_cluster_resample_with_fixed_calibration_threshold_then_seed_mean"
    )

    matched_oracle = budget_summary["realized_rate_matched_oracle"]
    assert matched_oracle["regret"]["runs"] == len(LOCKED_SEEDS)
    assert matched_oracle["regret"]["mean"] == pytest.approx(
        np.mean(
            [
                record["evaluation"]["policies"]["online_selected_router"][budget][
                    "oracle_regret"
                ]
                for record in singleton_records
            ]
        )
    )
    assert set(matched_oracle["by_seed"]) == expected_seed_keys

    gate = aggregate["online_fixed_threshold_gate"]
    assert gate["comparison"] == "online_selected_router_minus_no_query"
    assert gate["evaluation_protocol"] == (
        "calibration_fitted_fixed_threshold_per_example"
    )
    assert gate["target_budget"] == 0.5
    assert gate["paired_error_reduction_vs_no_query"] == paired
    assert gate["realized_rate_matched_oracle"] == matched_oracle
    assert gate["passed"] is (
        paired["mean"] >= 0.0 and paired["cluster_ci95"][0] > 0.0
    )
    assert gate["status"] == ("positive" if gate["passed"] else "negative")


def test_raw_aggregate_requires_exact_locked_mode_seed_grid(raw_campaign):
    directory, config, paths = raw_campaign
    output = directory / "missing-seed.json"

    completed = _run_aggregate(config, paths["singleton"][:-1], paths["pair"], output)

    assert completed.returncode != 0
    assert "seed set mismatch" in completed.stderr
    assert not output.exists()


def test_raw_aggregate_rejects_cross_mode_projection_drift(raw_campaign):
    directory, config, paths = raw_campaign

    def mutate(record):
        projection = record["projection_fit"]
        projection["modalities"]["audio"]["mean_sha256"] = "f" * 64
        payload = copy.deepcopy(projection)
        payload.pop("attestation_sha256")
        projection["attestation_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    changed = _mutated_copy(
        paths["pair"][0], directory / "pair-11-projection-drift.json", mutate
    )
    pair = [changed, *paths["pair"][1:]]
    output = directory / "projection-drift.json"

    completed = _run_aggregate(config, paths["singleton"], pair, output)

    assert completed.returncode != 0
    assert "projection" in completed.stderr.lower()
    assert not output.exists()


@pytest.mark.parametrize(
    ("name", "mutate", "expected"),
    [
        (
            "core",
            lambda record: record.pop("core_protocol"),
            "core protocol",
        ),
        (
            "cache",
            lambda record: record["data_access"].__setitem__("cache_sha256", "e" * 64),
            "cache sha256",
        ),
        (
            "bootstrap",
            lambda record: record["bootstrap"].__setitem__("method", "utterance_resample"),
            "bootstrap",
        ),
    ],
)
def test_raw_aggregate_rejects_incomplete_or_unlocked_records(
    raw_campaign, name, mutate, expected
):
    directory, config, paths = raw_campaign
    changed = _mutated_copy(paths["singleton"][0], directory / f"{name}.json", mutate)
    singleton = [changed, *paths["singleton"][1:]]
    output = directory / f"{name}-aggregate.json"

    completed = _run_aggregate(config, singleton, paths["pair"], output)

    assert completed.returncode != 0
    assert expected in completed.stderr.lower()
    assert not output.exists()


def test_raw_verifier_binds_full_result_and_projection_attestation(raw_campaign):
    directory, config, paths = raw_campaign
    result = paths["singleton"][0]
    verification = directory / "singleton-11.verification.json"
    cache = directory / "raw.npz"

    written = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT),
            str(result),
            "--config",
            str(config),
            "--cache",
            str(cache),
            "--mode",
            "singleton",
            "--seed",
            "11",
            "--write-verification",
            str(verification),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert written.returncode == 0, written.stderr
    saved = json.loads(verification.read_text(encoding="utf-8"))
    source = json.loads(result.read_text(encoding="utf-8"))
    assert saved["schema"] == "conflictbench.mosei-raw-value-verification.v1"
    assert saved["result_sha256"] == hashlib.sha256(result.read_bytes()).hexdigest()
    assert saved["projection_attestation_sha256"] == source["projection_fit"][
        "attestation_sha256"
    ]
    assert saved["core_protocol"] == source["core_protocol"]

    checked = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT),
            str(result),
            "--config",
            str(config),
            "--cache",
            str(cache),
            "--mode",
            "singleton",
            "--seed",
            "11",
            "--check-verification",
            str(verification),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr


def test_raw_verifier_rejects_result_changed_after_verification(raw_campaign):
    directory, config, paths = raw_campaign
    original = paths["pair"][0]
    result = directory / "pair-11-verified.json"
    result.write_bytes(original.read_bytes())
    verification = directory / "pair-11.verification.json"
    cache = directory / "raw.npz"
    arguments = [
        sys.executable,
        str(VERIFY_SCRIPT),
        str(result),
        "--config",
        str(config),
        "--cache",
        str(cache),
        "--mode",
        "pair",
        "--seed",
        "11",
    ]
    written = subprocess.run(
        [*arguments, "--write-verification", str(verification)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert written.returncode == 0, written.stderr
    record = json.loads(result.read_text(encoding="utf-8"))
    record["evaluation"]["split"] = "changed"
    result.write_text(json.dumps(record), encoding="utf-8")

    checked = subprocess.run(
        [*arguments, "--check-verification", str(verification)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert checked.returncode != 0
    assert "verification record" in checked.stdout.lower()
