import numpy as np
from pathlib import Path
import json
import subprocess
import sys
import importlib.util

from conflictbench.identifiability import (
    calibrate_candidate_values,
    calibrated_risk_aware_actions,
    candidate_prior_actions,
    cue_policy_actions,
    evaluate_gain,
    evaluate_source_recovery,
    generate_controlled_value_simulation,
    oracle_actions,
    random_actions,
    risk_aware_actions,
)


ROOT = Path(__file__).parents[1]
AGGREGATE_SCRIPT = ROOT / "scripts" / "aggregate_identifiability.py"
SPEC = importlib.util.spec_from_file_location("aggregate_identifiability", AGGREGATE_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AGGREGATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGGREGATOR)
CALIBRATED_AGGREGATE_SCRIPT = ROOT / "scripts" / "aggregate_calibrated_identifiability.py"
CALIBRATED_SPEC = importlib.util.spec_from_file_location(
    "aggregate_calibrated_identifiability", CALIBRATED_AGGREGATE_SCRIPT
)
assert CALIBRATED_SPEC is not None and CALIBRATED_SPEC.loader is not None
CALIBRATED_AGGREGATOR = importlib.util.module_from_spec(CALIBRATED_SPEC)
CALIBRATED_SPEC.loader.exec_module(CALIBRATED_AGGREGATOR)


def test_zero_observable_cue_matches_random_policy_at_exact_budget():
    """A router cannot beat matched random choice when its cue is independent."""

    simulation = generate_controlled_value_simulation(
        seed=17,
        samples=40_000,
        oracle_headroom=0.20,
        cue_strength=0.0,
    )
    cue = evaluate_gain(simulation, cue_policy_actions(simulation, budget=0.50, seed=23))
    random = evaluate_gain(simulation, random_actions(simulation, budget=0.50, seed=23))
    prior = evaluate_gain(simulation, candidate_prior_actions(simulation, budget=0.50, seed=23))

    assert abs(cue["mean_gain"] - random["mean_gain"]) < 0.005
    assert abs(cue["mean_gain"] - prior["mean_gain"]) < 0.005
    assert cue["query_rate"] == 0.50
    assert random["query_rate"] == 0.50
    assert prior["query_rate"] == 0.50


def test_observable_cue_increases_feasible_value_without_changing_oracle_headroom():
    """Changing cue strength must not change the value available to the oracle."""

    weak = generate_controlled_value_simulation(
        seed=31,
        samples=40_000,
        oracle_headroom=0.20,
        cue_strength=0.20,
    )
    strong = generate_controlled_value_simulation(
        seed=31,
        samples=40_000,
        oracle_headroom=0.20,
        cue_strength=0.80,
    )

    weak_cue = evaluate_gain(weak, cue_policy_actions(weak, budget=0.50, seed=41))
    strong_cue = evaluate_gain(strong, cue_policy_actions(strong, budget=0.50, seed=41))
    weak_oracle = evaluate_gain(weak, oracle_actions(weak, budget=0.50, seed=41))
    strong_oracle = evaluate_gain(strong, oracle_actions(strong, budget=0.50, seed=41))

    assert strong_cue["mean_gain"] > weak_cue["mean_gain"] + 0.05
    assert np.isclose(weak_oracle["mean_gain"], strong_oracle["mean_gain"], atol=1e-12)
    assert np.isclose(strong_oracle["mean_gain"], 0.10, atol=1e-12)


def test_oracle_uses_hidden_candidate_only_as_an_explicit_upper_reference():
    """The oracle must strictly dominate a feasible cue policy when its cue is imperfect."""

    simulation = generate_controlled_value_simulation(
        seed=53,
        samples=40_000,
        oracle_headroom=0.20,
        cue_strength=0.60,
    )
    cue = evaluate_gain(simulation, cue_policy_actions(simulation, budget=0.50, seed=67))
    oracle = evaluate_gain(simulation, oracle_actions(simulation, budget=0.50, seed=67))

    assert oracle["mean_gain"] > cue["mean_gain"]
    assert oracle["uses_hidden_outcome"] is True
    assert cue["uses_hidden_outcome"] is False


def test_cost_aware_policy_abstains_without_positive_prequery_value():
    low = generate_controlled_value_simulation(
        seed=61, samples=40_000, oracle_headroom=0.20, cue_strength=0.0, candidate_costs=(0.02, 0.02)
    )
    high = generate_controlled_value_simulation(
        seed=61, samples=40_000, oracle_headroom=0.20, cue_strength=0.8, candidate_costs=(0.02, 0.02)
    )

    low_result = evaluate_gain(low, risk_aware_actions(low))
    high_result = evaluate_gain(high, risk_aware_actions(high))

    assert low_result["query_rate"] == 0.0
    assert low_result["abstention_rate"] == 1.0
    assert high_result["query_rate"] == 1.0
    assert high_result["abstention_rate"] == 0.0
    assert high_result["mean_gain"] > 0.0


def test_calibrated_policy_uses_held_out_value_estimates_for_selective_acquisition():
    """A calibration-only lower value estimate must select only the affordable source."""

    calibration = generate_controlled_value_simulation(
        seed=83,
        samples=40_000,
        oracle_headroom=0.20,
        cue_strength=0.40,
        candidate_costs=(0.02, 0.10),
    )
    evaluation = generate_controlled_value_simulation(
        seed=89,
        samples=40_000,
        oracle_headroom=0.20,
        cue_strength=0.40,
        candidate_costs=(0.02, 0.10),
    )

    calibration_values = calibrate_candidate_values(calibration)
    actions = calibrated_risk_aware_actions(evaluation, calibration_values)
    result = evaluate_gain(evaluation, actions)
    recovery = evaluate_source_recovery(evaluation, actions)

    assert calibration_values.lower_expected_gain[0] > 0.02
    assert calibration_values.lower_expected_gain[1] < 0.10
    assert 0.45 < result["query_rate"] < 0.55
    assert recovery["0"]["query_count"] > 0
    assert recovery["0"]["post_query_helpful_rate"] > 0.5
    assert recovery["1"]["query_count"] == 0
    assert recovery["1"]["post_query_helpful_rate"] is None


def test_controlled_runner_records_all_policy_roles_and_cue_sweep(tmp_path):
    """The experiment record must expose feasible and oracle policies separately."""

    root = ROOT
    script = root / "scripts" / "run_identifiability_simulation.py"
    output = tmp_path / "controlled.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--seed",
            "71",
            "--samples",
            "40000",
            "--oracle-headroom",
            "0.20",
            "--budget",
            "0.50",
            "--candidate-costs",
            "0.02,0.02",
            "--cue-strengths",
            "0.0,0.8",
            "--output",
            str(output),
        ],
        env={"PYTHONPATH": str(root / "reproducibility" / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["schema"] == "conflictbench.identifiability-control.v1"
    assert record["contract"] == {
        "budget": 0.5,
        "candidate_costs": [0.02, 0.02],
        "oracle_headroom": 0.2,
        "samples": 40000,
    }
    zero = record["results"]["0.0"]
    strong = record["results"]["0.8"]
    assert abs(zero["cue_policy"]["mean_gain"] - zero["candidate_prior"]["mean_gain"]) < 0.005
    assert strong["cue_policy"]["mean_gain"] > zero["cue_policy"]["mean_gain"] + 0.05
    assert strong["oracle"]["uses_hidden_outcome"] is True
    assert strong["cue_policy"]["uses_hidden_outcome"] is False
    assert zero["risk_aware"]["abstention_rate"] == 1.0
    assert strong["risk_aware"]["query_rate"] == 1.0


def test_calibrated_runner_separates_calibration_from_evaluation(tmp_path):
    """The calibrated policy must record held-out calibration and source recovery."""

    script = ROOT / "scripts" / "run_calibrated_identifiability_simulation.py"
    output = tmp_path / "calibrated.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--seed",
            "97",
            "--calibration-seed",
            "101",
            "--samples",
            "40000",
            "--calibration-samples",
            "40000",
            "--oracle-headroom",
            "0.20",
            "--budget",
            "0.50",
            "--candidate-costs",
            "0.02,0.10",
            "--cue-strengths",
            "0.0,0.4,0.8",
            "--output",
            str(output),
        ],
        env={"PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["schema"] == "conflictbench.calibrated-identifiability-control.v1"
    assert record["contract"]["calibration_samples"] == 40000
    assert record["calibration"]["0.4"]["lower_expected_gain"][0] > 0.02
    strong = record["results"]["0.4"]
    assert strong["calibrated_risk_aware"]["query_rate"] > 0.45
    assert strong["source_recovery"]["0"]["query_count"] > 0
    assert strong["source_recovery"]["1"]["query_count"] == 0
    # The saved 0.50 budget binds fixed-budget comparators, while this
    # calibration rule can request both candidates when each clears cost.
    assert record["contract"]["budget"] == 0.50
    assert record["results"]["0.8"]["calibrated_risk_aware"]["query_rate"] == 1.0


def _seed_record(seed: int, weak: float, strong: float) -> dict[str, object]:
    return {
        "schema": "conflictbench.identifiability-control.v1",
        "seed": seed,
        "contract": {"samples": 40000, "oracle_headroom": 0.2, "budget": 0.5},
        "results": {
            "0.0": {
                "candidate_prior": {"mean_gain": 0.0, "query_rate": 0.5, "uses_hidden_outcome": False},
                "random": {"mean_gain": 0.0, "query_rate": 0.5, "uses_hidden_outcome": False},
                "cue_policy": {"mean_gain": weak, "query_rate": 0.5, "uses_hidden_outcome": False},
                "oracle": {"mean_gain": 0.1, "query_rate": 0.5, "uses_hidden_outcome": True},
                "risk_aware": {"mean_gain": 0.0, "query_rate": 0.0, "uses_hidden_outcome": False},
            },
            "0.8": {
                "candidate_prior": {"mean_gain": 0.0, "query_rate": 0.5, "uses_hidden_outcome": False},
                "random": {"mean_gain": 0.0, "query_rate": 0.5, "uses_hidden_outcome": False},
                "cue_policy": {"mean_gain": strong, "query_rate": 0.5, "uses_hidden_outcome": False},
                "oracle": {"mean_gain": 0.1, "query_rate": 0.5, "uses_hidden_outcome": True},
                "risk_aware": {"mean_gain": strong - 0.02, "query_rate": 1.0, "uses_hidden_outcome": False},
            },
        },
    }


def test_aggregate_requires_every_expected_seed_and_reports_policy_gap(tmp_path):
    """A partial seed set must not become a scientific aggregate."""

    (tmp_path / "seed-101.json").write_text(json.dumps(_seed_record(101, 0.001, 0.079)), encoding="utf-8")
    (tmp_path / "seed-102.json").write_text(json.dumps(_seed_record(102, -0.001, 0.081)), encoding="utf-8")

    record = AGGREGATOR.aggregate_results(tmp_path, expected_seeds=[101, 102])

    assert record["schema"] == "conflictbench.identifiability-aggregate.v1"
    assert record["seed_count"] == 2
    assert record["results"]["0.8"]["cue_minus_random_mean"] == 0.08
    assert record["results"]["0.8"]["oracle_mean"] == 0.1
    with np.testing.assert_raises_regex(ValueError, "missing expected seed outputs"):
        AGGREGATOR.aggregate_results(tmp_path, expected_seeds=[101, 102, 103])


def test_calibrated_aggregate_pools_source_specific_post_query_recovery(tmp_path):
    """A complete calibrated seed family must retain source-specific recovery."""

    contract = {
        "samples": 40,
        "calibration_samples": 40,
        "oracle_headroom": 0.2,
        "budget": 0.5,
        "candidate_costs": [0.02, 0.10],
    }
    for seed, value in ((301, 0.03), (302, 0.05)):
        record = {
            "schema": "conflictbench.calibrated-identifiability-control.v1",
            "seed": seed,
            "contract": contract,
            "results": {
                "0.4": {
                    "calibrated_risk_aware": {"mean_gain": value, "query_rate": 0.5},
                    "source_recovery": {
                        "0": {"query_count": 10, "post_query_helpful_count": 7},
                        "1": {"query_count": 0, "post_query_helpful_count": 0},
                    },
                }
            },
        }
        (tmp_path / f"seed-{seed}.json").write_text(json.dumps(record), encoding="utf-8")

    aggregate = CALIBRATED_AGGREGATOR.aggregate_results(tmp_path, expected_seeds=[301, 302])

    summary = aggregate["results"]["0.4"]
    assert summary["calibrated_risk_aware_mean"] == 0.04
    assert summary["calibrated_query_rate_mean"] == 0.5
    assert summary["source_recovery"]["0"] == {
        "query_count": 20,
        "post_query_helpful_count": 14,
        "post_query_helpful_rate": 0.7,
    }
