#!/usr/bin/env python3
"""Evaluate aligned CREMA-D feature grids with actor-held-out routing.

This script is intentionally CPU-only.  It joins cached feature roots by
``pair_id``, holds actors out before fitting any head or router, chooses every
operating point on calibration actors, and reports actor-cluster bootstrap
intervals on evaluation actors only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class EvaluationError(ValueError):
    """Raised when cached features cannot support a valid held-out analysis."""


def _load_feature_root(root: Path, expected_shards: int) -> dict[str, np.ndarray]:
    parts = [np.load(path) for path in sorted(root.glob("features-*-shard*.npz"))]
    if len(parts) != expected_shards:
        raise EvaluationError(f"{root} has {len(parts)} shards; expected {expected_shards}")
    required = ("pair_id", "integrated_vote", "audio", "video")
    if any(any(key not in part for key in required) for part in parts):
        raise EvaluationError(f"{root} lacks a required feature key")
    loaded = {key: np.concatenate([part[key] for part in parts]) for key in required}
    if len(set(loaded["pair_id"].tolist())) != len(loaded["pair_id"]):
        raise EvaluationError(f"{root} has duplicate pair identifiers")
    return loaded


def _align(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    positions = {pair_id: index for index, pair_id in enumerate(candidate["pair_id"].tolist())}
    reference_ids = reference["pair_id"].tolist()
    if set(reference_ids) != set(positions):
        raise EvaluationError("feature roots have different pair identifiers")
    take = np.asarray([positions[pair_id] for pair_id in reference_ids], dtype=int)
    if not np.array_equal(reference["integrated_vote"], candidate["integrated_vote"][take]):
        raise EvaluationError("feature roots have different labels")
    return {key: value[take] for key, value in candidate.items()}


def _roles(actors: np.ndarray, seed: int) -> dict[str, str]:
    ordered = sorted(
        set(actors.tolist()), key=lambda actor: hashlib.sha256(f"{seed}:{actor}".encode()).hexdigest()
    )
    if len(ordered) != 91:
        raise EvaluationError(f"expected 91 actors, found {len(ordered)}")
    roles = {actor: "task_fit" for actor in ordered[:45]}
    roles.update({actor: "router_fit" for actor in ordered[45:60]})
    roles.update({actor: "calibration" for actor in ordered[60:75]})
    roles.update({actor: "evaluation" for actor in ordered[75:]})
    return roles


def _bootstrap(
    actors: np.ndarray, values: dict[str, np.ndarray], draws: int, seed: int
) -> dict[str, dict[str, float]]:
    unique = np.unique(actors)
    if len(unique) < 2 or draws <= 0:
        raise EvaluationError("actor bootstrap needs at least two actors and positive draws")
    positions = {actor: np.flatnonzero(actors == actor) for actor in unique}
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique), size=(draws, len(unique)))
    counts = np.asarray([len(positions[actor]) for actor in unique], dtype=float)
    sampled_counts = counts[sampled].sum(axis=1)
    samples: dict[str, np.ndarray] = {}
    for name, vector in values.items():
        actor_sums = np.asarray(
            [float(vector[positions[actor]].sum()) for actor in unique], dtype=float
        )
        samples[name] = actor_sums[sampled].sum(axis=1) / sampled_counts
    return {
        name: {
            "estimate": float(vector.mean()),
            "bootstrap_95_low": float(np.quantile(samples[name], 0.025)),
            "bootstrap_95_high": float(np.quantile(samples[name], 0.975)),
        }
        for name, vector in values.items()
    }


def _threshold_for_budget(scores: np.ndarray, gains: np.ndarray, budget: float) -> float:
    if not 0.0 <= budget <= 1.0:
        raise EvaluationError("budget must lie in [0, 1]")
    candidates = np.r_[1.1, np.unique(scores)[::-1], -0.1]
    feasible = [(float((scores >= threshold).mean()), threshold) for threshold in candidates]
    feasible = [item for item in feasible if item[0] <= budget + 1e-12]
    if not feasible:
        raise EvaluationError("no threshold satisfies the requested budget")
    best_value = max(float(gains[scores >= threshold].sum()) for _, threshold in feasible)
    return float(next(threshold for _, threshold in feasible if float(gains[scores >= threshold].sum()) == best_value))


def _fit_combo(
    audio: np.ndarray,
    video: np.ndarray,
    labels: np.ndarray,
    actors: np.ndarray,
    roles: dict[str, str],
    budgets: Sequence[float],
    draws: int,
    seed: int,
) -> dict[str, object]:
    singleton = np.asarray([":" not in value for value in labels])
    role_array = np.asarray([roles[actor] for actor in actors])
    task = (role_array == "task_fit") & singleton
    router_fit = (role_array == "router_fit") & singleton
    calibration = (role_array == "calibration") & singleton
    evaluation = (role_array == "evaluation") & singleton
    if min(task.sum(), router_fit.sum(), calibration.sum(), evaluation.sum()) == 0:
        raise EvaluationError("an actor role has no singleton-labelled clips")
    fused = np.concatenate((audio, video), axis=1)
    start_head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)).fit(audio[task], labels[task])
    fused_head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)).fit(fused[task], labels[task])
    start_prediction = start_head.predict(audio)
    fused_prediction = fused_head.predict(fused)
    benefit = (fused_prediction == labels).astype(float) - (start_prediction == labels).astype(float)
    router_labels = benefit[router_fit] > 0.0
    if len(np.unique(router_labels)) == 1:
        score = np.full(len(audio), float(router_labels[0]), dtype=float)
        router_status = "constant_router_fit_target"
    else:
        router = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed + 1)).fit(audio[router_fit], router_labels)
        positive = list(router[-1].classes_).index(True)
        score = router.predict_proba(audio)[:, positive]
        router_status = "fitted"
    start_correct = (start_prediction[evaluation] == labels[evaluation]).astype(float)
    full_correct = (fused_prediction[evaluation] == labels[evaluation]).astype(float)
    result: dict[str, object] = {
        "feature_dimensions": {"audio": int(audio.shape[1]), "video": int(video.shape[1])},
        "router_status": router_status,
        "roles": {role: int(sum(value == role for value in roles.values())) for role in ("task_fit", "router_fit", "calibration", "evaluation")},
        "operating_points": {},
    }
    calibration_gain = benefit[calibration]
    for budget in budgets:
        threshold = _threshold_for_budget(score[calibration], calibration_gain, float(budget))
        acquire = score >= threshold
        policy_correct = np.where(acquire[evaluation], full_correct, start_correct)
        rate = float(acquire[evaluation].mean())
        random_expected = rate * full_correct + (1.0 - rate) * start_correct
        metrics = _bootstrap(
            actors[evaluation],
            {
                "start_accuracy": start_correct,
                "full_multimodal_accuracy": full_correct,
                "policy_accuracy": policy_correct,
                "policy_minus_start": policy_correct - start_correct,
                "policy_minus_random_matched_budget": policy_correct - random_expected,
                "full_multimodal_minus_start": full_correct - start_correct,
            },
            draws,
            seed + int(round(10_000 * budget)),
        )
        result["operating_points"][f"{budget:.2f}"] = {
            "calibration_budget": float(budget),
            "threshold": threshold,
            "evaluation_request_rate": rate,
            "actor_cluster_bootstrap": metrics,
        }
    return result


def evaluate(
    index_path: Path,
    first_root: Path,
    second_root: Path,
    expected_shards: int,
    seed: int,
    budgets: Sequence[float],
    draws: int,
) -> dict[str, object]:
    first = _load_feature_root(first_root, expected_shards)
    second = _align(first, _load_feature_root(second_root, expected_shards))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    actor_for = {record["pair_id"]: record["actor_id"] for record in index["records"]}
    if set(first["pair_id"].tolist()) != set(actor_for):
        raise EvaluationError("index and cached features have different pair identifiers")
    actors = np.asarray([actor_for[pair_id] for pair_id in first["pair_id"].tolist()])
    roles = _roles(actors, seed)
    return {
        "schema": "conflictbench.cremad-comprehensive-evaluation.v1",
        "seed": seed,
        "protocol": {
            "split_unit": "actor",
            "selection": "router and thresholds fit outside evaluation actors",
            "random_control": "expected matched request-rate policy",
            "bootstrap_unit": "evaluation_actor",
            "budgets": [float(budget) for budget in budgets],
        },
        "encoder_grid": {
            "first_audio_first_video": _fit_combo(first["audio"], first["video"], first["integrated_vote"], actors, roles, budgets, draws, seed),
            "first_audio_second_video": _fit_combo(first["audio"], second["video"], first["integrated_vote"], actors, roles, budgets, draws, seed),
            "second_audio_first_video": _fit_combo(second["audio"], first["video"], first["integrated_vote"], actors, roles, budgets, draws, seed),
            "second_audio_second_video": _fit_combo(second["audio"], second["video"], first["integrated_vote"], actors, roles, budgets, draws, seed),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--first-feature-root", type=Path, required=True)
    parser.add_argument("--second-feature-root", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budgets", type=float, nargs="+", default=(0.0, 0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit("immutable output exists")
    report = evaluate(args.index, args.first_feature_root, args.second_feature_root, args.expected_shards, args.seed, args.budgets, args.bootstrap_draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "seed": args.seed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
