#!/usr/bin/env python3
"""Evaluate a fixed CREMA-D acquisition router with actor-level uncertainty.

The router is fit only on router-fit actors and its threshold is selected only
on calibration actors.  Evaluation joins actor IDs solely for clustered
resampling; clip identifiers never enter a learned feature or decision rule.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def accuracy(prediction: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(prediction == labels))


def set_accuracy(prediction: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean([predicted in label.split(":") for predicted, label in zip(prediction, labels)]))


def selected_accuracy(labels: np.ndarray, audio_prediction: np.ndarray, fused_prediction: np.ndarray, score: np.ndarray, threshold: float) -> float:
    return accuracy(np.where(score >= threshold, fused_prediction, audio_prediction), labels)


def actor_bootstrap(actor_ids: np.ndarray, values: dict[str, np.ndarray], draws: int, seed: int) -> dict[str, dict[str, float]]:
    unique = np.unique(actor_ids)
    if len(unique) < 2:
        raise ValueError("need at least two evaluation actors")
    positions = {actor: np.flatnonzero(actor_ids == actor) for actor in unique}
    rng = np.random.default_rng(seed)
    result = {}
    for name, per_clip in values.items():
        samples = np.empty(draws, dtype=float)
        for draw in range(draws):
            sampled_actors = rng.choice(unique, size=len(unique), replace=True)
            indices = np.concatenate([positions[actor] for actor in sampled_actors])
            samples[draw] = float(per_clip[indices].mean())
        result[name] = {
            "estimate": float(per_clip.mean()),
            "bootstrap_95_low": float(np.quantile(samples, 0.025)),
            "bootstrap_95_high": float(np.quantile(samples, 0.975)),
        }
    return result


parser = argparse.ArgumentParser()
parser.add_argument("--index", type=Path, required=True)
parser.add_argument("--feature-root", type=Path, required=True)
parser.add_argument("--expected-shards", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--bootstrap-draws", type=int, default=10000)
parser.add_argument("--seed", type=int, default=20270918)
parser.add_argument("--start-modality", choices=("audio", "video"), default="audio")
parser.add_argument("--predictions-output", type=Path, default=None)
args = parser.parse_args()
if args.output.exists():
    raise SystemExit("immutable output exists")
if args.predictions_output is not None and args.predictions_output.exists():
    raise SystemExit("immutable predictions output exists")
parts = [np.load(path) for path in sorted(args.feature_root.glob("features-*-shard*.npz"))]
if len(parts) != args.expected_shards:
    raise SystemExit("incomplete feature shards")
pair_ids = np.concatenate([part["pair_id"] for part in parts])
roles = np.concatenate([part["role"] for part in parts])
labels = np.concatenate([part["integrated_vote"] for part in parts])
audio = np.concatenate([part["audio"] for part in parts])
video = np.concatenate([part["video"] for part in parts])
if len(set(pair_ids.tolist())) != len(pair_ids):
    raise SystemExit("duplicate pair IDs")
index = json.loads(args.index.read_text())
metadata = {record["pair_id"]: record for record in index["records"]}
if set(pair_ids.tolist()) != set(metadata):
    raise SystemExit("index and feature IDs differ")
actors = np.array([metadata[pair_id]["actor_id"] for pair_id in pair_ids])
voice = np.array([":".join(metadata[pair_id]["voice_vote"]) for pair_id in pair_ids])
face = np.array([":".join(metadata[pair_id]["face_vote"]) for pair_id in pair_ids])
singleton = np.array([":" not in value for value in labels])
task = (roles == "task_fit") & singleton
router_fit = (roles == "router_fit") & singleton
calibration = (roles == "calibration") & singleton
evaluation_singleton = (roles == "evaluation") & singleton
evaluation_all = roles == "evaluation"
features = np.concatenate((audio, video), axis=1)
start_features = audio if args.start_modality == "audio" else video
start_head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=20270918)).fit(start_features[task], labels[task])
fused_head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=20270918)).fit(features[task], labels[task])
start_prediction = start_head.predict(start_features)
fused_prediction = fused_head.predict(features)
benefit = (fused_prediction == labels).astype(int) - (start_prediction == labels).astype(int)
router = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=20270919)).fit(start_features[router_fit], benefit[router_fit] > 0)
positive = list(router[-1].classes_).index(True)
score = router.predict_proba(start_features)[:, positive]
threshold = max(np.unique(score[calibration]), key=lambda value: selected_accuracy(labels[calibration], start_prediction[calibration], fused_prediction[calibration], score[calibration], value))
acquire = score >= threshold
policy_prediction = np.where(acquire, fused_prediction, start_prediction)
mask = evaluation_singleton
request_rate = float(acquire[mask].mean())
start_correct = (start_prediction[mask] == labels[mask]).astype(float)
fused_correct = (fused_prediction[mask] == labels[mask]).astype(float)
policy_correct = (policy_prediction[mask] == labels[mask]).astype(float)
random_expected = request_rate * fused_correct + (1.0 - request_rate) * start_correct
uncertainty = actor_bootstrap(
    actors[mask],
    {
        "start_accuracy": start_correct,
        "full_multimodal_accuracy": fused_correct,
        "policy_accuracy": policy_correct,
        "policy_minus_start": policy_correct - start_correct,
        "policy_minus_random_matched_budget": policy_correct - random_expected,
        "full_multimodal_minus_start": fused_correct - start_correct,
    },
    args.bootstrap_draws,
    args.seed,
)
all_mask = evaluation_all
discordant = voice[all_mask] != face[all_mask]
def partition_metrics(partition: np.ndarray) -> dict[str, float | int]:
    if not partition.any():
        return {"count": 0}
    return {
        "count": int(partition.sum()),
        "start_set_accuracy": set_accuracy(start_prediction[all_mask][partition], labels[all_mask][partition]),
        "full_multimodal_set_accuracy": set_accuracy(fused_prediction[all_mask][partition], labels[all_mask][partition]),
        "policy_set_accuracy": set_accuracy(policy_prediction[all_mask][partition], labels[all_mask][partition]),
        "request_rate": float(acquire[all_mask][partition].mean()),
    }
result = {
    "schema": "conflictbench.cremad-temporal-policy-evaluation.v1",
    "start_modality": args.start_modality,
    "roles": {role: int((roles == role).sum()) for role in ("task_fit", "router_fit", "calibration", "evaluation")},
    "threshold": float(threshold),
    "singleton_evaluation_request_rate": request_rate,
    "actor_cluster_bootstrap": {"unit": "actor", "actors": int(len(np.unique(actors[mask]))), "draws": args.bootstrap_draws, "seed": args.seed, "metrics": uncertainty},
    "tied_vote_sensitivity": {"all_evaluation": partition_metrics(np.ones(all_mask.sum(), dtype=bool)), "voice_face_agree": partition_metrics(~discordant), "voice_face_disagree": partition_metrics(discordant)},
}
if args.predictions_output is not None:
    prediction_records = [
        {
            "pair_id": str(pair_ids[index]),
            "actor_id": str(actors[index]),
            "label": str(labels[index]),
            "start_prediction": str(start_prediction[index]),
            "fused_prediction": str(fused_prediction[index]),
            "policy_prediction": str(policy_prediction[index]),
            "query": bool(acquire[index]),
            "query_score": float(score[index]),
        }
        for index in np.flatnonzero(mask)
    ]
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.write_text(
        json.dumps({"schema": "conflictbench.cremad-evaluation-predictions.v1",
                    "start_modality": args.start_modality,
                    "records": prediction_records}, indent=2, sort_keys=True) + "\n"
    )
args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
