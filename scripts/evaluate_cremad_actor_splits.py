#!/usr/bin/env python3
"""Replicate frozen CREMA-D baseline evaluation over deterministic actor roles."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


parser = argparse.ArgumentParser()
parser.add_argument("--index", type=Path, required=True)
parser.add_argument("--feature-root", type=Path, required=True)
parser.add_argument("--expected-shards", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise SystemExit("immutable output exists")
parts = [np.load(path) for path in sorted(args.feature_root.glob("features-*-shard*.npz"))]
if len(parts) != args.expected_shards:
    raise SystemExit("incomplete feature shards")
pair_ids = np.concatenate([part["pair_id"] for part in parts])
labels = np.concatenate([part["integrated_vote"] for part in parts])
audio = np.concatenate([part["audio"] for part in parts])
video = np.concatenate([part["video"] for part in parts])
index = json.loads(args.index.read_text())
actor_for = {record["pair_id"]: record["actor_id"] for record in index["records"]}
if set(pair_ids.tolist()) != set(actor_for):
    raise SystemExit("index and feature IDs differ")
actors = np.array([actor_for[pair_id] for pair_id in pair_ids])
actor_order = sorted(set(actors.tolist()), key=lambda actor: hashlib.sha256(f"{args.seed}:{actor}".encode()).hexdigest())
if len(actor_order) != 91:
    raise SystemExit("unexpected actor count")
roles = {actor: "task_fit" for actor in actor_order[:45]}
roles.update({actor: "router_fit" for actor in actor_order[45:60]})
roles.update({actor: "calibration" for actor in actor_order[60:75]})
roles.update({actor: "evaluation" for actor in actor_order[75:]})
singleton = np.array([":" not in label for label in labels])
train = np.array([roles[actor] == "task_fit" for actor in actors]) & singleton
evaluation = np.array([roles[actor] == "evaluation" for actor in actors]) & singleton
if not train.any() or not evaluation.any():
    raise SystemExit("invalid roles")
result = {"schema": "conflictbench.cremad-actor-split-replication.v1", "seed": args.seed, "actors": {role: sum(value == role for value in roles.values()) for role in ("task_fit", "router_fit", "calibration", "evaluation")}, "singleton": {"train": int(train.sum()), "evaluation": int(evaluation.sum())}, "methods": {}}
for name, values in {"audio": audio, "video": video, "audio_video": np.concatenate((audio, video), axis=1)}.items():
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=20270918)).fit(values[train], labels[train])
    result["methods"][name] = {"accuracy": float(accuracy_score(labels[evaluation], model.predict(values[evaluation]))), "feature_dim": int(values.shape[1])}
result["fused_minus_audio"] = result["methods"]["audio_video"]["accuracy"] - result["methods"]["audio"]["accuracy"]
args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
