#!/usr/bin/env python3
"""Actor-disjoint audio-to-video acquisition policy on frozen CREMA-D features."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


parser = argparse.ArgumentParser()
parser.add_argument("--feature-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--expected-shards", type=int, default=12)
args = parser.parse_args()
if args.output.exists():
    raise SystemExit("immutable output exists")
parts = [np.load(path) for path in sorted(args.feature_root.glob("features-*-shard*.npz"))]
if len(parts) != args.expected_shards:
    raise SystemExit("incomplete feature shards")
roles = np.concatenate([part["role"] for part in parts])
labels = np.concatenate([part["integrated_vote"] for part in parts])
audio_features = np.concatenate([part["audio"] for part in parts])
video_features = np.concatenate([part["video"] for part in parts])
keep = np.array([":" not in label for label in labels])
features = np.concatenate((audio_features, video_features), axis=1)
task_fit = (roles == "task_fit") & keep
router_fit = (roles == "router_fit") & keep
calibration = (roles == "calibration") & keep
evaluation = (roles == "evaluation") & keep
audio_head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=20270918)).fit(audio_features[task_fit], labels[task_fit])
fused_head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=20270918)).fit(features[task_fit], labels[task_fit])
audio_prediction = audio_head.predict(audio_features)
fused_prediction = fused_head.predict(features)
benefit = (fused_prediction == labels).astype(int) - (audio_prediction == labels).astype(int)
router = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=20270919)).fit(audio_features[router_fit], benefit[router_fit] > 0)
positive_index = list(router[-1].classes_).index(True)
score = router.predict_proba(audio_features)[:, positive_index]
threshold = max(np.unique(score[calibration]), key=lambda value: accuracy_score(labels[calibration], np.where(score[calibration] >= value, fused_prediction[calibration], audio_prediction[calibration])))
acquire = score[evaluation] >= threshold
policy_prediction = np.where(acquire, fused_prediction[evaluation], audio_prediction[evaluation])
result = {"schema": "conflictbench.cremad-audio-video-acquisition.v1", "threshold": float(threshold), "roles": {"task_fit": int(task_fit.sum()), "router_fit": int(router_fit.sum()), "calibration": int(calibration.sum()), "evaluation": int(evaluation.sum())}, "evaluation": {"audio_accuracy": float(accuracy_score(labels[evaluation], audio_prediction[evaluation])), "fused_accuracy": float(accuracy_score(labels[evaluation], fused_prediction[evaluation])), "policy_accuracy": float(accuracy_score(labels[evaluation], policy_prediction)), "video_acquisition_rate": float(acquire.mean())}}
result["evaluation"]["policy_minus_audio"] = result["evaluation"]["policy_accuracy"] - result["evaluation"]["audio_accuracy"]
args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
