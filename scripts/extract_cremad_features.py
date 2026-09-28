#!/usr/bin/env python3
"""Extract label-blind audio/video descriptors from an admitted CREMA-D index."""
from __future__ import annotations

import argparse
import av
import json
import numpy as np
from pathlib import Path
import subprocess
import sys
import torch


def audio_features(path: str) -> np.ndarray:
    ffmpeg = str(Path(sys.executable).parent / "ffmpeg")
    raw = subprocess.run([ffmpeg, "-v", "error", "-i", path, "-f", "f32le", "-acodec", "pcm_f32le", "-"], capture_output=True, check=True).stdout
    values = torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy()).cuda()
    return torch.stack((values.mean(), values.std(), values.abs().mean(), values.square().mean().sqrt())).cpu().numpy()


def video_features(path: str) -> np.ndarray:
    container = av.open(path); stream = container.streams.video[0]; frames=[]
    for frame in container.decode(stream):
        frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) == 32: break
    container.close()
    if not frames: raise ValueError("no video frames")
    values = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0).cuda()
    return torch.cat((values.mean((0,1,2)), values.std((0,1,2)))).cpu().numpy()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--index", type=Path, required=True); parser.add_argument("--shard", type=int, required=True); parser.add_argument("--shards", type=int, required=True); parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists() or not 0 <= args.shard < args.shards: raise SystemExit("invalid immutable shard output")
index=json.loads(args.index.read_text()); records=index["records"]
selected=[record for pos,record in enumerate(records) if pos % args.shards == args.shard]
audio=[]; video=[]; ids=[]; roles=[]; integrated=[]
for record in selected:
    audio.append(audio_features(record["audio_path"])); video.append(video_features(record["video_path"])); ids.append(record["pair_id"]); roles.append(record["role"]); integrated.append(record["integrated_vote"])
args.output.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(args.output, pair_id=np.array(ids), role=np.array(roles), integrated_vote=np.array([":".join(value) for value in integrated]), audio=np.stack(audio), video=np.stack(video))
print(json.dumps({"shard":args.shard,"records":len(ids),"audio_dim":4,"video_dim":6},sort_keys=True))
