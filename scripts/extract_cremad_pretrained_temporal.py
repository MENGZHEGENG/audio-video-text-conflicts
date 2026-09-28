#!/usr/bin/env python3
"""Extract frozen audio and uniformly pooled video features from pinned models.

This is intentionally a representation control, not a model search: it uses
the same pretrained checkpoints as the one-frame diagnostic and changes only
the video observation from the first decoded frame to a fixed, uniformly
spaced set of frames.  Filename tokens are never read as features.
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys

import av
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoModel
from conflictbench.posconv_compat import (
    _checkpoint_tensors,
    apply_legacy_posconv,
    checkpoint_file,
    patch_tensors,
)


def uniform_indices(frame_count: int, sample_count: int) -> list[int]:
    """Return deterministic, endpoint-inclusive indices for a nonempty video."""
    if frame_count < 1:
        raise ValueError("video has no decodable frames")
    count = min(frame_count, sample_count)
    return sorted(set(np.linspace(0, frame_count - 1, count, dtype=int).tolist()))


def decode_uniform_frames(path: str, sample_count: int) -> tuple[list[np.ndarray], int]:
    """Decode and retain a fixed uniform frame sample, closing every container."""
    container = av.open(path)
    try:
        stream = container.streams.video[0]
        reported_count = int(stream.frames or 0)
        if reported_count < 1:
            all_frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
            if not all_frames:
                raise ValueError(f"no video frames in {path}")
            indices = uniform_indices(len(all_frames), sample_count)
            return [all_frames[index] for index in indices], len(all_frames)
        wanted = set(uniform_indices(reported_count, sample_count))
        selected = []
        decoded = 0
        for frame in container.decode(stream):
            if decoded in wanted:
                selected.append(frame.to_ndarray(format="rgb24"))
            decoded += 1
        if len(selected) != len(wanted):
            raise ValueError(f"decoded {len(selected)} of {len(wanted)} requested frames from {path}")
        return selected, decoded
    finally:
        container.close()


parser = argparse.ArgumentParser()
parser.add_argument("--index", type=pathlib.Path, required=True)
parser.add_argument("--shard", type=int, required=True)
parser.add_argument("--shards", type=int, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--video-frames", type=int, default=8)
parser.add_argument("--max-records", type=int, default=None)
parser.add_argument("--audio-model", default="facebook/wav2vec2-base-960h")
parser.add_argument("--video-model", default="google/vit-base-patch16-224")
parser.add_argument("--audio-revision", default="22aad52d435eb6dbaf354bdad9b0da84ce7d6156")
parser.add_argument("--video-revision", default="3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3")
parser.add_argument("--audio-posconv-patch", type=pathlib.Path)
args = parser.parse_args()
if args.output.exists():
    raise SystemExit("immutable output exists")
if args.shards < 1 or not 0 <= args.shard < args.shards or args.video_frames < 1:
    raise SystemExit("invalid shard or frame count")

records = json.loads(args.index.read_text())["records"][args.shard :: args.shards]
if args.max_records is not None:
    records = records[: args.max_records]
if not records:
    raise SystemExit("empty shard")
audio_extractor = AutoFeatureExtractor.from_pretrained(args.audio_model, revision=args.audio_revision)
audio_model = AutoModel.from_pretrained(args.audio_model, revision=args.audio_revision).cuda().eval()
if args.audio_posconv_patch is not None:
    posconv_g, posconv_v, expected_posconv_digest = patch_tensors(
        args.audio_posconv_patch, args.audio_model, args.audio_revision)
else:
    # Existing callers remain valid, but they still restore the legacy weights.
    # The staged array passes a prepared patch to avoid re-reading a large
    # checkpoint in every shard.
    from huggingface_hub import snapshot_download

    snapshot = pathlib.Path(snapshot_download(args.audio_model, revision=args.audio_revision,
                                              local_files_only=True))
    posconv_g, posconv_v = _checkpoint_tensors(checkpoint_file(snapshot))
    expected_posconv_digest = None
audio_posconv_digest = apply_legacy_posconv(audio_model, posconv_g, posconv_v)
if expected_posconv_digest is not None and audio_posconv_digest != expected_posconv_digest:
    raise SystemExit("restored positional-convolution digest mismatch")
video_processor = AutoImageProcessor.from_pretrained(args.video_model, revision=args.video_revision)
video_model = AutoModel.from_pretrained(args.video_model, revision=args.video_revision).cuda().eval()
ffmpeg = shutil.which("ffmpeg") or str(pathlib.Path(sys.executable).parent / "ffmpeg")
if not pathlib.Path(ffmpeg).is_file():
    raise SystemExit("ffmpeg executable is required on PATH or beside Python")

audio, video, pair_ids, roles, votes, frame_counts = [], [], [], [], [], []
with torch.no_grad():
    for record in records:
        raw = subprocess.run(
            [ffmpeg, "-v", "error", "-i", record["audio_path"], "-ar", "16000", "-ac", "1", "-f", "f32le", "-acodec", "pcm_f32le", "-"],
            capture_output=True,
            check=True,
        ).stdout
        waveform = np.frombuffer(raw, dtype=np.float32)
        frames, decoded_count = decode_uniform_frames(record["video_path"], args.video_frames)
        audio_inputs = {key: value.cuda() for key, value in audio_extractor(waveform, sampling_rate=16000, return_tensors="pt").items()}
        video_inputs = {key: value.cuda() for key, value in video_processor(images=frames, return_tensors="pt").items()}
        audio.append(audio_model(**audio_inputs).last_hidden_state.mean(1).cpu().numpy()[0])
        frame_embeddings = video_model(**video_inputs).last_hidden_state[:, 0].cpu().numpy()
        video.append(frame_embeddings.mean(axis=0))
        pair_ids.append(record["pair_id"])
        roles.append(record["role"])
        votes.append(":".join(record["integrated_vote"]))
        frame_counts.append(decoded_count)

args.output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    args.output,
    pair_id=np.array(pair_ids),
    role=np.array(roles),
    integrated_vote=np.array(votes),
    audio=np.stack(audio),
    video=np.stack(video),
    decoded_video_frames=np.array(frame_counts, dtype=np.int32),
    video_sample_count=np.array(args.video_frames, dtype=np.int32),
    audio_model=np.array(args.audio_model),
    video_model=np.array(args.video_model),
    audio_revision=np.array(args.audio_revision),
    audio_posconv_sha256=np.array(audio_posconv_digest),
    video_revision=np.array(args.video_revision),
)
print(json.dumps({"audio_model": args.audio_model, "audio_dim": int(audio[-1].shape[0]), "decoded_frame_max": max(frame_counts), "decoded_frame_min": min(frame_counts), "records": len(pair_ids), "shard": args.shard, "video_frames": args.video_frames, "video_model": args.video_model, "video_dim": int(video[-1].shape[0])}, sort_keys=True))
