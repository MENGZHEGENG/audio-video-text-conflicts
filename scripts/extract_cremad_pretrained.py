#!/usr/bin/env python3
import argparse, av, json, pathlib, subprocess, sys
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoModel, ViTModel

p = argparse.ArgumentParser()
p.add_argument("--index", type=pathlib.Path, required=True)
p.add_argument("--shard", type=int, required=True)
p.add_argument("--shards", type=int, required=True)
p.add_argument("--output", type=pathlib.Path, required=True)
a = p.parse_args()
if a.output.exists(): raise SystemExit("immutable output")
records = json.loads(a.index.read_text())["records"][a.shard::a.shards]
ae = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
am = AutoModel.from_pretrained("facebook/wav2vec2-base-960h").cuda().eval()
vi = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
vm = ViTModel.from_pretrained("google/vit-base-patch16-224").cuda().eval()
ffmpeg = str(pathlib.Path(sys.executable).parent / "ffmpeg")
audio=[]; video=[]; ids=[]; roles=[]; labels=[]
with torch.no_grad():
    for r in records:
        raw=subprocess.run([ffmpeg,"-v","error","-i",r["audio_path"],"-f","f32le","-acodec","pcm_f32le","-"],capture_output=True,check=True).stdout
        waveform=np.frombuffer(raw,dtype=np.float32)
        c=av.open(r["video_path"]); frame=next(c.decode(c.streams.video[0])).to_ndarray(format="rgb24"); c.close()
        audio.append(am(**{k:v.cuda() for k,v in ae(waveform,sampling_rate=16000,return_tensors="pt").items()}).last_hidden_state.mean(1).cpu().numpy()[0])
        video.append(vm(**{k:v.cuda() for k,v in vi(images=frame,return_tensors="pt").items()}).last_hidden_state[:,0].cpu().numpy()[0])
        ids.append(r["pair_id"]); roles.append(r["role"]); labels.append(":".join(r["integrated_vote"]))
a.output.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(a.output,pair_id=np.array(ids),role=np.array(roles),integrated_vote=np.array(labels),audio=np.stack(audio),video=np.stack(video))
print(json.dumps({"shard":a.shard,"records":len(ids),"dim":768},sort_keys=True))
