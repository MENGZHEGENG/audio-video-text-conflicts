"""Verify and apply pinned speech positional-convolution weights.

The pinned wav2vec 2.0 and HuBERT checkpoints use ``weight_g``/``weight_v``.
PyTorch 2.5 parametrized weight normalization expects ``original0`` and
``original1``. Transformers 4.44 emits missing-weight warnings for the legacy
names, but the loaded parameters match these checkpoints exactly in the
audited runtime. This module copies the pinned tensors explicitly so each
feature shard can record and verify their digest. The copy is redundant for
the audited model/runtime pair; the warning alone does not imply random
weights or corrupted features.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _digest(g: np.ndarray, v: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name, value in (("weight_g", g), ("weight_v", v)):
        contiguous = np.ascontiguousarray(value)
        digest.update(json.dumps({"name": name, "dtype": str(contiguous.dtype),
                                  "shape": list(contiguous.shape)}, sort_keys=True).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _checkpoint_tensors(checkpoint: Path) -> tuple[np.ndarray, np.ndarray]:
    import torch

    if checkpoint.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            matches = {
                suffix: [key for key in keys if key.endswith("encoder.pos_conv_embed.conv." + suffix)]
                for suffix in ("weight_g", "weight_v")
            }
            if any(len(found) != 1 for found in matches.values()):
                raise ValueError("checkpoint lacks one legacy positional-convolution weight pair")
            g = handle.get_tensor(matches["weight_g"][0]).numpy()
            v = handle.get_tensor(matches["weight_v"][0]).numpy()
    elif checkpoint.suffix == ".bin":
        state: Any = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError("checkpoint state dict is missing")
        matches = {
            suffix: [key for key in state if key.endswith("encoder.pos_conv_embed.conv." + suffix)]
            for suffix in ("weight_g", "weight_v")
        }
        if any(len(found) != 1 for found in matches.values()):
            raise ValueError("checkpoint lacks one legacy positional-convolution weight pair")
        g = state[matches["weight_g"][0]].detach().cpu().numpy()
        v = state[matches["weight_v"][0]].detach().cpu().numpy()
    else:
        raise ValueError(f"unsupported checkpoint format: {checkpoint}")
    return np.ascontiguousarray(g), np.ascontiguousarray(v)


def checkpoint_file(snapshot: Path) -> Path:
    for filename in ("model.safetensors", "pytorch_model.bin"):
        candidate = snapshot / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no model checkpoint in {snapshot}")


def prepare_patch(checkpoint: Path, model_id: str, revision: str, output: Path) -> str:
    if output.exists():
        raise FileExistsError(output)
    g, v = _checkpoint_tensors(checkpoint)
    digest = _digest(g, v)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.part")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, weight_g=g, weight_v=v,
                                model_id=np.array(model_id), revision=np.array(revision),
                                tensor_sha256=np.array(digest))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def patch_tensors(path: Path, model_id: str, revision: str) -> tuple[np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["model_id"]) != model_id or str(saved["revision"]) != revision:
            raise ValueError("positional-convolution patch model identity mismatch")
        g = saved["weight_g"].copy()
        v = saved["weight_v"].copy()
        digest = str(saved["tensor_sha256"])
    if _digest(g, v) != digest:
        raise ValueError("positional-convolution patch digest mismatch")
    return g, v, digest


def apply_legacy_posconv(model: Any, g: np.ndarray, v: np.ndarray) -> str:
    import torch

    conv = model.encoder.pos_conv_embed.conv
    parameterization = conv.parametrizations.weight
    first, second = parameterization.original0, parameterization.original1
    if tuple(first.shape) != tuple(g.shape) or tuple(second.shape) != tuple(v.shape):
        raise ValueError("positional-convolution weight shapes do not match this runtime")
    with torch.no_grad():
        first.copy_(torch.from_numpy(g).to(device=first.device, dtype=first.dtype))
        second.copy_(torch.from_numpy(v).to(device=second.device, dtype=second.dtype))
    if not torch.isfinite(conv.weight).all():
        raise ValueError("restored positional-convolution weights are not finite")
    return _digest(g, v)
