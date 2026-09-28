#!/usr/bin/env python3
"""Run a bounded multimodal compatibility probe for a pinned frozen model."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import pathlib
import tempfile
from typing import Any

MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
MODEL_REVISION = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
RESULT_SCHEMA = "conflictbench.perception-omni-probe.v1"


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _base_record(cache_dir: pathlib.Path, dry_run: bool) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "status": "dry_run" if dry_run else "pending",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": "Qwen Research License Agreement; non-commercial research use",
        },
        "cache_directory": str(cache_dir.resolve()),
        "probe": {
            "audio_sample_rate_hz": 16000,
            "audio_duration_seconds": 2,
            "image_size_pixels": [224, 224],
            "return_audio": False,
            "maximum_new_tokens": 16,
        },
        "source_sha256": _sha256(pathlib.Path(__file__).resolve()),
    }


def run_probe(cache_dir: pathlib.Path) -> dict[str, Any]:
    """Load the pinned model and execute one deterministic image-audio prompt."""

    import numpy as np
    import torch
    from PIL import Image
    from transformers import (
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniProcessor,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("the compatibility probe requires a visible CUDA device")

    torch.manual_seed(0)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(cache_dir),
        dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.disable_talker()
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(cache_dir),
        use_fast=False,
    )

    sample_rate = 16000
    seconds = 2
    samples = np.arange(sample_rate * seconds, dtype=np.float32)
    audio = (0.05 * np.sin(2.0 * np.pi * 440.0 * samples / sample_rate)).astype(
        np.float32
    )
    image_values = np.zeros((224, 224, 3), dtype=np.uint8)
    image_values[..., 2] = 192
    image = Image.fromarray(image_values, mode="RGB")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "probe-image"},
                {"type": "audio", "audio": "probe-audio"},
                {
                    "type": "text",
                    "text": (
                        "This is a compatibility check. Reply with exactly one of "
                        "these labels: A, B, or C. Choose A."
                    ),
                },
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(
        text=prompt,
        images=[image],
        audio=[audio],
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(model.device).to(model.dtype)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=16,
            return_audio=False,
        )
    response = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return {
        "status": "pass",
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runtime": {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "qwen_omni_utils": importlib.metadata.version("qwen-omni-utils"),
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "model_dtype": str(model.dtype),
        },
        "response": response,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    record = _base_record(args.cache_dir, args.dry_run)
    if not args.dry_run:
        record.update(run_probe(args.cache_dir))
    _atomic_json(args.output, record)


if __name__ == "__main__":
    main()
