from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from conflictbench.posconv_compat import apply_legacy_posconv, patch_tensors, prepare_patch


def test_legacy_checkpoint_weights_apply_to_parametrized_values(tmp_path: Path) -> None:
    conv = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(4, 4, 3))
    originals = conv.parametrizations.weight
    g = torch.full_like(originals.original0, 2.0)
    v = torch.arange(originals.original1.numel(), dtype=torch.float32).reshape_as(originals.original1) + 1
    checkpoint = tmp_path / "model.safetensors"
    safetensors_torch.save_file({
        "wav2vec2.encoder.pos_conv_embed.conv.weight_g": g,
        "wav2vec2.encoder.pos_conv_embed.conv.weight_v": v,
    }, checkpoint)
    patch = tmp_path / "patch.npz"
    digest = prepare_patch(checkpoint, "test/model", "revision123", patch)
    loaded_g, loaded_v, loaded_digest = patch_tensors(patch, "test/model", "revision123")
    model = SimpleNamespace(encoder=SimpleNamespace(pos_conv_embed=SimpleNamespace(conv=conv)))
    assert apply_legacy_posconv(model, loaded_g, loaded_v) == digest == loaded_digest
    torch.testing.assert_close(originals.original0, g)
    torch.testing.assert_close(originals.original1, v)
    with pytest.raises(ValueError, match="identity"):
        patch_tensors(patch, "wrong/model", "revision123")
    with pytest.raises(ValueError, match="shapes"):
        apply_legacy_posconv(model, np.zeros((1, 1)), loaded_v)


def test_legacy_pytorch_bin_checkpoint_is_supported(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pytorch_model.bin"
    torch.save({
        "hubert.encoder.pos_conv_embed.conv.weight_g": torch.ones(2, 1, 1),
        "hubert.encoder.pos_conv_embed.conv.weight_v": torch.ones(2, 2, 3),
    }, checkpoint)
    patch = tmp_path / "patch.npz"
    prepare_patch(checkpoint, "test/hubert", "revision456", patch)
    g, v, _ = patch_tensors(patch, "test/hubert", "revision456")
    assert g.shape == (2, 1, 1)
    assert v.shape == (2, 2, 3)
