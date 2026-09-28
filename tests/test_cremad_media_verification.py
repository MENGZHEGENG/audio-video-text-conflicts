"""Tests for byte-level CREMA-D media verification."""

import hashlib
import json
from pathlib import Path

import pytest

from conflictbench.cremad_media_verification import verify_media


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_fixture(root: Path) -> Path:
    (root / "AudioWAV").mkdir()
    (root / "VideoFlash").mkdir()
    audio = b"audio bytes"
    video = b"video bytes"
    (root / "AudioWAV" / "1001_IEO_NEU_XX.wav").write_bytes(audio)
    (root / "VideoFlash" / "1001_IEO_NEU_XX.flv").write_bytes(video)
    index = {
        "schema": "conflictbench.cremad-pointer-index.v1",
        "repository": "https://example.test/CREMA-D",
        "commit": "a" * 40,
        "record_index_sha256": "b" * 64,
        "records": [
            {
                "clip_id": "1001_IEO_NEU_XX",
                "audio": {"sha256": _sha256(audio), "size_bytes": len(audio)},
                "video": {"sha256": _sha256(video), "size_bytes": len(video)},
            }
        ],
    }
    path = root / "pointer-index.json"
    path.write_text(json.dumps(index), encoding="utf-8")
    return path


def test_verify_media_accepts_matching_bytes(tmp_path: Path) -> None:
    index = _write_fixture(tmp_path)
    result = verify_media(tmp_path, index)
    assert result["clip_count"] == 1
    assert result["media_count"] == 2
    assert result["total_bytes"] == len(b"audio bytesvideo bytes")


def test_verify_media_rejects_changed_byte(tmp_path: Path) -> None:
    index = _write_fixture(tmp_path)
    (tmp_path / "AudioWAV" / "1001_IEO_NEU_XX.wav").write_bytes(b"wrong bytes")
    with pytest.raises(ValueError, match="media bytes"):
        verify_media(tmp_path, index)
