"""Source-independent tests for the CREMA-D LFS downloader."""

import hashlib
import io
import json
from pathlib import Path
from email.message import Message
from urllib.error import HTTPError

import pytest

from conflictbench.cremad_lfs_download import DownloadTask
from conflictbench.cremad_lfs_download import _batch_actions
from conflictbench.cremad_lfs_download import _download_one
from conflictbench.cremad_lfs_download import _unique_objects
from conflictbench.cremad_lfs_download import tasks_from_pointer_index


def _index(path: Path) -> Path:
    content = b"verified media"
    record = {
        "schema": "conflictbench.cremad-pointer-index.v1",
        "records": [{
            "clip_id": "1001_IEO_NEU_XX",
            "audio": {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)},
            "video": {"sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)},
        }],
    }
    output = path / "index.json"
    output.write_text(json.dumps(record), encoding="utf-8")
    return output


def test_tasks_from_pointer_index_uses_only_expected_paths(tmp_path: Path) -> None:
    tasks = tasks_from_pointer_index(tmp_path, _index(tmp_path))
    assert [task.path.name for task in tasks] == ["1001_IEO_NEU_XX.wav", "1001_IEO_NEU_XX.flv"]
    assert len(_unique_objects(tasks)) == 1


def test_download_one_authenticates_before_replacing_pointer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    content = b"verified media"
    destination = tmp_path / "AudioWAV" / "1001_IEO_NEU_XX.wav"
    destination.parent.mkdir()
    destination.write_text("pointer", encoding="utf-8")
    task = DownloadTask(destination, hashlib.sha256(content).hexdigest(), len(content))
    monkeypatch.setattr("conflictbench.cremad_lfs_download.urlopen", lambda *_args, **_kwargs: io.BytesIO(content))
    _download_one(task, "https://example.test/media", {})
    assert destination.read_bytes() == content


def test_download_one_removes_partial_on_digest_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "AudioWAV" / "1001_IEO_NEU_XX.wav"
    destination.parent.mkdir()
    task = DownloadTask(destination, "0" * 64, len(b"wrong"))
    monkeypatch.setattr("conflictbench.cremad_lfs_download.urlopen", lambda *_args, **_kwargs: io.BytesIO(b"wrong"))
    with pytest.raises(ValueError, match="downloaded media mismatch"):
        _download_one(task, "https://example.test/media", {})
    assert not destination.with_name(f"{destination.name}.admission.partial").exists()


def test_batch_actions_honors_retry_after_on_rate_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    task = DownloadTask(tmp_path / "media.wav", "a" * 64, 1)
    headers = Message()
    headers["Retry-After"] = "2"
    rate_limited = HTTPError("https://example.test", 429, "rate", headers, None)
    response = io.BytesIO(json.dumps({"objects": [{"oid": task.oid, "actions": {"download": {"href": "https://example.test/media"}}}]}).encode())
    replies = iter([rate_limited, response])
    def fake_urlopen(*_args: object, **_kwargs: object) -> io.BytesIO:
        reply = next(replies)
        if isinstance(reply, HTTPError):
            raise reply
        return reply

    monkeypatch.setattr("conflictbench.cremad_lfs_download.urlopen", fake_urlopen)
    delays: list[int] = []
    monkeypatch.setattr("conflictbench.cremad_lfs_download.time.sleep", delays.append)
    assert _batch_actions([task])[task.oid][0] == "https://example.test/media"
    assert delays == [2]
