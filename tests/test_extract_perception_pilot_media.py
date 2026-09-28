from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
import zipfile

import pytest

SCRIPT = (
    pathlib.Path(__file__).parents[1] / "scripts" / "extract_perception_pilot_media.py"
)
SPEC = importlib.util.spec_from_file_location("extract_perception_pilot_media", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_video_id_grammar_matches_authenticated_training_ids() -> None:
    for video_id in ("video_68", "video_1007", "video_10045"):
        assert MODULE._VIDEO_ID.fullmatch(video_id) is not None
        assert MODULE._VIDEO_FILENAME.fullmatch(f"{video_id}.mp4") is not None

    for unsafe in ("video_", "video_1/../2", "video_1234567", "video_-1"):
        assert MODULE._VIDEO_ID.fullmatch(unsafe) is None


def test_member_map_selects_nested_regular_files(tmp_path: pathlib.Path) -> None:
    archive_path = tmp_path / "train.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("train_videos/video_0001.mp4", b"one")
        archive.writestr("train_videos/video_0002.mp4", b"two")
        archive.writestr("train_videos/unused.mp4", b"unused")
    with zipfile.ZipFile(archive_path) as archive:
        selected = MODULE._member_map(archive, {"video_0001.mp4", "video_0002.mp4"})
    assert sorted(selected) == ["video_0001.mp4", "video_0002.mp4"]


def test_member_map_rejects_duplicate_basename(tmp_path: pathlib.Path) -> None:
    archive_path = tmp_path / "train.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("a/video_0001.mp4", b"one")
        archive.writestr("b/video_0001.mp4", b"two")
    with (
        zipfile.ZipFile(archive_path) as archive,
        pytest.raises(MODULE.ExtractionError, match="duplicated"),
    ):
        MODULE._member_map(archive, {"video_0001.mp4"})


def test_extract_one_is_write_once_and_resumable(tmp_path: pathlib.Path) -> None:
    archive_path = tmp_path / "train.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("train/video_0001.mp4", b"media-bytes")
    destination = tmp_path / "video_0001.mp4"
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo("train/video_0001.mp4")
        first = MODULE._extract_one(archive, info, destination)
        second = MODULE._extract_one(archive, info, destination)
    assert first["resumed_existing"] is False
    assert second["resumed_existing"] is True
    assert first["sha256"] == second["sha256"] == _sha256(destination)


def test_extract_one_rejects_changed_existing_file(tmp_path: pathlib.Path) -> None:
    archive_path = tmp_path / "train.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("video_0001.mp4", b"expected")
    destination = tmp_path / "video_0001.mp4"
    destination.write_bytes(b"changed!")
    with (
        zipfile.ZipFile(archive_path) as archive,
        pytest.raises(MODULE.ExtractionError, match="existing destination differs"),
    ):
        MODULE._extract_one(archive, archive.getinfo("video_0001.mp4"), destination)


def test_extract_one_compares_existing_file_with_decompressed_member_sha256(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "train.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("video_0001.mp4", b"abcd")
    destination = tmp_path / "video_0001.mp4"
    destination.write_bytes(b"efgh")
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo("video_0001.mp4")
        monkeypatch.setattr(MODULE, "_crc32_path", lambda path: info.CRC)
        with pytest.raises(
            MODULE.ExtractionError, match="existing destination differs"
        ):
            MODULE._extract_one(archive, info, destination)


def test_locked_json_rejects_digest_mismatch(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "pilot.json"
    source.write_text(json.dumps({"ok": True}), encoding="utf-8")
    with pytest.raises(MODULE.ExtractionError, match="SHA-256 differs"):
        MODULE._locked_json(source, "0" * 64, "pilot index")


def test_write_once_refuses_existing_receipt(tmp_path: pathlib.Path) -> None:
    receipt = tmp_path / "receipt.json"
    MODULE._write_once(receipt, {"status": "first"})
    with pytest.raises(MODULE.ExtractionError, match="already exists"):
        MODULE._write_once(receipt, {"status": "second"})
    assert json.loads(receipt.read_text(encoding="utf-8")) == {"status": "first"}


def test_probe_streams_requires_audio_and_video(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "video_0001.mp4"
    media.write_bytes(b"media")

    class Completed:
        returncode = 0
        stdout = '{"streams":[{"codec_type":"video"},{"codec_type":"audio"}]}'

    monkeypatch.setattr(MODULE.subprocess, "run", lambda *args, **kwargs: Completed())
    assert MODULE._probe_streams(pathlib.Path("/bin/ffprobe"), media) == [
        "audio",
        "video",
    ]


def test_probe_streams_rejects_missing_audio(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "video_0001.mp4"
    media.write_bytes(b"media")

    class Completed:
        returncode = 0
        stdout = '{"streams":[{"codec_type":"video"}]}'

    monkeypatch.setattr(MODULE.subprocess, "run", lambda *args, **kwargs: Completed())
    with pytest.raises(
        MODULE.ExtractionError, match="audio or video stream is missing"
    ):
        MODULE._probe_streams(pathlib.Path("/bin/ffprobe"), media)


def test_probe_identity_records_digest_and_version(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = tmp_path / "ffprobe"
    probe.write_bytes(b"probe executable")

    class Completed:
        returncode = 0
        stdout = "ffprobe version 7.1\nmore details\n"

    monkeypatch.setattr(MODULE.subprocess, "run", lambda *args, **kwargs: Completed())
    assert MODULE._probe_identity(probe) == {
        "filename": "ffprobe",
        "sha256": _sha256(probe),
        "version": "ffprobe version 7.1",
    }


def test_extract_rejects_archive_symlink_before_reading_other_inputs(
    tmp_path: pathlib.Path,
) -> None:
    target = tmp_path / "train.zip"
    target.write_bytes(b"not needed")
    link = tmp_path / "train-link.zip"
    link.symlink_to(target)
    with pytest.raises(MODULE.ExtractionError, match="must not be a symlink"):
        MODULE.extract_selected_media(
            archive_path=link,
            expected_archive_sha256=_sha256(target),
            pilot_path=tmp_path / "missing-pilot.json",
            expected_pilot_sha256="0" * 64,
            output_root=tmp_path / "media",
            ffprobe_path=tmp_path / "ffprobe",
        )


def _receipt_fixture(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, str, str]:
    media_root = tmp_path / "selected"
    media_root.mkdir()
    files = []
    for index in range(100):
        media = media_root / f"video_{index:04d}.mp4"
        media.write_bytes(f"media-{index}".encode())
        files.append(
            {
                "filename": media.name,
                "size_bytes": media.stat().st_size,
                "sha256": _sha256(media),
                "zip_crc32": "00000000",
                "resumed_existing": False,
                "stream_types": ["audio", "video"],
            }
        )
    ffprobe = tmp_path / "ffprobe"
    ffprobe.write_bytes(b"probe")
    ffprobe.chmod(0o755)
    monkeypatch.setattr(
        MODULE,
        "_probe_streams",
        lambda probe, media: ["audio", "video"],
    )
    monkeypatch.setattr(
        MODULE,
        "_probe_identity",
        lambda probe: {
            "filename": "ffprobe",
            "sha256": _sha256(ffprobe),
            "version": "fixture",
        },
    )
    archive_sha256 = "a" * 64
    pilot_sha256 = "b" * 64
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "conflictbench.perception-pilot-media-receipt.v1",
                "status": "selected_media_extracted",
                "input_digests": {
                    "train_archive_sha256": archive_sha256,
                    "pilot_index_sha256": pilot_sha256,
                },
                "media_root_name": "selected",
                "media_file_count": 100,
                "media_files": files,
                "stream_probe": {
                    "filename": "ffprobe",
                    "sha256": _sha256(ffprobe),
                    "version": "fixture",
                },
            }
        ),
        encoding="utf-8",
    )
    return receipt, media_root, ffprobe, archive_sha256, pilot_sha256


def test_validate_media_receipt_reauthenticates_every_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, media_root, ffprobe, archive_sha256, pilot_sha256 = _receipt_fixture(
        tmp_path, monkeypatch
    )
    result = MODULE.validate_media_receipt(
        receipt_path=receipt,
        output_root=media_root,
        expected_archive_sha256=archive_sha256,
        expected_pilot_sha256=pilot_sha256,
        ffprobe_path=ffprobe,
    )
    assert result["media_file_count"] == 100


def test_validate_media_receipt_rejects_changed_media(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, media_root, ffprobe, archive_sha256, pilot_sha256 = _receipt_fixture(
        tmp_path, monkeypatch
    )
    (media_root / "video_0042.mp4").write_bytes(b"changed")
    with pytest.raises(MODULE.ExtractionError, match="size differs|digest differs"):
        MODULE.validate_media_receipt(
            receipt_path=receipt,
            output_root=media_root,
            expected_archive_sha256=archive_sha256,
            expected_pilot_sha256=pilot_sha256,
            ffprobe_path=ffprobe,
        )


def test_validate_media_receipt_rejects_extra_media(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, media_root, ffprobe, archive_sha256, pilot_sha256 = _receipt_fixture(
        tmp_path, monkeypatch
    )
    (media_root / "extra.mp4").write_bytes(b"extra")
    with pytest.raises(MODULE.ExtractionError, match="membership differs"):
        MODULE.validate_media_receipt(
            receipt_path=receipt,
            output_root=media_root,
            expected_archive_sha256=archive_sha256,
            expected_pilot_sha256=pilot_sha256,
            ffprobe_path=ffprobe,
        )
