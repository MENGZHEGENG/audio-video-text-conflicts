#!/usr/bin/env python3
"""Extract only hash-locked Perception Test pilot videos from the train archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import stat
import subprocess
import tempfile
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from typing import Any

from conflictbench.perception_media_pilot import validate_media_pilot_attestation

_VIDEO_ID = re.compile(r"video_[0-9]{1,6}")
_VIDEO_FILENAME = re.compile(r"video_[0-9]{1,6}\.mp4")
_PAIR_ROLES = {"same_answer_nuisance", "opposite_answer_candidate"}


class ExtractionError(ValueError):
    """Raised when a locked input or archive member is invalid."""


def _sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _locked_json(path: pathlib.Path, expected_sha256: str, label: str) -> Any:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ExtractionError(f"{label} expected SHA-256 is invalid")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ExtractionError(f"{label} cannot be read") from exc
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ExtractionError(f"{label} SHA-256 differs")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"{label} is not valid JSON") from exc


def _pilot_video_ids(pilot: Any) -> list[str]:
    if not isinstance(pilot, Mapping):
        raise ExtractionError("pilot index must be a JSON object")
    try:
        validate_media_pilot_attestation(pilot)
    except (TypeError, ValueError) as exc:
        raise ExtractionError("pilot attestation is invalid") from exc
    pairs = pilot.get("candidate_index")
    if not isinstance(pairs, list) or len(pairs) != 200:
        raise ExtractionError("pilot index must contain exactly 200 pairs")
    pair_ids: set[str] = set()
    video_ids: set[str] = set()
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ExtractionError("pilot pair must be an object")
        pair_id = pair.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id or pair_id in pair_ids:
            raise ExtractionError("pilot pair IDs must be unique strings")
        pair_ids.add(pair_id)
        if pair.get("official_split") != "train" or pair.get("role") not in _PAIR_ROLES:
            raise ExtractionError("pilot pair split or role is invalid")
        for source_role in ("target", "donor"):
            source = pair.get(source_role)
            if not isinstance(source, Mapping):
                raise ExtractionError(f"pilot {source_role} must be an object")
            video_id = source.get("video_id")
            if not isinstance(video_id, str) or _VIDEO_ID.fullmatch(video_id) is None:
                raise ExtractionError("pilot video ID is invalid")
            video_ids.add(video_id)
    if len(video_ids) < 100 or len(video_ids) > 300:
        raise ExtractionError("pilot unique-video count is outside the locked bounds")
    return sorted(video_ids)


def _member_map(
    archive: zipfile.ZipFile, wanted: set[str]
) -> dict[str, zipfile.ZipInfo]:
    selected: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        raw_name = info.filename
        pure = pathlib.PurePosixPath(raw_name)
        if (
            not raw_name
            or "\\" in raw_name
            or pure.is_absolute()
            or ".." in pure.parts
            or info.is_dir()
        ):
            continue
        basename = pure.name
        if basename not in wanted:
            continue
        mode = info.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        if info.flag_bits & 0x1 or file_type not in (0, stat.S_IFREG):
            raise ExtractionError(
                f"selected member is encrypted or non-regular: {raw_name}"
            )
        if info.file_size <= 0 or info.file_size > 2 * 1024 * 1024 * 1024:
            raise ExtractionError(f"selected member size is invalid: {raw_name}")
        if basename in selected:
            raise ExtractionError(f"selected member basename is duplicated: {basename}")
        selected[basename] = info
    missing = sorted(wanted - set(selected))
    if missing:
        raise ExtractionError(f"archive lacks {len(missing)} selected videos")
    return selected


def _crc32_path(path: pathlib.Path) -> int:
    checksum = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


def _member_sha256(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    with archive.open(info, "r") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_one(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: pathlib.Path,
) -> dict[str, Any]:
    if destination.is_symlink():
        raise ExtractionError(f"destination must not be a symlink: {destination.name}")
    if destination.exists():
        file_stat = destination.stat()
        destination_sha256 = _sha256_path(destination)
        member_sha256 = _member_sha256(archive, info)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_size != info.file_size
            or _crc32_path(destination) != info.CRC
            or destination_sha256 != member_sha256
        ):
            raise ExtractionError(f"existing destination differs: {destination.name}")
        destination.chmod(0o440)
        return {
            "filename": destination.name,
            "size_bytes": file_stat.st_size,
            "sha256": destination_sha256,
            "zip_crc32": f"{info.CRC:08x}",
            "resumed_existing": True,
        }

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = pathlib.Path(temporary_name)
    digest = hashlib.sha256()
    checksum = 0
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as target, archive.open(info, "r") as source:
            while True:
                chunk = source.read(8 * 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                digest.update(chunk)
                checksum = zlib.crc32(chunk, checksum)
                size += len(chunk)
            target.flush()
            os.fsync(target.fileno())
        if size != info.file_size or checksum & 0xFFFFFFFF != info.CRC:
            raise ExtractionError(f"selected member checksum differs: {info.filename}")
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise ExtractionError(
                f"destination appeared during extraction: {destination.name}"
            ) from exc
        destination.chmod(0o440)
        return {
            "filename": destination.name,
            "size_bytes": size,
            "sha256": digest.hexdigest(),
            "zip_crc32": f"{info.CRC:08x}",
            "resumed_existing": False,
        }
    finally:
        temporary.unlink(missing_ok=True)


def _probe_streams(ffprobe_path: pathlib.Path, media_path: pathlib.Path) -> list[str]:
    try:
        completed = subprocess.run(
            [
                str(ffprobe_path),
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "json",
                str(media_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ExtractionError(f"media stream probe failed: {media_path.name}") from exc
    if completed.returncode != 0 or len(completed.stdout) > 1_000_000:
        raise ExtractionError(f"media stream probe failed: {media_path.name}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ExtractionError(
            f"media stream probe is invalid: {media_path.name}"
        ) from exc
    streams = payload.get("streams") if isinstance(payload, Mapping) else None
    if not isinstance(streams, list):
        raise ExtractionError(f"media streams are missing: {media_path.name}")
    stream_types = sorted(
        {
            stream.get("codec_type")
            for stream in streams
            if isinstance(stream, Mapping) and isinstance(stream.get("codec_type"), str)
        }
    )
    if not {"audio", "video"} <= set(stream_types):
        raise ExtractionError(f"audio or video stream is missing: {media_path.name}")
    return stream_types


def _probe_identity(ffprobe_path: pathlib.Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [str(ffprobe_path), "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ExtractionError("ffprobe identity check failed") from exc
    first_line = completed.stdout.splitlines()[0] if completed.stdout else ""
    if completed.returncode != 0 or not first_line or len(first_line) > 500:
        raise ExtractionError("ffprobe identity check failed")
    return {
        "filename": ffprobe_path.name,
        "sha256": _sha256_path(ffprobe_path),
        "version": first_line,
    }


def extract_selected_media(
    *,
    archive_path: pathlib.Path,
    expected_archive_sha256: str,
    pilot_path: pathlib.Path,
    expected_pilot_sha256: str,
    output_root: pathlib.Path,
    ffprobe_path: pathlib.Path,
) -> dict[str, Any]:
    """Authenticate inputs and extract the exact unique videos in the pilot."""

    archive_input = pathlib.Path(archive_path)
    if archive_input.is_symlink():
        raise ExtractionError("train archive must not be a symlink")
    archive_path = archive_input.resolve(strict=True)
    if not archive_path.is_file():
        raise ExtractionError("train archive must be a regular non-symlink file")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_archive_sha256):
        raise ExtractionError("train archive expected SHA-256 is invalid")
    if _sha256_path(archive_path) != expected_archive_sha256:
        raise ExtractionError("train archive SHA-256 differs")
    pilot_input = pathlib.Path(pilot_path)
    if pilot_input.is_symlink():
        raise ExtractionError("pilot index must not be a symlink")
    pilot = _locked_json(pilot_input, expected_pilot_sha256, "pilot index")
    video_ids = _pilot_video_ids(pilot)
    output_root.mkdir(parents=True, exist_ok=True)
    if output_root.is_symlink() or not output_root.is_dir():
        raise ExtractionError("output root must be a non-symlink directory")
    output_root = output_root.resolve(strict=True)
    ffprobe_path = pathlib.Path(ffprobe_path).resolve(strict=True)
    if not ffprobe_path.is_file() or not os.access(ffprobe_path, os.X_OK):
        raise ExtractionError("ffprobe must be an executable regular file")
    wanted = {f"{video_id}.mp4" for video_id in video_ids}
    existing_names = {path.name for path in output_root.iterdir()}
    extra_names = sorted(existing_names - wanted)
    if extra_names:
        raise ExtractionError(
            f"output root contains {len(extra_names)} unselected entries"
        )
    with zipfile.ZipFile(archive_path) as archive:
        members = _member_map(archive, wanted)
        files = [
            _extract_one(archive, members[name], output_root / name)
            for name in sorted(members)
        ]
    for record in files:
        media_path = output_root / record["filename"]
        record["stream_types"] = _probe_streams(ffprobe_path, media_path)
        if _sha256_path(media_path) != record["sha256"]:
            raise ExtractionError(
                f"media changed during stream probe: {media_path.name}"
            )
    if {path.name for path in output_root.iterdir()} != wanted:
        raise ExtractionError(
            "output root membership differs from the selected media set"
        )
    return {
        "schema": "conflictbench.perception-pilot-media-receipt.v1",
        "status": "selected_media_extracted",
        "input_digests": {
            "train_archive_sha256": expected_archive_sha256,
            "pilot_index_sha256": expected_pilot_sha256,
        },
        "media_root_name": output_root.name,
        "media_file_count": len(files),
        "media_files": files,
        "stream_probe": _probe_identity(ffprobe_path),
    }


def validate_media_receipt(
    *,
    receipt_path: pathlib.Path,
    output_root: pathlib.Path,
    expected_archive_sha256: str,
    expected_pilot_sha256: str,
    ffprobe_path: pathlib.Path,
) -> dict[str, Any]:
    """Reauthenticate a completed extraction and every selected media byte."""

    receipt_input = pathlib.Path(receipt_path)
    if receipt_input.is_symlink():
        raise ExtractionError("media receipt must not be a symlink")
    try:
        receipt = json.loads(receipt_input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExtractionError("media receipt is not valid JSON") from exc
    if not isinstance(receipt, Mapping):
        raise ExtractionError("media receipt must be a JSON object")
    if receipt.get("schema") != "conflictbench.perception-pilot-media-receipt.v1":
        raise ExtractionError("media receipt schema differs")
    if receipt.get("status") != "selected_media_extracted":
        raise ExtractionError("media receipt status differs")
    input_digests = receipt.get("input_digests")
    if not isinstance(input_digests, Mapping):
        raise ExtractionError("media receipt input digests are missing")
    if input_digests.get("train_archive_sha256") != expected_archive_sha256:
        raise ExtractionError("media receipt archive digest differs")
    if input_digests.get("pilot_index_sha256") != expected_pilot_sha256:
        raise ExtractionError("media receipt pilot digest differs")

    root_input = pathlib.Path(output_root)
    if root_input.is_symlink() or not root_input.is_dir():
        raise ExtractionError("media root must be a non-symlink directory")
    root = root_input.resolve(strict=True)
    if receipt.get("media_root_name") != root.name:
        raise ExtractionError("media receipt root name differs")
    probe_input = pathlib.Path(ffprobe_path)
    if probe_input.is_symlink():
        raise ExtractionError("ffprobe must not be a symlink")
    ffprobe_path = probe_input.resolve(strict=True)
    if not ffprobe_path.is_file() or not os.access(ffprobe_path, os.X_OK):
        raise ExtractionError("ffprobe must be an executable regular file")
    files = receipt.get("media_files")
    if not isinstance(files, list) or not 100 <= len(files) <= 300:
        raise ExtractionError("media receipt file list is invalid")
    if receipt.get("media_file_count") != len(files):
        raise ExtractionError("media receipt file count differs")

    expected_names: set[str] = set()
    for item in files:
        if not isinstance(item, Mapping):
            raise ExtractionError("media receipt file entry is invalid")
        filename = item.get("filename")
        if (
            not isinstance(filename, str)
            or _VIDEO_FILENAME.fullmatch(filename) is None
            or filename in expected_names
        ):
            raise ExtractionError("media receipt filename is invalid or duplicated")
        expected_names.add(filename)
        media_path = root / filename
        if media_path.is_symlink():
            raise ExtractionError(f"media file must not be a symlink: {filename}")
        try:
            media_stat = media_path.stat()
        except OSError as exc:
            raise ExtractionError(f"media file is missing: {filename}") from exc
        if not stat.S_ISREG(media_stat.st_mode) or media_stat.st_size != item.get(
            "size_bytes"
        ):
            raise ExtractionError(f"media file size differs: {filename}")
        expected_sha256 = item.get("sha256")
        if (
            not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
            or _sha256_path(media_path) != expected_sha256
        ):
            raise ExtractionError(f"media file digest differs: {filename}")
        recorded_streams = item.get("stream_types")
        if not isinstance(recorded_streams, list) or not {
            "audio",
            "video",
        } <= set(recorded_streams):
            raise ExtractionError(f"media stream record differs: {filename}")
        if _probe_streams(ffprobe_path, media_path) != sorted(set(recorded_streams)):
            raise ExtractionError(f"media stream probe differs: {filename}")
    if {path.name for path in root.iterdir()} != expected_names:
        raise ExtractionError("media root membership differs from the receipt")

    probe = receipt.get("stream_probe")
    if not isinstance(probe, Mapping) or dict(probe) != _probe_identity(ffprobe_path):
        raise ExtractionError("media receipt stream-probe identity differs")
    return dict(receipt)


def _write_once(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ExtractionError(f"output already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=pathlib.Path)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--pilot-index", type=pathlib.Path)
    parser.add_argument("--pilot-index-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=pathlib.Path)
    parser.add_argument("--ffprobe", required=True, type=pathlib.Path)
    parser.add_argument("--receipt", required=True, type=pathlib.Path)
    parser.add_argument("--verify-existing-receipt", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_existing_receipt:
        validate_media_receipt(
            receipt_path=args.receipt,
            output_root=args.output_root,
            expected_archive_sha256=args.archive_sha256,
            expected_pilot_sha256=args.pilot_index_sha256,
            ffprobe_path=args.ffprobe,
        )
        return 0
    if args.archive is None or args.pilot_index is None:
        parser.error("--archive and --pilot-index are required for extraction")
    result = extract_selected_media(
        archive_path=args.archive,
        expected_archive_sha256=args.archive_sha256,
        pilot_path=args.pilot_index,
        expected_pilot_sha256=args.pilot_index_sha256,
        output_root=args.output_root,
        ffprobe_path=args.ffprobe,
    )
    _write_once(args.receipt, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
