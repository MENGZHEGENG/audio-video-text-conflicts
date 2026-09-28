"""Minimal authenticated-by-digest downloader for public CREMA-D Git-LFS objects."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping
from urllib.error import HTTPError
from urllib.request import Request, urlopen


BATCH_URL = "https://github.com/CheyneyComputerScience/CREMA-D.git/info/lfs/objects/batch"
POINTER_SCHEMA = "conflictbench.cremad-pointer-index.v1"
CLIP_ID = re.compile(r"[0-9]{4}_[A-Z]{3}_[A-Z]{3}_[A-Z]{2}")
MEDIA = {"audio": ("AudioWAV", ".wav"), "video": ("VideoFlash", ".flv")}


@dataclass(frozen=True)
class DownloadTask:
    """One media file tied to an official LFS object identity."""

    path: Path
    oid: str
    size_bytes: int


def _mapping(value: object, error: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(error)
    return value


def tasks_from_pointer_index(repository_root: Path, pointer_index_path: Path) -> list[DownloadTask]:
    """Derive destination paths solely from a validated pointer index."""

    index = json.loads(pointer_index_path.read_text(encoding="utf-8"))
    if not isinstance(index, Mapping) or index.get("schema") != POINTER_SCHEMA:
        raise ValueError("pointer index")
    records = index.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("pointer records")
    tasks: list[DownloadTask] = []
    paths: set[Path] = set()
    for record_value in records:
        record = _mapping(record_value, "pointer records")
        clip_id = record.get("clip_id")
        if not isinstance(clip_id, str) or CLIP_ID.fullmatch(clip_id) is None:
            raise ValueError("pointer records")
        for modality, (directory, suffix) in MEDIA.items():
            expected = _mapping(record.get(modality), "pointer records")
            oid = expected.get("sha256")
            size = expected.get("size_bytes")
            if (
                not isinstance(oid, str)
                or re.fullmatch(r"[0-9a-f]{64}", oid) is None
                or not isinstance(size, int)
                or isinstance(size, bool)
                or size <= 0
            ):
                raise ValueError("pointer records")
            path = repository_root / directory / f"{clip_id}{suffix}"
            if path in paths:
                raise ValueError("pointer records")
            paths.add(path)
            tasks.append(DownloadTask(path=path, oid=oid, size_bytes=size))
    return tasks


def _batch_actions(tasks: list[DownloadTask]) -> dict[str, tuple[str, Mapping[str, str]]]:
    body = json.dumps(
        {
            "operation": "download",
            "transfers": ["basic"],
            "objects": [{"oid": task.oid, "size": task.size_bytes} for task in tasks],
        }
    ).encode("utf-8")
    request = Request(
        BATCH_URL,
        data=body,
        headers={
            "Accept": "application/vnd.git-lfs+json",
            "Content-Type": "application/vnd.git-lfs+json",
            "User-Agent": "cremad-source-admission/1",
        },
    )
    payload: object | None = None
    for attempt in range(4):
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.load(response)
            break
        except HTTPError as exc:
            if exc.code != 429 or attempt == 3:
                raise
            retry_after = exc.headers.get("Retry-After", "60")
            try:
                delay_seconds = int(retry_after)
            except ValueError:
                delay_seconds = 60
            time.sleep(max(1, min(delay_seconds, 120)))
    if payload is None:
        raise ValueError("LFS batch response")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("objects"), list):
        raise ValueError("LFS batch response")
    actions: dict[str, tuple[str, Mapping[str, str]]] = {}
    for value in payload["objects"]:
        item = _mapping(value, "LFS batch response")
        oid = item.get("oid")
        action_map = _mapping(item.get("actions"), "LFS batch response")
        download = _mapping(action_map.get("download"), "LFS batch response")
        href = download.get("href")
        headers = download.get("header", {})
        if not isinstance(oid, str) or not isinstance(href, str) or not isinstance(headers, Mapping):
            raise ValueError("LFS batch response")
        normalized_headers = {str(key): str(value) for key, value in headers.items()}
        if oid in actions:
            raise ValueError("LFS batch response")
        actions[oid] = (href, normalized_headers)
    expected = {task.oid for task in tasks}
    if set(actions) != expected:
        raise ValueError("LFS batch response")
    return actions


def _download_one(task: DownloadTask, href: str, headers: Mapping[str, str]) -> None:
    partial = task.path.with_name(f"{task.path.name}.admission.partial")
    if partial.exists():
        raise ValueError(f"partial media exists: {task.path.name}")
    request_headers = {"User-Agent": "cremad-source-admission/1", **headers}
    digest = hashlib.sha256()
    written = 0
    try:
        with urlopen(Request(href, headers=request_headers), timeout=120) as response:
            with partial.open("xb") as handle:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(block)
                    digest.update(block)
                    written += len(block)
        if written != task.size_bytes or digest.hexdigest() != task.oid:
            raise ValueError(f"downloaded media mismatch: {task.path.name}")
        partial.replace(task.path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _unique_objects(tasks: list[DownloadTask]) -> list[DownloadTask]:
    """Request each content-addressed LFS object once, retaining every path."""

    unique: dict[str, DownloadTask] = {}
    for task in tasks:
        prior = unique.get(task.oid)
        if prior is not None and prior.size_bytes != task.size_bytes:
            raise ValueError("pointer records")
        unique.setdefault(task.oid, task)
    return list(unique.values())


def download_media(repository_root: Path, pointer_index_path: Path, workers: int = 4) -> dict[str, int]:
    """Download and individually authenticate all media described by pointers."""

    if workers < 1 or workers > 8:
        raise ValueError("workers")
    tasks = tasks_from_pointer_index(repository_root, pointer_index_path)
    unique_tasks = _unique_objects(tasks)
    actions: dict[str, tuple[str, Mapping[str, str]]] = {}
    for offset in range(0, len(unique_tasks), 100):
        actions.update(_batch_actions(unique_tasks[offset : offset + 100]))
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_download_one, task, *actions[task.oid])
            for task in tasks
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:  # retain all surfaced failures for debugging
                failures.append(str(exc))
    if failures:
        raise ValueError("media download: " + "; ".join(sorted(failures)[:5]))
    return {"clip_count": len(tasks) // 2, "media_count": len(tasks), "total_bytes": sum(task.size_bytes for task in tasks)}
