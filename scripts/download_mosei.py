#!/usr/bin/env python3
"""Download the official CMU-MOSEI CSD inputs and standard folds.

The feature URLs are the ones published by the CMU Multimodal SDK.  The
legacy hosting service can be unavailable from some networks; in that case
the script stops without creating partial output and suggests using an
approved mirror or an existing local copy with the same filenames.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pathlib
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


DEFAULT_BASE_URL = "http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI"
DEFAULT_HF_REPO = "reeha-parkar/cmu-mosei-comp-seq"
DEFAULT_HF_REVISION = "5f8d513c34278006d27f98e5609564e6d4b353d7"
DEFAULT_FOLD_URL = (
    "https://raw.githubusercontent.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/"
    "4f2eadcd7e7b9e20e83b868cdad385b41830285c/"
    "mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/cmu_mosei_std_folds.py"
)
RESOURCES = {
    "audio": ("acoustic", "CMU_MOSEI_COVAREP.csd"),
    "video": ("visual", "CMU_MOSEI_VisualOpenFace2.csd"),
    "text": ("language", "CMU_MOSEI_TimestampedWordVectors.csd"),
    "labels": ("labels", "CMU_MOSEI_Labels.csd"),
}
HF_FILENAMES = {
    "audio": "CMU_MOSEI_COVAREP.csd",
    "video": "CMU_MOSEI_OpenFace2.csd",
    "text": "CMU_MOSEI_TimestampedWordVectors.csd",
    "labels": "CMU_MOSEI_Labels.csd",
}


def _urlopen(url: str, timeout: float) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "conflictbench/0.1"})
    return urllib.request.urlopen(request, timeout=timeout)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_expected_hashes(values: list[str]) -> dict[str, str]:
    expected: dict[str, str] = {}
    permitted = {*RESOURCES, "folds"}
    for value in values:
        name, separator, digest = value.partition("=")
        digest = digest.lower()
        if separator != "=" or name not in permitted:
            raise ValueError(
                "expected hashes must use NAME=SHA256 with NAME in "
                + ", ".join(sorted(permitted))
            )
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"invalid SHA-256 for {name}")
        if name in expected:
            raise ValueError(f"duplicate expected SHA-256 for {name}")
        expected[name] = digest
    return expected


def _download(
    url: str,
    destination: pathlib.Path,
    *,
    timeout: float,
    force: bool,
    expected_sha256: str | None = None,
) -> tuple[str, str]:
    if destination.exists() and not force:
        with destination.open("rb") as handle:
            signature = handle.read(8)
        if signature != b"\x89HDF\r\n\x1a\n":
            raise RuntimeError(f"existing file is not an HDF5 CSD: {destination}")
        digest = _sha256_path(destination)
        if expected_sha256 is not None and digest != expected_sha256:
            raise RuntimeError(f"existing file SHA-256 mismatch: {destination}")
        return "existing", digest

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    temporary = pathlib.Path(temporary_name)
    try:
        with _urlopen(url, timeout) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        with temporary.open("rb") as handle:
            if handle.read(8) != b"\x89HDF\r\n\x1a\n":
                raise RuntimeError(f"download was not an HDF5 CSD (check URL or access): {url}")
        digest = _sha256_path(temporary)
        if expected_sha256 is not None and digest != expected_sha256:
            raise RuntimeError(f"downloaded file SHA-256 mismatch: {destination.name}")
        temporary.replace(destination)
        return "downloaded", digest
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"could not download {url}: {exc}. The legacy CMU host may be unavailable; "
            "use an approved mirror or point --base-url at one."
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _resource_urls(
    base_url: str | None,
    hf_repo: str | None,
    hf_revision: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Resolve resource URLs and retain enough provenance to reproduce them."""

    if hf_repo:
        encoded_repo = urllib.parse.quote(hf_repo.strip("/"), safe="/")
        encoded_revision = urllib.parse.quote(hf_revision, safe="")
        urls = {
            name: (
                f"https://huggingface.co/datasets/{encoded_repo}/resolve/"
                f"{encoded_revision}/data/{urllib.parse.quote(HF_FILENAMES[name])}?download=true"
            )
            for name in RESOURCES
        }
        source = {
            "kind": "huggingface_dataset",
            "repository": hf_repo.strip("/"),
            "revision": hf_revision,
            "unofficial_mirror": True,
        }
        return urls, source

    root = (base_url or DEFAULT_BASE_URL).rstrip("/")
    urls = {name: f"{root}/{subdirectory}/{filename}" for name, (subdirectory, filename) in RESOURCES.items()}
    return urls, {"kind": "cmu_multimodal_sdk", "base_url": root}


def _write_json_atomic(path: pathlib.Path, value: dict[str, Any]) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = pathlib.Path(temporary_name)
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _folds_from_python(source: str) -> dict[str, list[str]]:
    tree = ast.parse(source, filename="cmu_mosei_std_folds.py")
    values: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not target.id.startswith("standard_") or not target.id.endswith("_fold"):
            continue
        parsed = ast.literal_eval(node.value)
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise ValueError(f"unexpected fold definition: {target.id}")
        values[target.id] = parsed
    aliases = {"train": "standard_train_fold", "valid": "standard_valid_fold", "test": "standard_test_fold"}
    missing = [name for name, source_name in aliases.items() if source_name not in values]
    if missing:
        raise ValueError(f"official fold source is missing: {missing}")
    return {name: values[source_name] for name, source_name in aliases.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--base-url", help="CMU-style MOSEI root URL for an approved mirror")
    parser.add_argument(
        "--hf-repo",
        help="Hugging Face dataset repository containing a data/ directory (use with --hf-revision)",
    )
    parser.add_argument("--hf-revision", default=DEFAULT_HF_REVISION, help="Pinned Hugging Face commit or tag")
    parser.add_argument("--fold-url", default=DEFAULT_FOLD_URL)
    parser.add_argument(
        "--expected-sha256",
        action="append",
        default=[],
        metavar="NAME=SHA256",
        help="optional expected digest for audio, video, text, labels, or folds",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--force", action="store_true", help="replace existing CSD files")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.base_url and args.hf_repo:
        parser.error("--base-url and --hf-repo are mutually exclusive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        expected_hashes = _parse_expected_hashes(args.expected_sha256)
    except ValueError as exc:
        parser.error(str(exc))
    statuses: dict[str, str] = {}
    observed_hashes: dict[str, str] = {}
    urls, source = _resource_urls(args.base_url, args.hf_repo, args.hf_revision)
    try:
        for name, (subdirectory, filename) in RESOURCES.items():
            url = urls[name]
            statuses[name], observed_hashes[name] = _download(
                url,
                output_dir / filename,
                timeout=args.timeout,
                force=args.force,
                expected_sha256=expected_hashes.get(name),
            )
        with _urlopen(args.fold_url, args.timeout) as response:
            fold_bytes = response.read()
        observed_hashes["folds"] = _sha256_bytes(fold_bytes)
        if (
            "folds" in expected_hashes
            and observed_hashes["folds"] != expected_hashes["folds"]
        ):
            raise RuntimeError("fold definition SHA-256 mismatch")
        fold_source = fold_bytes.decode("utf-8")
        folds = _folds_from_python(fold_source)
        (output_dir / "splits.json").write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        parser.exit(2, f"download failed: {exc}\n")

    result = {
        "output_dir": str(output_dir),
        "files": {name: str(output_dir / filename) for name, (_, filename) in RESOURCES.items()},
        "splits": str(output_dir / "splits.json"),
        "status": statuses,
        "fold_source": args.fold_url,
        "source": source,
        "urls": urls,
        "sha256": observed_hashes,
        "expected_sha256": expected_hashes,
    }
    _write_json_atomic(output_dir / "download_record.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
