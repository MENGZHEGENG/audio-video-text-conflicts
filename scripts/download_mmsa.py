"""Download one public MMSA processed descriptor pickle.

Only pre-extracted features are requested; raw videos and annotation archives
are intentionally outside this helper's scope.  The source IDs and hashes are
the values published by the MMSA project and are recorded beside the file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


FILES = {
    "mosi": {
        "dataset": "CMU-MOSI",
        "drive_id": "1VqjkYqcgUlggZVN7B3NpXwQ-BIWIXxkj",
        "filename": "aligned_50.pkl",
        "sha256": "d3994fd25681f9c7ad6e9c6596a6fe9b4beb85ff7d478ba978b124139002e5f9",
    },
    "sims": {
        "dataset": "CH-SIMS",
        "drive_id": "1l_Nb9h3BRa3S-N76YcSQ3ixakr0gn6Qr",
        "filename": "unaligned_39.pkl",
        "sha256": "c9e20c13ec0454d98bb9c1e520e490c75146bfa2dfeeea78d84de047dbdd442f",
    },
    "sims_v2": {
        "dataset": "CH-SIMS v2",
        "drive_id": "13JdO6GbPHOGZ8yLBFUrNR8c2FvHN-E_O",
        "filename": "unaligned.pkl",
        "sha256": None,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_expected_sha256(pinned: str | None, supplied: str | None) -> str:
    """Return a required normalized digest, preserving a built-in pin."""

    if pinned is not None and supplied is not None and supplied.lower() != pinned:
        raise ValueError("supplied expected SHA-256 differs from the built-in pin")
    value = pinned or supplied
    if value is None:
        raise ValueError("an expected SHA-256 is required for an unpinned source")
    normalized = value.lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ValueError("expected SHA-256 must contain 64 lowercase hexadecimal characters")
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(FILES))
    parser.add_argument("--drive-id", help="override the published Google Drive file ID")
    parser.add_argument(
        "--expected-sha256",
        help="required 64-hex digest when the selected source has no built-in pin",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.dataset and not args.drive_id:
        parser.error("provide --dataset or --drive-id")
    spec = dict(FILES.get(args.dataset, {}))
    drive_id = args.drive_id or spec["drive_id"]
    filename = spec.get("filename") or "descriptor.pkl"
    try:
        expected = resolve_expected_sha256(spec.get("sha256"), args.expected_sha256)
    except ValueError as exc:
        parser.error(str(exc))
    destination = args.output_dir.expanduser().resolve() / filename
    args.output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    if destination.exists() and not args.force:
        digest = sha256(destination)
        if digest != expected:
            raise SystemExit(f"existing file hash mismatch: {destination}")
    else:
        try:
            import gdown
        except ImportError as exc:
            raise SystemExit("install gdown (for example, python -m pip install gdown)") from exc
        downloaded = gdown.download(id=drive_id, output=str(destination), quiet=False, resume=True)
        if not downloaded:
            raise SystemExit("Google Drive download failed")
    digest = sha256(destination)
    if digest != expected:
        raise SystemExit(f"SHA256 mismatch for {destination}: {digest} != {expected}")
    record = {
        "dataset": spec.get("dataset", args.dataset or "custom"),
        "source": "Google Drive",
        "drive_id": drive_id,
        "filename": filename,
        "sha256": digest,
        "expected_sha256": expected,
        "path": str(destination),
    }
    (args.output_dir.expanduser().resolve() / "download_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
