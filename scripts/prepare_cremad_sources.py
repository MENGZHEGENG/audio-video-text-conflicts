#!/usr/bin/env python3
"""Admit pinned CREMA-D media and build its actor-disjoint evaluation index.

Run the pointer phase before ``git lfs pull`` and the verify phase afterward.
The dataset remains outside this code repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from conflictbench.cremad_index import build_index
from conflictbench.cremad_media_verification import verify_media
from conflictbench.cremad_preflight import build_pointer_index
from conflictbench.cremad_registry import EXPECTED_COMMIT, build_registry
from conflictbench.source_registry import verify_source_registry_files


def write_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pointer", "verify"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    repository_root = args.repository_root.resolve()
    if not repository_root.is_relative_to(data_root):
        parser.error("the dataset repository must be inside --data-root")
    pointer_path = data_root / "cremad_pointer_records.json"
    if args.phase == "pointer":
        write_new(pointer_path, build_pointer_index(repository_root, EXPECTED_COMMIT))
        print(pointer_path)
        return

    media_path = data_root / "cremad_media_verification.json"
    role_path = data_root / "cremad_actor_roles.json"
    registry_path = data_root / "cremad_source_registry.json"
    index_path = data_root / "cremad_paired_index.json"
    if not pointer_path.is_file():
        parser.error("run the pointer phase before downloading LFS media")
    if any(path.exists() for path in (media_path, role_path, registry_path, index_path)):
        parser.error("a verify-phase output already exists")
    media = verify_media(repository_root, pointer_path)
    write_new(media_path, media)
    registry = build_registry(data_root, repository_root, pointer_path, media_path, role_path)
    verify_source_registry_files(registry, data_root)
    write_new(registry_path, registry)
    write_new(index_path, build_index(repository_root, role_path))
    print(json.dumps({"records": 7442, "registry": str(registry_path), "index": str(index_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
