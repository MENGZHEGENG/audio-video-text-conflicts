from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from conflictbench.mosei_raw_value_study import implementation_sha256

RELEASE_ROOT = Path(__file__).parents[1]
ENTRYPOINTS = (
    "cache_mosei_raw_descriptors.py",
    "run_mosei_raw_value_study.py",
    "verify_mosei_raw_value.py",
    "aggregate_mosei_raw_value.py",
)
IMPLEMENTATION_SOURCES = {
    "mosei.py",
    "mosei_raw.py",
    "mosei_raw_value_study.py",
    "mosei_value_study.py",
    "value_of_information.py",
}


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_raw_value_entrypoint_runs_from_outside_release_tree(entrypoint, tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(RELEASE_ROOT / "src"), env.get("PYTHONPATH", "")))
    )

    completed = subprocess.run(
        [sys.executable, str(RELEASE_ROOT / "scripts" / entrypoint), "--help"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()


def test_raw_implementation_digest_reads_only_release_sources(monkeypatch):
    original_read_bytes = Path.read_bytes
    observed: list[Path] = []

    def record_read(path: Path) -> bytes:
        observed.append(path.resolve())
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", record_read)

    digest = implementation_sha256()

    assert len(digest) == 64
    assert {path.name for path in observed} == IMPLEMENTATION_SOURCES
    assert all(path.is_relative_to(RELEASE_ROOT) for path in observed)
