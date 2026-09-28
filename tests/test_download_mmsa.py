import hashlib
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "download_mmsa.py"
SPEC = importlib.util.spec_from_file_location("download_mmsa", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_public_processed_file_specs_are_pinned():
    assert MODULE.FILES["mosi"]["drive_id"] == "1VqjkYqcgUlggZVN7B3NpXwQ-BIWIXxkj"
    assert MODULE.FILES["sims"]["drive_id"] == "1l_Nb9h3BRa3S-N76YcSQ3ixakr0gn6Qr"
    assert MODULE.FILES["mosi"]["sha256"] == "d3994fd25681f9c7ad6e9c6596a6fe9b4beb85ff7d478ba978b124139002e5f9"
    assert MODULE.FILES["sims"]["sha256"] == "c9e20c13ec0454d98bb9c1e520e490c75146bfa2dfeeea78d84de047dbdd442f"


def test_unpinned_source_requires_an_explicit_sha256():
    with pytest.raises(ValueError, match="expected SHA-256"):
        MODULE.resolve_expected_sha256(None, None)
    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        MODULE.resolve_expected_sha256(None, "not-a-digest")

    digest = "a" * 64
    assert MODULE.resolve_expected_sha256(None, digest.upper()) == digest


def test_sha256_helper_is_streaming_compatible(tmp_path):
    path = tmp_path / "small.bin"
    path.write_bytes(b"conflictbench\n")
    assert MODULE.sha256(path) == hashlib.sha256(b"conflictbench\n").hexdigest()
