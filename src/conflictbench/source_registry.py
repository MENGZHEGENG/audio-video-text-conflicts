"""Validation for provenance-complete natural-data source registries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping


SCHEMA = "conflictbench.natural-source-registry.v4"
REQUIRED_ROLES = ("task_fit", "router_fit", "calibration", "evaluation")
OPTIONAL_ROLES = ("same_label_donor",)
ROLES = REQUIRED_ROLES + OPTIONAL_ROLES
LABELS = ("integrated", "audio", "video", "text")
GROUP_UNITS = ("source", "speaker", "meeting_team")
FILE_ROLES = (
    "release_index",
    "data",
    "labels",
    "role_allocation",
    "group_mapping",
    "annotation_archive",
    "license",
    "audio",
    "video",
    "download_index",
    "trusted_index",
    "audited_url_inventory",
    "tls_tofu_baseline",
)
REQUIRED_FILE_ROLES = {
    "release_index",
    "data",
    "labels",
    "role_allocation",
    "group_mapping",
}
SOURCE_AUTHENTICATION_FILE_ROLES = {
    "trusted_index",
    "download_index",
    "audited_url_inventory",
    "tls_tofu_baseline",
}
SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class SourceRegistryValidation:
    """A compact validated summary for an admitted natural-data source."""

    dataset_name: str
    group_count: int
    role_counts: dict[str, int]
    file_count: int


def _mapping(record: Mapping[str, Any], key: str, error: str) -> Mapping[str, Any]:
    value = record.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(error)
    return value


def _nonempty_string(record: Mapping[str, Any], key: str, error: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(error)
    return value.strip()


def _sha256(record: Mapping[str, Any], key: str, error: str) -> str:
    value = _nonempty_string(record, key, error).lower()
    if SHA256.fullmatch(value) is None:
        raise ValueError(error)
    return value


def _validate_files(record: Mapping[str, Any]) -> tuple[int, set[str]]:
    """Require every phase-zero input to be tied to a verified index entry.

    A release can store labels, split assignments, and source groups in one
    file, so one file may carry several roles.  The explicit roles prevent
    a checked model bundle from being silently reused as unverified metadata.
    """

    files = record.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("files")

    file_roles: set[str] = set()
    filenames: set[str] = set()
    for file in files:
        if not isinstance(file, Mapping):
            raise ValueError("files")
        filename = _nonempty_string(file, "filename", "files")
        if filename in filenames:
            raise ValueError("files")
        filenames.add(filename)
        size_bytes = file.get("size_bytes")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
            raise ValueError("files")
        source_url = _nonempty_string(file, "source_url", "files")
        if not source_url.startswith(("https://", "http://")):
            raise ValueError("files")
        expected = _sha256(file, "expected_sha256", "files")
        observed = _sha256(file, "observed_sha256", "files")
        if observed != expected:
            raise ValueError("files")
        roles = file.get("roles")
        if not isinstance(roles, list) or not roles:
            raise ValueError("files")
        if any(role not in FILE_ROLES for role in roles) or len(set(roles)) != len(roles):
            raise ValueError("files")
        file_roles.update(roles)

    if not REQUIRED_FILE_ROLES.issubset(file_roles):
        raise ValueError("files")
    return len(files), file_roles


def _file_path(root: Path, filename: str) -> Path:
    """Return a staged file path only when it remains inside ``root``."""

    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("files")
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("files") from exc
    return candidate


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source_registry(record: Mapping[str, Any]) -> SourceRegistryValidation:
    """Reject a source unless every Phase 0 provenance and split gate passes."""

    if record.get("schema") != SCHEMA:
        raise ValueError("schema")

    dataset = _mapping(record, "dataset", "dataset")
    dataset_name = _nonempty_string(dataset, "name", "dataset")
    _nonempty_string(dataset, "release", "dataset")

    terms = _mapping(record, "terms", "terms")
    terms_url = _nonempty_string(terms, "url", "terms")
    if not terms_url.startswith(("https://", "http://")):
        raise ValueError("terms")
    _nonempty_string(terms, "reviewed_on", "terms")
    try:
        date.fromisoformat(str(terms["reviewed_on"]))
    except ValueError as exc:
        raise ValueError("terms") from exc
    if terms.get("research_use_permitted") is not True:
        raise ValueError("terms")

    file_count, file_roles = _validate_files(record)

    source_authentication = record.get("source_authentication")
    if source_authentication is None:
        if SOURCE_AUTHENTICATION_FILE_ROLES.intersection(file_roles):
            raise ValueError("source authentication")
    else:
        if source_authentication == "tls_tofu":
            if not {"audited_url_inventory", "tls_tofu_baseline"}.issubset(file_roles):
                raise ValueError("source authentication files")
            if {"trusted_index", "download_index"}.intersection(file_roles):
                raise ValueError("source authentication files")
        elif source_authentication == "publisher_checksum":
            if not {"trusted_index", "download_index"}.issubset(file_roles):
                raise ValueError("source authentication files")
            if {"audited_url_inventory", "tls_tofu_baseline"}.intersection(file_roles):
                raise ValueError("source authentication files")
        else:
            raise ValueError("source authentication")

    labels = _mapping(record, "labels", "labels")
    if any(not isinstance(labels.get(label), str) or not labels[label].strip() for label in LABELS):
        raise ValueError("labels")
    if labels.get("file_role") != "labels":
        raise ValueError("labels")
    _nonempty_string(record, "group_field", "groups")

    grouping = _mapping(record, "grouping", "grouping")
    if grouping.get("unit") not in GROUP_UNITS:
        raise ValueError("grouping")
    provenance_url = _nonempty_string(grouping, "provenance_url", "grouping")
    if not provenance_url.startswith(("https://", "http://")):
        raise ValueError("grouping")
    if grouping.get("file_role") != "group_mapping":
        raise ValueError("grouping")

    role_allocation = _mapping(record, "role_allocation", "splits")
    _nonempty_string(role_allocation, "method", "splits")
    if role_allocation.get("file_role") != "role_allocation":
        raise ValueError("splits")

    role_groups = _mapping(record, "role_groups", "groups")
    if not set(REQUIRED_ROLES).issubset(role_groups) or not set(role_groups).issubset(ROLES):
        raise ValueError("groups")
    seen: set[str] = set()
    role_counts: dict[str, int] = {}
    for role in ROLES:
        if role not in role_groups:
            continue
        groups = role_groups.get(role)
        if not isinstance(groups, list) or not groups:
            raise ValueError("groups")
        normalized = [group for group in groups if isinstance(group, str) and group.strip()]
        if len(normalized) != len(groups) or len(set(normalized)) != len(normalized):
            raise ValueError("groups")
        overlap = seen.intersection(normalized)
        if overlap:
            raise ValueError("overlap")
        seen.update(normalized)
        role_counts[role] = len(normalized)

    return SourceRegistryValidation(
        dataset_name=dataset_name,
        group_count=len(seen),
        role_counts=role_counts,
        file_count=file_count,
    )


def verify_source_registry_files(
    record: Mapping[str, Any], source_root: Path
) -> SourceRegistryValidation:
    """Validate the registry and recompute every staged file's identity.

    Registry metadata alone is not source admission.  This operation is the
    only path used by the CLI and binds the declared index records to bytes
    staged under a caller-provided root.
    """

    validated = validate_source_registry(record)
    if not source_root.is_dir():
        raise ValueError("file root")
    files = record["files"]
    assert isinstance(files, list)  # established by validate_source_registry
    for file in files:
        assert isinstance(file, Mapping)
        filename = _nonempty_string(file, "filename", "files")
        path = _file_path(source_root, filename)
        if not path.is_file():
            raise ValueError("file bytes")
        size_bytes = file["size_bytes"]
        assert isinstance(size_bytes, int)
        if path.stat().st_size != size_bytes:
            raise ValueError("file bytes")
        observed = _file_sha256(path)
        if observed != file["expected_sha256"] or observed != file["observed_sha256"]:
            raise ValueError("file bytes")
    return validated
