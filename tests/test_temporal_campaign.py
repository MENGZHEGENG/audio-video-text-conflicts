import importlib.util
from pathlib import Path

import pytest

from conflictbench.temporal_campaign import (
    TEMPORAL_CAMPAIGN_SCHEMA,
    expand_campaign_spec,
    resolve_campaign_job,
    validate_campaign_spec,
)


_ANALYSIS_PATH = Path(__file__).parents[1] / "scripts" / "analyze_temporal_campaign.py"
_ANALYSIS_SPEC = importlib.util.spec_from_file_location("analyze_temporal_campaign", _ANALYSIS_PATH)
assert _ANALYSIS_SPEC is not None and _ANALYSIS_SPEC.loader is not None
analysis = importlib.util.module_from_spec(_ANALYSIS_SPEC)
_ANALYSIS_SPEC.loader.exec_module(analysis)


def _campaign_spec():
    return {
        "schema": TEMPORAL_CAMPAIGN_SCHEMA,
        "name": "test_temporal",
        "base_config": {
            "dataset": "synthetic_temporal",
            "train_size": 8,
            "eval_size": 6,
            "train_mechanisms": ["clean"],
            "seen_mechanisms": ["clean"],
            "unseen_mechanisms": ["burst"],
            "seq_len": 8,
            "noise": 0.2,
        },
        "cells": [
            {"id": "short", "hypothesis": "short", "overrides": {"seq_len": 8}},
            {"id": "long", "hypothesis": "long", "overrides": {"seq_len": 16}},
        ],
        "primary_seeds": [11, 23],
        "replication_seeds": [701],
    }


def test_campaign_spec_expansion_is_stable_and_embeds_cell_metadata():
    spec = _campaign_spec()
    jobs = expand_campaign_spec(spec, "primary")
    assert len(jobs) == 4
    assert [(job.cell_id, job.seed, job.index) for job in jobs] == [
        ("short", 11, 0),
        ("short", 23, 1),
        ("long", 11, 2),
        ("long", 23, 3),
    ]
    assert jobs[2].config["campaign_cell"] == "long"
    assert jobs[2].config["campaign_run_index"] == 2
    replication = resolve_campaign_job(spec, "replication", 0)
    assert (replication.cell_id, replication.seed, replication.seed_group) == (
        "short",
        701,
        "replication",
    )


def test_campaign_spec_rejects_duplicate_or_protected_fields():
    spec = _campaign_spec()
    spec["cells"][1]["id"] = "short"
    with pytest.raises(ValueError, match="duplicate cell"):
        validate_campaign_spec(spec)

    spec = _campaign_spec()
    spec["cells"][0]["overrides"] = {"seed": 3}
    with pytest.raises(ValueError, match="protected"):
        validate_campaign_spec(spec)


def test_campaign_spec_rejects_overlapping_seed_groups():
    spec = _campaign_spec()
    spec["replication_seeds"] = [11]
    with pytest.raises(ValueError, match="disjoint"):
        validate_campaign_spec(spec)


def test_campaign_replication_overrides_are_group_specific_and_provenance_checked():
    spec = _campaign_spec()
    spec["replication_overrides"] = {"temporal_max_steps": 8000}
    primary = resolve_campaign_job(spec, "primary", 0)
    replication = resolve_campaign_job(spec, "replication", 0)
    assert "temporal_max_steps" not in primary.config
    assert replication.config["temporal_max_steps"] == 8000

    spec["replication_overrides"] = {"seed": 3}
    with pytest.raises(ValueError, match="protected"):
        validate_campaign_spec(spec)


def test_campaign_record_requires_spec_config_and_resource_device():
    spec = _campaign_spec()
    expected = expand_campaign_spec(spec, "primary")[0]
    record = {
        "config": dict(expected.config),
        "environment": {"used_device": "cuda"},
    }
    assert analysis._campaign_record_error(record, expected) is None

    record["config"] = {**expected.config, "seq_len": 99}
    assert analysis._campaign_record_error(record, expected) == "resolved config does not match the campaign job"

    record["config"] = dict(expected.config)
    record["environment"] = {"used_device": "cpu"}
    assert analysis._campaign_record_error(record, expected) == (
        "recorded device does not match primary seed group"
    )
