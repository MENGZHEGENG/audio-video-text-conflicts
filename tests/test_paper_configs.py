from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_mosei_paper_config_matches_reported_training_schedule() -> None:
    config = json.loads(
        (ROOT / "configs" / "mosei_paper.json").read_text(encoding="utf-8")
    )
    assert config["dataset"] == "mosei"
    assert config["epochs"] == 40
    assert config["batch_size"] == 256
    assert config["learning_rate"] == 0.0015
    assert config["initial_threshold"] == 0.35
    assert config["seed"] == 11


def test_public_tree_uses_research_groups_not_allocation_metadata() -> None:
    prohibited = (
        "gpu" + "_seeds",
        "cpu" + "_seeds",
        "campaign_" + "job_index",
        "GPU " + "allocation",
    )
    paths = [
        *ROOT.glob("configs/*.json"),
        *ROOT.glob("scripts/*.py"),
        *ROOT.glob("src/**/*.py"),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert not any(term in text for term in prohibited), path


def test_anonymous_software_license_names_a_holder() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Copyright (c) 2026 The Authors" in license_text
