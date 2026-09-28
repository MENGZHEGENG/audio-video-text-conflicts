from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_scalar_campaign.py"
SPEC = importlib.util.spec_from_file_location("run_scalar_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_load_seed_plan_preserves_groups_and_rejects_overlap(tmp_path: Path) -> None:
    path = tmp_path / "seeds.json"
    path.write_text(
        json.dumps({"primary_seeds": [11, 23], "replication_seeds": [37, 41]}),
        encoding="utf-8",
    )
    assert RUNNER.load_seed_plan(path) == (11, 23, 37, 41)

    path.write_text(
        json.dumps({"primary_seeds": [11, 23], "replication_seeds": [23, 41]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="disjoint"):
        RUNNER.load_seed_plan(path)


def test_run_campaign_writes_paired_records_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign.json"
    acquisition = tmp_path / "acquisition.json"
    campaign.write_text(json.dumps({"name": "base", "seed": 0}), encoding="utf-8")
    acquisition.write_text(
        json.dumps({"name": "acquisition", "seed": 0}), encoding="utf-8"
    )
    calls: list[tuple[str, int, str]] = []

    def fake_run(config: dict, device_preference: str) -> dict:
        calls.append((config["name"], config["seed"], device_preference))
        return {"seed": config["seed"], "validation": {"ok": True}}

    monkeypatch.setattr(RUNNER, "run_from_config", fake_run)
    output = tmp_path / "runs"
    RUNNER.run_campaign(campaign, acquisition, (11, 23), output, "cpu")

    assert calls == [
        ("base", 11, "cpu"),
        ("acquisition", 11, "cpu"),
        ("base", 23, "cpu"),
        ("acquisition", 23, "cpu"),
    ]
    assert json.loads((output / "campaign" / "seed-11.json").read_text())["seed"] == 11
    assert (
        json.loads((output / "acquisition" / "seed-23.json").read_text())["seed"] == 23
    )

    with pytest.raises(FileExistsError):
        RUNNER.run_campaign(campaign, acquisition, (11, 23), output, "cpu")
