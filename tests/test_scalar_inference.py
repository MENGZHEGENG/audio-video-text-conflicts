from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from scalar_inference import AnalysisError, analyze_runs, main

NOVEL = ("ambiguity", "burst", "mixed")


def _method(accuracy: float, utility: float) -> dict:
    return {
        "metrics": {"decision_accuracy": accuracy, "utility": utility},
        "by_mechanism": {
            mechanism: {
                "n": 10,
                "decision_accuracy": accuracy + offset,
                "utility": utility + offset,
            }
            for mechanism, offset in zip(NOVEL, (-0.1, 0.0, 0.1), strict=True)
        },
    }


def _write_run(path: Path, seed: int, methods: dict[str, tuple[float, float]]) -> None:
    payload = {
        "seed": seed,
        "validation": {
            "ok": True,
            "base_methods_verified": True,
            "learned_methods_verified": True,
            "learned_methods_skipped": [],
        },
        "splits": {
            "unseen": {
                "methods": {
                    name: _method(accuracy, utility)
                    for name, (accuracy, utility) in methods.items()
                }
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_analysis_uses_paired_seed_differences_and_novel_only_rates(tmp_path):
    campaign = []
    acquisition = []
    for seed, active, majority, learned in (
        (11, 0.8, 0.6, 0.7),
        (23, 0.9, 0.5, 0.6),
    ):
        campaign_path = tmp_path / f"campaign-{seed}.json"
        acquisition_path = tmp_path / f"acquisition-{seed}.json"
        _write_run(
            campaign_path,
            seed,
            {
                "active_diagnostic": (active, active - 0.2),
                "majority": (majority, majority - 0.4),
            },
        )
        _write_run(
            acquisition_path,
            seed,
            {"learned_acquisition": (learned, learned - 0.3)},
        )
        campaign.append(campaign_path)
        acquisition.append(acquisition_path)

    report = analyze_runs(
        campaign,
        acquisition,
        bootstrap_replicates=200,
        bootstrap_seed=7,
    )

    assert report["run_count"] == 2
    assert report["seeds"] == [11, 23]
    assert report["paired_active_minus_majority"]["decision_accuracy"][
        "mean"
    ] == pytest.approx(0.3)
    assert report["paired_active_minus_majority"]["utility"]["mean"] == pytest.approx(
        0.5
    )
    assert report["novel_only"]["active_diagnostic"]["decision_accuracy"][
        "mean"
    ] == pytest.approx(0.85)
    assert report["novel_only"]["learned_acquisition"]["decision_accuracy"][
        "mean"
    ] == pytest.approx(0.65)
    assert report["novel_only"]["majority"]["mechanism_macro_accuracy"][
        "mean"
    ] == pytest.approx(0.55)


def test_analysis_rejects_duplicate_or_mismatched_seeds(tmp_path):
    first = tmp_path / "campaign-a.json"
    duplicate = tmp_path / "campaign-b.json"
    acquisition = tmp_path / "acquisition.json"
    _write_run(first, 11, {"active_diagnostic": (0.8, 0.6), "majority": (0.6, 0.2)})
    _write_run(duplicate, 11, {"active_diagnostic": (0.7, 0.5), "majority": (0.5, 0.1)})
    _write_run(acquisition, 23, {"learned_acquisition": (0.7, 0.4)})

    with pytest.raises(AnalysisError, match="duplicate seed"):
        analyze_runs([first, duplicate], [acquisition])

    with pytest.raises(AnalysisError, match="seed sets differ"):
        analyze_runs([first], [acquisition])


def test_analysis_rejects_unverified_run(tmp_path):
    campaign = tmp_path / "campaign.json"
    acquisition = tmp_path / "acquisition.json"
    _write_run(campaign, 11, {"active_diagnostic": (0.8, 0.6), "majority": (0.6, 0.2)})
    _write_run(acquisition, 11, {"learned_acquisition": (0.7, 0.4)})
    payload = json.loads(campaign.read_text(encoding="utf-8"))
    payload["validation"]["ok"] = False
    campaign.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AnalysisError, match="not verified"):
        analyze_runs([campaign], [acquisition])


def test_cli_writes_deterministic_report(tmp_path):
    campaign = tmp_path / "campaign-11.json"
    acquisition = tmp_path / "acquisition-11.json"
    output = tmp_path / "summary.json"
    _write_run(campaign, 11, {"active_diagnostic": (0.8, 0.6), "majority": (0.6, 0.2)})
    _write_run(acquisition, 11, {"learned_acquisition": (0.7, 0.4)})

    assert (
        main(
            [
                "--campaign",
                str(campaign),
                "--acquisition",
                str(acquisition),
                "--output",
                str(output),
                "--bootstrap-replicates",
                "50",
            ]
        )
        == 0
    )
    first = output.read_bytes()
    assert (
        main(
            [
                "--campaign",
                str(campaign),
                "--acquisition",
                str(acquisition),
                "--output",
                str(output),
                "--bootstrap-replicates",
                "50",
            ]
        )
        == 0
    )
    assert output.read_bytes() == first
