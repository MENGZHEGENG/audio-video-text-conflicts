from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1]
RELEASE_ROOT = Path(__file__).parents[1]
SCRIPTS = RELEASE_ROOT / "scripts"
REPRODUCIBILITY_SRC = RELEASE_ROOT / "src"
sys.path[:0] = [str(SCRIPTS), str(REPRODUCIBILITY_SRC)]

from mosei_cluster_uncertainty import (
    ClusterUncertaintyError,
    cluster_bootstrap,
    video_id,
)


def test_video_id_uses_official_grouping_convention() -> None:
    assert video_id("abc[17]") == "abc"
    assert video_id("abc_17") == "abc"
    assert video_id("abc") == "abc"


def test_cluster_bootstrap_preserves_points_and_paired_identity() -> None:
    result = cluster_bootstrap(
        {
            "threshold": [1, 0, 1, 1],
            "majority": [1, 0, 1, 1],
            "weighted": [0, 0, 1, 1],
        },
        ["v1[0]", "v1[1]", "v2[0]", "v3[0]"],
        replicates=1_000,
        seed=7,
    )
    assert result["utterances"] == 4
    assert result["video_clusters"] == 3
    assert result["methods"]["threshold"]["point"] == 0.75
    assert result["paired"]["threshold_minus_majority"] == {
        "point": 0.0,
        "video_cluster_bootstrap_interval_95": [0.0, 0.0],
    }


def test_cluster_bootstrap_rejects_nonbinary_outcomes() -> None:
    with pytest.raises(ClusterUncertaintyError):
        cluster_bootstrap(
            {"threshold": [0, 2]},
            ["v1[0]", "v2[0]"],
            replicates=100,
            seed=1,
        )
