from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from perception_source_uncertainty import (
    SourceUncertaintyError,
    paired_bootstrap_interval,
    wilson_interval,
)


def test_wilson_interval_matches_twenty_pair_cells() -> None:
    assert wilson_interval(3, 20) == pytest.approx(
        [0.052368745896216595, 0.36041886474075696]
    )
    assert wilson_interval(5, 20) == pytest.approx(
        [0.11186170140766569, 0.468700877618744]
    )


def test_paired_bootstrap_preserves_identical_outcomes() -> None:
    point, interval = paired_bootstrap_interval(
        [1, 0, 1, 0],
        [1, 0, 1, 0],
        replicates=1_000,
        rng=np.random.default_rng(7),
    )
    assert point == 0.0
    assert interval == [0.0, 0.0]


def test_paired_bootstrap_rejects_mismatched_inputs() -> None:
    with pytest.raises(SourceUncertaintyError):
        paired_bootstrap_interval(
            [1, 0],
            [1],
            replicates=1_000,
            rng=np.random.default_rng(7),
        )
