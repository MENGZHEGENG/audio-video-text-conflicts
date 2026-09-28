"""Small, source-independent checks for CREMA-D metadata parsing."""

import pytest

from conflictbench.cremad_preflight import parse_vote_set


def test_parse_vote_set_keeps_singleton_and_tied_votes() -> None:
    assert parse_vote_set("H") == ("H",)
    assert parse_vote_set("A:D") == ("A", "D")


@pytest.mark.parametrize("value", ["", "A::D", "A:A", "X"])
def test_parse_vote_set_rejects_malformed_votes(value: str) -> None:
    with pytest.raises(ValueError, match="label values"):
        parse_vote_set(value)
