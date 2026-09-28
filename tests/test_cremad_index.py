from conflictbench.cremad_preflight import parse_vote_set


def test_cremad_vote_sets_preserve_ties_for_sensitivity_analysis() -> None:
    assert parse_vote_set("A:D") == ("A", "D")
    assert parse_vote_set("H") == ("H",)
