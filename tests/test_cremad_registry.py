"""Tests for label-independent CREMA-D actor role allocation."""

from conflictbench.cremad_registry import ROLE_COUNTS
from conflictbench.cremad_registry import allocate_actor_roles


def test_allocate_actor_roles_is_complete_disjoint_and_stable() -> None:
    actors = {str(1000 + index) for index in range(1, 92)}
    first = allocate_actor_roles(actors)
    assert allocate_actor_roles(set(reversed(sorted(actors)))) == first
    assert {role: len(groups) for role, groups in first.items()} == ROLE_COUNTS
    assert set().union(*[set(groups) for groups in first.values()]) == actors
