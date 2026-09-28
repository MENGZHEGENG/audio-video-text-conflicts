import numpy as np


def test_masked_observations_ignore_unobserved_values():
    from conflictbench.value_of_information import masked_observations

    observations = np.array([[1.0, 99.0, -0.5], [8.0, 2.0, -7.0]])
    observed_mask = np.array([[True, False, True], [False, True, False]])
    changed = observations.copy()
    changed[~observed_mask] = np.array([-1_000.0, 500.0, np.nan])

    expected = np.array(
        [
            [1.0, 0.0, -0.5, 1.0, 0.0, 1.0],
            [0.0, 2.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    np.testing.assert_array_equal(masked_observations(observations, observed_mask), expected)
    np.testing.assert_array_equal(masked_observations(changed, observed_mask), expected)


def test_realized_value_is_loss_reduction_without_query_cost():
    from conflictbench.value_of_information import realized_acquisition_value

    loss_before = np.array([0.8, 0.4])
    loss_after = np.array([[99.0, 0.3, 0.9], [0.2, -99.0, 0.1]])
    observed_mask = np.array([[True, False, False], [False, True, False]])

    expected = np.array([[np.nan, 0.5, -0.1], [0.2, np.nan, 0.3]])
    np.testing.assert_allclose(
        realized_acquisition_value(loss_before, loss_after, observed_mask),
        expected,
        equal_nan=True,
    )


def test_exact_budget_oracle_is_no_worse_than_a_feasible_selection():
    from conflictbench.value_of_information import select_exact_budget

    values = np.array(
        [
            [np.nan, 0.2, 0.7],
            [0.6, np.nan, 0.5],
            [-0.1, 0.1, np.nan],
            [np.nan, np.nan, -0.2],
        ]
    )
    observed_mask = np.isnan(values)
    selected = select_exact_budget(values, observed_mask, query_count=2)
    feasible = np.array(
        [
            [False, True, False],
            [False, False, False],
            [False, True, False],
            [False, False, False],
        ]
    )

    assert int(selected.sum()) == 2
    assert np.all(selected.sum(axis=1) <= 1)
    assert not np.any(selected & observed_mask)
    assert float(values[selected].sum()) >= float(values[feasible].sum())
    np.testing.assert_array_equal(
        selected,
        np.array(
            [
                [False, False, True],
                [True, False, False],
                [False, False, False],
                [False, False, False],
            ]
        ),
    )


def test_exact_budget_ties_use_example_then_modality_order():
    from conflictbench.value_of_information import select_exact_budget

    scores = np.ones((3, 3), dtype=float)
    observed_mask = np.array(
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
        ]
    )

    selected = select_exact_budget(scores, observed_mask, query_count=2)
    np.testing.assert_array_equal(
        selected,
        np.array(
            [
                [False, True, False],
                [True, False, False],
                [False, False, False],
            ]
        ),
    )


def test_random_selection_matches_reference_query_rate():
    from conflictbench.value_of_information import random_matched_selection

    observed_mask = np.array(
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
            [True, True, False],
            [False, True, True],
            [True, False, True],
        ]
    )
    reference = np.zeros_like(observed_mask)
    reference[[0, 2, 4], [1, 0, 0]] = True

    first = random_matched_selection(reference, observed_mask, seed=17)
    second = random_matched_selection(reference, observed_mask, seed=17)

    assert int(first.sum()) == int(reference.sum())
    assert np.all(first.sum(axis=1) <= 1)
    assert not np.any(first & observed_mask)
    np.testing.assert_array_equal(first, second)


def test_source_group_split_is_disjoint_and_exhaustive():
    from conflictbench.value_of_information import source_group_split

    group_ids = np.array(["a", "a", "b", "b", "c", "d", "e", "f", "g", "g", "h", "i"])
    split = source_group_split(group_ids, fractions=(0.5, 0.25, 0.25), seed=23)
    repeated = source_group_split(group_ids, fractions=(0.5, 0.25, 0.25), seed=23)

    all_indices = np.concatenate([split["train"], split["dev"], split["test"]])
    np.testing.assert_array_equal(np.sort(all_indices), np.arange(len(group_ids)))
    assert set(group_ids[split["train"]]).isdisjoint(group_ids[split["dev"]])
    assert set(group_ids[split["train"]]).isdisjoint(group_ids[split["test"]])
    assert set(group_ids[split["dev"]]).isdisjoint(group_ids[split["test"]])
    assert all(len(split[name]) > 0 for name in ("train", "dev", "test"))
    for name in ("train", "dev", "test"):
        np.testing.assert_array_equal(split[name], repeated[name])


def test_grouped_bootstrap_resamples_whole_groups_and_is_repeatable():
    from conflictbench.value_of_information import grouped_bootstrap_indices

    group_ids = np.array(["a", "a", "b", "c", "c", "c"])
    samples = grouped_bootstrap_indices(group_ids, n_resamples=12, seed=31)
    repeated = grouped_bootstrap_indices(group_ids, n_resamples=12, seed=31)

    assert len(samples) == 12
    for sample, repeated_sample in zip(samples, repeated):
        np.testing.assert_array_equal(sample, repeated_sample)
        counts = np.bincount(sample, minlength=len(group_ids))
        assert counts[0] == counts[1]
        assert counts[3] == counts[4] == counts[5]
        assert int(counts[0] + counts[2] + counts[3]) == 3
