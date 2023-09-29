import numpy as np
import pmhn._trees._simulate as simulate
import pytest


def test_generate_valid_tree():
    """
    We want to test if valid trees are generated. Here we test the size requirements
    and the waiting times.
    """
    rng = np.random.default_rng()
    theta = np.array(
        [
            [-1.41, 0.00, 0.00, 4.91, 1.03, 0.00, -1.91, -0.74, -1.35, 1.48],
            [-1.12, -2.26, 0.00, 0.82, 0.00, 0.00, 1.16, 0.00, -1.62, 0.00],
            [0.00, -0.86, -2.55, 1.58, 0.00, 0.00, 1.02, -2.70, 0.00, 0.68],
            [0.00, 0.00, 0.00, -3.69, 0.00, 0.00, -0.95, 1.42, 0.00, -1.01],
            [-3.08, -1.42, -3.14, 0.00, -3.95, 3.90, -1.46, -2.00, 0.00, 2.87],
            [-2.24, 0.00, 0.00, 0.00, 0.00, -2.38, -2.13, 1.50, 0.00, 1.35],
            [0.00, 0.00, 0.00, 0.00, 1.52, 0.00, -1.79, 0.00, 0.00, 0.00],
            [1.69, 0.76, 0.00, 1.29, 1.73, -0.82, -1.38, -4.65, 0.92, 0.00],
            [-1.22, 0.00, 0.00, 0.00, 0.65, -1.14, 0.00, 0.00, -3.25, 0.00],
            [0.97, 1.75, 0.00, -3.66, -1.28, 0.00, 1.66, 0.00, 0.00, -3.03],
        ]
    )
    mean_sampling_time = 1.0
    sampling_time = rng.exponential(scale=mean_sampling_time)
    min_tree_size = 2
    max_tree_size = 11

    for _ in range(10000):
        tree, sampling_time = simulate.generate_valid_tree(
            rng, theta, sampling_time, mean_sampling_time, min_tree_size, max_tree_size
        )

        assert min_tree_size <= len(tree) <= max_tree_size

        for node, time in tree.items():
            assert time < sampling_time


def test_generate_tree_no_size_constraints():
    """
    Here we test the case where there are no size requirements.
    """

    rng = np.random.default_rng()
    theta = np.array(
        [
            [-1.41, 0.00, 0.00, 4.91, 1.03, 0.00, -1.91, -0.74, -1.35, 1.48],
            [-1.12, -2.26, 0.00, 0.82, 0.00, 0.00, 1.16, 0.00, -1.62, 0.00],
            [0.00, -0.86, -2.55, 1.58, 0.00, 0.00, 1.02, -2.70, 0.00, 0.68],
            [0.00, 0.00, 0.00, -3.69, 0.00, 0.00, -0.95, 1.42, 0.00, -1.01],
            [-3.08, -1.42, -3.14, 0.00, -3.95, 3.90, -1.46, -2.00, 0.00, 2.87],
            [-2.24, 0.00, 0.00, 0.00, 0.00, -2.38, -2.13, 1.50, 0.00, 1.35],
            [0.00, 0.00, 0.00, 0.00, 1.52, 0.00, -1.79, 0.00, 0.00, 0.00],
            [1.69, 0.76, 0.00, 1.29, 1.73, -0.82, -1.38, -4.65, 0.92, 0.00],
            [-1.22, 0.00, 0.00, 0.00, 0.65, -1.14, 0.00, 0.00, -3.25, 0.00],
            [0.97, 1.75, 0.00, -3.66, -1.28, 0.00, 1.66, 0.00, 0.00, -3.03],
        ]
    )
    mean_sampling_time = 1.0
    sampling_time = rng.exponential(scale=mean_sampling_time)

    for _ in range(10000):
        tree, sampling_time = simulate.generate_valid_tree(
            rng, theta, sampling_time, mean_sampling_time=mean_sampling_time
        )

        for node, time in tree.items():
            assert time < sampling_time


def test_generate_tree_no_min_size_constraint():
    """
    Here we test the case where min_tree_size is None but max_tree_size is specified.
    """
    rng = np.random.default_rng()
    theta = np.array(
        [
            [-1.41, 0.00, 0.00, 4.91, 1.03, 0.00, -1.91, -0.74, -1.35, 1.48],
            [-1.12, -2.26, 0.00, 0.82, 0.00, 0.00, 1.16, 0.00, -1.62, 0.00],
            [0.00, -0.86, -2.55, 1.58, 0.00, 0.00, 1.02, -2.70, 0.00, 0.68],
            [0.00, 0.00, 0.00, -3.69, 0.00, 0.00, -0.95, 1.42, 0.00, -1.01],
            [-3.08, -1.42, -3.14, 0.00, -3.95, 3.90, -1.46, -2.00, 0.00, 2.87],
            [-2.24, 0.00, 0.00, 0.00, 0.00, -2.38, -2.13, 1.50, 0.00, 1.35],
            [0.00, 0.00, 0.00, 0.00, 1.52, 0.00, -1.79, 0.00, 0.00, 0.00],
            [1.69, 0.76, 0.00, 1.29, 1.73, -0.82, -1.38, -4.65, 0.92, 0.00],
            [-1.22, 0.00, 0.00, 0.00, 0.65, -1.14, 0.00, 0.00, -3.25, 0.00],
            [0.97, 1.75, 0.00, -3.66, -1.28, 0.00, 1.66, 0.00, 0.00, -3.03],
        ]
    )
    mean_sampling_time = 1.0
    sampling_time = rng.exponential(scale=mean_sampling_time)

    max_tree_size = 8

    for _ in range(10000):
        tree, sampling_time = simulate.generate_valid_tree(
            rng,
            theta,
            sampling_time,
            mean_sampling_time=mean_sampling_time,
            max_tree_size=max_tree_size,
        )
        assert len(tree) <= max_tree_size

        for node, time in tree.items():
            assert time < sampling_time


def test_generate_tree_no_max_size_constraint():
    """
    Here we test the case where max_tree_size is None but min_tree_size is specified.
    """
    rng = np.random.default_rng()
    theta = np.array(
        [
            [-1.41, 0.00, 0.00, 4.91, 1.03, 0.00, -1.91, -0.74, -1.35, 1.48],
            [-1.12, -2.26, 0.00, 0.82, 0.00, 0.00, 1.16, 0.00, -1.62, 0.00],
            [0.00, -0.86, -2.55, 1.58, 0.00, 0.00, 1.02, -2.70, 0.00, 0.68],
            [0.00, 0.00, 0.00, -3.69, 0.00, 0.00, -0.95, 1.42, 0.00, -1.01],
            [-3.08, -1.42, -3.14, 0.00, -3.95, 3.90, -1.46, -2.00, 0.00, 2.87],
            [-2.24, 0.00, 0.00, 0.00, 0.00, -2.38, -2.13, 1.50, 0.00, 1.35],
            [0.00, 0.00, 0.00, 0.00, 1.52, 0.00, -1.79, 0.00, 0.00, 0.00],
            [1.69, 0.76, 0.00, 1.29, 1.73, -0.82, -1.38, -4.65, 0.92, 0.00],
            [-1.22, 0.00, 0.00, 0.00, 0.65, -1.14, 0.00, 0.00, -3.25, 0.00],
            [0.97, 1.75, 0.00, -3.66, -1.28, 0.00, 1.66, 0.00, 0.00, -3.03],
        ]
    )
    mean_sampling_time = 1.0
    sampling_time = rng.exponential(scale=mean_sampling_time)

    min_tree_size = 3

    for _ in range(10000):
        tree, sampling_time = simulate.generate_valid_tree(
            rng,
            theta,
            sampling_time,
            mean_sampling_time=mean_sampling_time,
            min_tree_size=min_tree_size,
        )

        assert len(tree) >= min_tree_size

        for node, time in tree.items():
            assert time < sampling_time


def test_find_possible_mutations_normal():
    """
    We want to test if the possible_mutations list is correct.
    """
    old_mutations = [7, 2, 5, 9]
    n_mutations = 10
    possible_mutations = simulate._find_possible_mutations(old_mutations, n_mutations)

    assert possible_mutations == [1, 3, 4, 6, 8, 10]


def test_find_possible_mutations_edge():
    """
    We want to test if the possible_mutations list is correct. old_mutations with mutations on the edge.
    """

    old_mutations = [1, 10]
    n_mutations = 10
    possible_mutations = simulate._find_possible_mutations(old_mutations, n_mutations)

    assert possible_mutations == [i for i in range(2, 10)]


def test_find_possible_mutations_except_positive():
    """
    We want to test if an exception is correctly thrown when the old_mutations list is invalid (mutation number too large).
    """
    old_mutations = [212, 1, 3, 7]
    n_mutations = 10
    with pytest.raises(ValueError) as excinfo:
        simulate._find_possible_mutations(old_mutations, n_mutations)

    assert (
        str(excinfo.value)
        == "Invalid mutation 212 in old_mutations. It should be 0 <= mutation <= 10."
    )


def test_find_possible_mutations_except_negative():
    """
    We want to test if an exception is correctly thrown when the old_mutations list is invalid (mutation number too small).
    """
    old_mutations = [-23, 0, 5]
    n_mutations = 10
    with pytest.raises(ValueError) as excinfo:
        simulate._find_possible_mutations(old_mutations, n_mutations)

    assert (
        str(excinfo.value)
        == "Invalid mutation -23 in old_mutations. It should be 0 <= mutation <= 10."
    )
