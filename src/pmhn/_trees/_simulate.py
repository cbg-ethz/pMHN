from typing import Union, Sequence

import numpy as np

from pmhn._trees._interfaces import Tree


def _simulate_tree(
    rng,
    theta: np.ndarray,
    sampling_time: float,
) -> Tree:
    """Simulates a single tree with known sampling time.

    Args:
        rng: random number generator
        theta: real-valued (i.e., log-theta) matrix,
          shape (n_mutations, n_mutations)
        sampling_time: known sampling time

    Returns:
        a mutation tree

    Note:
        We assume that sampling time $t_s$ is known.
        Otherwise, this is the Algorithm 1 from in
        Appendix A1 to the TreeMHN paper
        (with the difference that in the paper `Theta_{jl}`
        is used, which is `Theta_{jl} = exp( theta_{jl} )`.
    """
    # TODO(Laurenz): This implementation is missing and is a priority.
    #   Note that the sampling time is known that our `theta` entries
    #   are log-Theta entries from the paper.
    raise NotImplementedError


def simulate_trees(
    rng,
    n_points: int,
    theta: np.ndarray,
    mean_sampling_time: Union[np.ndarray, float, Sequence[float]],
) -> tuple[np.ndarray, list[Tree]]:
    """Simulates a data set of trees with known sampling times.

    Args:
        n_points: number of trees to simulate.
        theta: the log-MHN matrix. Can be of shape (n_mutations, n_mutations)
            or (n_points, n_mutations, n_mutations).
        mean_sampling_time: the mean sampling time.
            Can be a float (shared between all data point)
            or an array of shape (n_points,).

    Returns:
        sampling times, shape (n_points,)
        sampled trees, list of length `n_points`
    """
    if n_points < 1:
        raise ValueError("n_trees must be at least 1")

    assert len(theta.shape) in {
        2,
        3,
    }, "Theta should have shape (m, m) or (n_points, m, m)."

    # Make sure mean_sampling_time is an array of shape (n_points,)
    if isinstance(mean_sampling_time, float):
        mean_sampling_time = np.full(n_points, fill_value=mean_sampling_time)
    else:
        mean_sampling_time = np.asarray(mean_sampling_time)

    assert (
        len(mean_sampling_time) == n_points
    ), "mean_sampling_time should have length n_points."

    # Make sure theta has shape (n_points, n, n)
    if len(theta.shape) == 2:
        theta = np.asarray([theta for _ in range(n_points)])

    assert theta.shape[0] == n_points, "Theta should have shape (n_points, n, n)."
    assert theta.shape[1] == theta.shape[2], "Each theta should be square."

    sampling_times = rng.exponential(scale=mean_sampling_time, size=n_points)

    trees = [
        _simulate_tree(rng, theta=th, sampling_time=t_s)
        for th, t_s in zip(theta, sampling_times)
    ]

    return sampling_times, trees
