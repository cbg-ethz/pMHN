from typing import Optional, Sequence, Union

import numpy as np
from anytree import Node


def generate_valid_tree(
    rng,
    theta: np.ndarray,
    sampling_time: float,
    mean_sampling_time: float,
    min_tree_size: Optional[int] = None,
    max_tree_size: Optional[int] = None,
) -> tuple[Node, float]:
    """
    Generates a single valid tree with known sampling time.

    Args:
        rng: random number generator
        theta: real-valued (i.e., log-theta) matrix,
          shape (n_mutations, n_mutations)
        sampling_time: known sampling time
        min_tree_size: minimum size of the tree
        max_tree_size: maximum size of the tree
        mean_sampling_time: parameter for rng
    Returns:
        A valid mutation tree that meets the size constraints if specified.

    Note:
        The min_tree_size and max_tree_size parameters consider the entire tree,
        i.e the root node is included.
        To disable the size constraints, leave min_tree_size and max_tree_size as None.
        If a tree is discarded, a new sampling time is drawn.
    """

    while True:
        tree = _simulate_tree(rng, theta, sampling_time, max_tree_size)
        if (min_tree_size is None or len(tree.descendants) + 1 >= min_tree_size) and (
            max_tree_size is None or len(tree.descendants) + 1 <= max_tree_size
        ):
            return tree, sampling_time
        else:
            sampling_time = rng.exponential(scale=mean_sampling_time)


def _find_possible_mutations(old_mutations: list[int], n_mutations: int) -> list[int]:
    """
    Args:
        old_mutations: list of ancestor mutations of
            a given node (including the node itself)
        n_mutations: total number of mutations
    Returns:
        a list of possible mutations that could appear next for a given node

    Note:
        We assume that mutations are labeled with a number between
        1 and n_mutations,
        so each element in old_mutations should be in that range
        (except for the root node = mutation 0).
        If this assumption is violated, an exception is raised.

    """
    for mutation in old_mutations:
        if mutation > n_mutations or mutation < 0:
            raise ValueError(
                f"Invalid mutation {mutation} in old_mutations."
                f" It should be 0 <= mutation <= {n_mutations}."
            )

    possible_mutations = list(
        set(range(1, n_mutations + 1)).difference(set(old_mutations))
    )
    return possible_mutations


def _simulate_tree(
    rng, theta: np.ndarray, sampling_time: float, max_tree_size: Optional[int] = None
) -> Node:
    """Simulates a single tree with known sampling time.

    Args:
        rng: random number generator
        theta: real-valued (i.e., log-theta) matrix,
          shape (n_mutations, n_mutations)
        sampling_time: known sampling time
        max_tree_size: maximum size of the tree
    Returns:
        a mutation tree

    Note:
        We assume that sampling time $t_s$ is known.
        Otherwise, this is the Algorithm 1 from in
        Appendix A1 to the TreeMHN paper
        (with the difference that in the paper `Theta_{jl}`
        is used, which is `Theta_{jl} = exp( theta_{jl} )`.

        If the tree is larger than max_tree_size, the function returns immediately.
    """
    # TODO(Pawel): This is part of https://github.com/cbg-ethz/pMHN/issues/14
    #   Note that the sampling time is known that our `theta` entries
    #   are log-Theta entries from the paper.
    n_mutations = len(theta)
    node_time_map = {}
    root = Node(0)
    node_time_map[root] = 0
    U_current = [root]
    exit_while = False
    while len(U_current) != 0:
        U_next = []
        for node in U_current:
            path = list(node.path)
            old_mutations = [node.name for node in path]  # type: ignore
            possible_mutations = _find_possible_mutations(
                old_mutations=old_mutations, n_mutations=n_mutations
            )
            for j in possible_mutations:
                new_node = Node(j, parent=node)
                # Here j lies in the range of 1 to n_mutations inclusive.
                # However, Python uses 0-based indexing for arrays. Therefore,
                # we subtract 1 from j when accessing elements in the
                # log-theta matrix to correctly map the 1-indexed mutation
                # to the 0-indexed matrix position.
                lamb = theta[j - 1][j - 1]
                for anc in [
                    ancestor for ancestor in node.path if ancestor.parent is not None
                ]:
                    lamb += theta[j - 1][anc.name - 1]  # type: ignore
                lamb = np.exp(lamb)
                waiting_time = node_time_map[node] + rng.exponential(1.0 / lamb)
                if waiting_time < sampling_time:
                    node_time_map[new_node] = waiting_time
                    U_next.append(new_node)
                    if (
                        max_tree_size is not None
                        and len(node_time_map) == max_tree_size + 1
                    ):
                        exit_while = True
                        break
                # delete Node, since it is not needed
                else:
                    new_node.parent = None
            if exit_while:
                break
        if exit_while:
            break
        U_current = U_next

    return root


def simulate_trees(
    rng,
    n_points: int,
    theta: np.ndarray,
    mean_sampling_time: Union[np.ndarray, float, Sequence[float]],
    min_tree_size: Optional[int] = None,
    max_tree_size: Optional[int] = None,
) -> tuple[np.ndarray, list[Node]]:
    """Simulates a data set of trees with known sampling times.

    Args:
        n_points: number of trees to simulate.
        theta: the log-MHN matrix. Can be of shape (n_mutations, n_mutations)
            or (n_points, n_mutations, n_mutations).
        mean_sampling_time: the mean sampling time.
            Can be a float (shared between all data point)
            or an array of shape (n_points,).
            min_tree_size: minimum size of the trees
            max_tree_size: maximum size of the trees

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

    generated_trees_and_times = [
        generate_valid_tree(
            rng,
            theta=th,
            sampling_time=t_s,
            mean_sampling_time=ms,
            min_tree_size=min_tree_size,
            max_tree_size=max_tree_size,
        )
        for th, t_s, ms in zip(theta, sampling_times, mean_sampling_time)
    ]

    trees = [tree for tree, _ in generated_trees_and_times]
    sampling_times = np.array([time for _, time in generated_trees_and_times])

    return sampling_times, trees
