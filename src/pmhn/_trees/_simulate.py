from typing import Union, Sequence
from anytree import Node
import numpy as np

from pmhn._trees._interfaces import Tree

def generate_valid_tree(rng, theta: np.ndarray, sampling_time: float, min_tree_size: int = 2, max_tree_size: int = 11):
    while True: 
        tree = _simulate_tree(rng, theta, sampling_time)
        if len(tree) >= min_tree_size and len(tree)<=max_tree_size:
            return tree 

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
    # TODO(Pawel): This is part of https://github.com/cbg-ethz/pMHN/issues/14
    #   Note that the sampling time is known that our `theta` entries
    #   are log-Theta entries from the paper.
    print("starting ... \n")
    theta_size=len(theta)
    node_time_map={}
    root=Node("0")
    node_time_map[root]=0
    U_current=[root]
    exit_while=False
    while len(U_current)!=0:
        U_next=[]
        for node in U_current:
            path=list(node.path)
            old_mutations=[int(node.name) for node in path]
            possible_mutations=list(set([i+1 for i in range(theta_size)]).difference(set(old_mutations)))
            for j in possible_mutations:
                new_node=Node(str(j),parent=node)
                l=np.exp(theta[j-1][j-1])
                for anc in [ancestor for ancestor in node.path if ancestor.parent is not None]:
                    l*=np.exp(theta[j-1][int(anc.name)-1])
                waiting_time=node_time_map[node]+rng.exponential(1.0/l)
                if waiting_time<sampling_time:
                    node_time_map[new_node]=waiting_time
                    U_next.append(new_node)
                    if len(node_time_map)==12:
                        exit_while=True
                        break
            if exit_while:
                break
        if exit_while:
            break
        U_current=U_next
    print("ending .. \n")
    return node_time_map


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
        generate_valid_tree(rng, theta=th, sampling_time=t_s)
        for th, t_s in zip(theta, sampling_times)
    ]

    return sampling_times, trees
