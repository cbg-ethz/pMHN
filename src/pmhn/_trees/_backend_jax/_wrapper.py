from typing import NamedTuple

import jax.numpy as jnp
from anytree import Node
from jaxtyping import Array, Int

from pmhn._trees._backend_jax._const import PADDING
from pmhn._trees._tree_utils import construct_paths_matrix


class IndexedPaths(NamedTuple):
    index: Int[Array, " K"]
    path: Int[Array, "K n_events"]

    def size(self) -> int:
        return self.index.shape[0]

    def path_length(self) -> int:
        return self.path.shape[1]


class DoublyIndexedPaths(NamedTuple):
    """
    Attrs:
        start: indices representing
          the starting nodes of transitions
        end: indices representing the ending
          nodes of the transitions.
          Use `None` if `start` should only be used.
        path: path representing the rate
          used in the transition
    """

    start: Int[Array, " K"]
    end: Int[Array, " K"]
    path: Int[Array, "K n_events"]


ExitPathsArray = Int[Array, "n_subtrees n_events"]


class WrappedTree(NamedTuple):
    """Tree wrapper encoding the paths used to
    construct the rates.

    Attrs:
        diag_paths: paths used to construct the diagonal rates.
            As on the diagonal we have sum of rates corresponding
            to different paths, the same `start` may appear several times
            (i.e., the length of this array is not equal to the number of subtrees)
        offdiag_paths: paths used to construct the off-diagonal rates.
        exit_paths: paths used to construct the exit rates (to the observed state)
            from omega
        n_genes: number of genes

    Note:
        The paths are padded with the `n_genes+1` value.
    """

    diag_paths: IndexedPaths
    offdiag_paths: DoublyIndexedPaths
    exit_paths: ExitPathsArray
    n_genes: int

    @property
    def n_subtrees(self) -> int:
        """Number of subtrees (including root-only as well
        as the given tree)."""
        return self.exit_paths.shape[0]


def _pad_trajectory(traj, total_length: int):
    t = list(traj)[1:]
    return [PADDING] * (total_length - len(t)) + t


def _construct_offdiag_paths(offdiag: dict) -> DoublyIndexedPaths:
    max_length = max(map(lambda x: len(x), offdiag.values())) - 1

    starts = []
    ends = []
    paths = []

    for (s, e), traj in offdiag.items():
        starts.append(s)
        ends.append(e)
        paths.append(_pad_trajectory(traj, total_length=max_length))

    starts = jnp.asarray(starts, dtype=int)
    ends = jnp.asarray(ends, dtype=int)

    if jnp.any(starts >= ends):
        raise ValueError("Starts should be smaller than ends.")

    return DoublyIndexedPaths(
        start=starts, end=ends, path=jnp.asarray(paths, dtype=int)
    )


def _construct_diag_paths(diag: list) -> IndexedPaths:
    # Note that this includes 0, so we subtract 1
    max_trajectory_length = max(map(lambda x: len(x), sum(diag, []))) - 1

    indices = []
    paths = []

    for i, traj_list in enumerate(diag):
        for traj in traj_list:
            indices.append(i)
            paths.append(_pad_trajectory(traj, total_length=max_trajectory_length))

    return IndexedPaths(
        index=jnp.asarray(indices, dtype=int), path=jnp.asarray(paths, dtype=int)
    )


def _construct_exit_paths(n_subtrees: int) -> ExitPathsArray:
    # TODO(Pawel): Replace with a more principled alternative.
    return jnp.zeros((n_subtrees, 1), dtype=int)


def wrap_tree(tree: Node, n_genes: int) -> tuple[WrappedTree, list[Node]]:
    """Wraps a tree into a `WrappedTree` object.

    Args:
        tree: a tree. The root should be annotated as 0 and the other
          nodes are annotated with integers from {1, ... `n_genes`} (inclusive)
        n_genes: number of genes

    Returns:
        wrapped tree, can be used to evaluate the likelihood
        list of sorted subtrees (e.g., for visualisation purposes)
    """
    # TODO(Pawel): UNTESTED
    paths_object = construct_paths_matrix(tree, n_genes=n_genes)

    return (
        WrappedTree(
            diag_paths=_construct_diag_paths(paths_object.diag),
            offdiag_paths=_construct_offdiag_paths(paths_object.offdiag),
            exit_paths=_construct_exit_paths(n_subtrees=paths_object.n_subtrees),
            n_genes=n_genes,
        ),
        paths_object.indices,
    )
