from typing import NamedTuple

from anytree import Node
from jaxtyping import Array, Int


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
    # TODO(Pawel): NOT-IMPLEMENTED
    raise NotImplementedError
