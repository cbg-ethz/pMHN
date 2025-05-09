import numpy as np
from anytree import LevelOrderGroupIter, Node

from pmhn._trees._tree_utils import bfs_compare, create_all_subtrees


class TreeWrapper:
    """A wrapper for a tree which stores all subtrees."""

    def __init__(self, tree: Node):
        self._subtrees_dict: dict[Node, int] = create_all_subtrees(tree)


class OriginalTreeMHNBackend:
    def __init__(self, jitter: float = 1e-10):
        self._jitter: float = jitter

    def _diag_entry(self, tree: Node, theta: np.ndarray, all_mut: set[int]) -> float:
        """
        Calculates a diagonal entry of the V matrix.

        Args:
            tree: a tree
            theta: real-valued (i.e., log-theta) matrix,
              shape (n_mutations, n_mutations)

            all_mut: set containing all possible mutations

        Returns:
            the diagonal entry of the V matrix corresponding to tree
        """
        lamb_sum = 0

        for level in LevelOrderGroupIter(tree):
            for node in level:
                tree_mutations = {n.name for n in node.path}.union(
                    {c.name for c in node.children}
                )
                exit_mutations = set(all_mut).difference(tree_mutations)

                for mutation in exit_mutations:
                    lamb = 0
                    exit_subclone = {
                        anc.name for anc in node.path if anc.parent is not None
                    }.union({mutation})

                    for j in exit_subclone:
                        lamb += theta[mutation - 1][j - 1]
                    lamb = np.exp(lamb)
                    lamb_sum -= lamb

        return lamb_sum

    def _off_diag_entry(self, tree1: Node, tree2: Node, theta: np.ndarray) -> float:
        """
        Calculates an off-diagonal entry of the V matrix.

        Args:
            tree1: the first tree
            tree2: the second tree
            theta: real-valued (i.e., log-theta) matrix,
              shape (n_mutations, n_mutations)

        Returns:
            the off-diagonal entry of the V matrix corresponding to tree1 and tree2
        """
        exit_node = bfs_compare(tree1, tree2)
        lamb = 0
        if exit_node is None:
            return lamb
        else:
            for j in [
                node.name  # type: ignore
                for node in exit_node.path
                if node.parent is not None
            ]:
                lamb += theta[exit_node.name - 1][j - 1]
            lamb = np.exp(lamb)
            return float(lamb)

    def loglikelihood(
        self, tree_wrapper: TreeWrapper, theta: np.ndarray, sampling_rate: float
    ) -> float:
        """Calculates loglikelihood `log P(tree | theta)`.

        Args:
            tree: a wrapper storing a tree (and its subtrees)
            theta: real-valued (i.e., log-theta) matrix,
              shape (n_mutations, n_mutations)
            sampling_rate: a scalar representing sampling rate

        Returns:
            loglikelihood of the tree
        """
        subtrees_size = len(tree_wrapper._subtrees_dict)
        x = np.zeros(subtrees_size)
        x[0] = 1
        n_mutations = len(theta)
        all_mut = set(i + 1 for i in range(n_mutations))
        for i, (subtree_i, subtree_size_i) in enumerate(
            tree_wrapper._subtrees_dict.items()
        ):
            V_col = {}
            V_diag = 0.0
            for j, (subtree_j, subtree_size_j) in enumerate(
                tree_wrapper._subtrees_dict.items()
            ):
                if subtree_size_i - subtree_size_j == 1:
                    V_col[j] = -self._off_diag_entry(subtree_j, subtree_i, theta)
                elif i == j:
                    V_diag = sampling_rate - self._diag_entry(subtree_i, theta, all_mut)
            for index, val in V_col.items():
                x[i] -= val * x[index]
            x[i] /= V_diag

        return np.log(x[-1] + self._jitter) + np.log(sampling_rate)
