from typing import Optional
from typing import cast
import numpy as np
from pmhn._trees._tree_utils_geno import create_mappings
from anytree import Node
from joblib import Parallel, delayed


class TreeWrapperCode:
    """Tree wrapper using smart encoding of subtrees."""

    def __init__(self, tree: Node) -> None:
        self._genotype_subtree_node_map: dict[
            tuple[tuple[Node, int], ...], tuple[int, int]
        ]
        self._index_subclone_map: dict[int, tuple[int, ...]]

        (
            self._genotype_subtree_node_map,
            self._index_subclone_map,
        ) = create_mappings(tree)


class TreeMHNBackendCode:
    def __init__(self, jitter: float = 1e-10) -> None:
        self._jitter: float = jitter

    def _diag_entry(
        self,
        tree_wrapper: TreeWrapperCode,
        genotype: tuple[tuple[Node, int], ...],
        theta: np.ndarray,
        all_mut: set[int],
    ) -> float:
        """Calculates a diagonal entry of the V matrix.

        Args:
            tree: a tree wrappper
            genotype: the genotype of a subtree
            theta: real-valued (i.e., log-theta) matrix,
              shape (n_mutations, n_mutations)
            all_mut: a set containing all possible mutations
        Returns:
            the diagonal entry of the V matrix corresponding to
            genotype
        """
        lamb_sum = 0
        for i, (node, val) in enumerate(genotype):
            if val:
                lineage = tree_wrapper._index_subclone_map[i]
                lineage = list(lineage)
                tree_mutations = set(lineage + [c.name for c in node.children])

                exit_mutations = all_mut.difference(tree_mutations)

                for mutation in exit_mutations:
                    lamb = 0
                    lamb += theta[mutation - 1][mutation - 1]
                    for j in lineage:
                        if j != 0:
                            lamb += theta[mutation - 1][j - 1]
                    lamb = np.exp(lamb)
                    lamb_sum -= lamb
        return lamb_sum

    def find_single_difference(
        self, arr1: np.ndarray, arr2: np.ndarray
    ) -> Optional[int]:
        """
        Checks if two binary arrays of equal size differ in only one entry.
        If so, the index of the differing entry is returned, otherwise None.

        Args:
            arr1: the first array
            arr2: the second array
        Returns:
            the index of the differing entry if there's
            a single difference, otherwise None.
        """
        differing_indices = np.nonzero(np.bitwise_xor(arr1, arr2))[0]

        return differing_indices[0] if len(differing_indices) == 1 else None

    def _off_diag_entry(
        self,
        tree_wrapper: TreeWrapperCode,
        genotype_i: np.ndarray,
        genotype_j: np.ndarray,
        theta: np.ndarray,
    ) -> float:
        """
        Calculates an off-diagonal entry of the V matrix.

        Args:
            tree: the original tree
            genotype_i: the genotype of a subtree
            genotype_j: the genotype of another subtree
            theta: real-valued (i.e., log-theta) matrix,
              shape (n_mutations, n_mutations)
        Returns:
            an off-diagonal entry of the V matrix corresponding to
            the genotype_i and genotype_j
        """
        index = self.find_single_difference(genotype_i, genotype_j)
        if index is None:
            return 0
        else:
            lamb = 0
            lineage = tree_wrapper._index_subclone_map[index]
            exit_mutation = lineage[-1]
            for mutation in lineage:
                if mutation != 0:
                    lamb += theta[exit_mutation - 1][mutation - 1]
            lamb = np.exp(lamb)
            return float(lamb)

    def loglikelihood(
        self,
        tree_wrapper: TreeWrapperCode,
        theta: np.ndarray,
        sampling_rate: float,
        all_mut: set[int],
    ) -> float:
        """
        Calculates loglikelihood `log P(tree | theta)`.

        Args:
            tree: a tree
            theta: real-valued (i.e., log-theta) matrix,
              shape (n_mutations, n_mutations)
            sampling_rate: a scalar of type float
            all_mut: a set containing all possible mutations

        Returns:
            the loglikelihood of tree
        """
        subtrees_size = len(tree_wrapper._genotype_subtree_node_map)
        x = np.zeros(subtrees_size)
        x[0] = 1
        genotype_lists = []
        for genotype in tree_wrapper._genotype_subtree_node_map.keys():
            genotype_lists.append(np.array([item[1] for item in genotype]))
        for genotype_i, (
            i,
            subtree_size_i,
        ) in tree_wrapper._genotype_subtree_node_map.items():
            V_col = []
            V_diag = 0.0
            for j, subtree_size_j in tree_wrapper._genotype_subtree_node_map.values():
                if subtree_size_i - subtree_size_j == 1:
                    V_col.append(
                        (
                            j,
                            -self._off_diag_entry(
                                tree_wrapper,
                                genotype_lists[j],
                                genotype_lists[i],
                                theta,
                            ),
                        )
                    )
                elif i == j:
                    V_diag = sampling_rate - self._diag_entry(
                        tree_wrapper, genotype_i, theta, all_mut
                    )
            for index, val in V_col:
                x[i] -= val * x[index]
            x[i] /= V_diag
        return np.log(x[-1] + self._jitter) + np.log(sampling_rate)

    def loglikelihood_tree_list(
        self,
        trees: list[TreeWrapperCode],
        theta: np.ndarray,
        sampling_rate: float,
        all_mut: set[int],
    ) -> list[float]:
        """
        Calculates the loglikelihood `log P(tree | theta)` for each tree in the list.

        Args:
            trees: a list of trees
            theta: real-valued (i.e., log-theta) matrix,
            shape (n_mutations, n_mutations)
            sampling_rate: a scalar of type float
            all_mut: a set containing all possible mutations
        Returns:
            a list of loglikelihoods, one for each tree
        """
        loglikelihoods = cast(
            list[float],
            Parallel(n_jobs=-1)(
                delayed(self.loglikelihood)(tree, theta, sampling_rate, all_mut)
                for tree in trees
            ),
        )
        return loglikelihoods
