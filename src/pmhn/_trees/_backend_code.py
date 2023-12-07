import numpy as np
from pmhn._trees._tree_utils_geno import create_mappings
from anytree import Node


class TreeWrapperCode:
    """Tree wrapper using smart encoding of subtrees."""

    def __init__(self, tree: Node) -> None:
        self._genotype_subtree_node_map: dict[tuple[tuple[Node, int], ...], int]
        self._genotype_list_subtree_map: dict[tuple[int, ...], int]
        self._index_subclone_map: dict[int, tuple[int, ...]]
        self._subclone_index_map: dict[tuple[int, ...], int]

        (
            self._genotype_subtree_node_map,
            self._genotype_list_subtree_map,
            self._index_subclone_map,
            self._subclone_index_map,
        ) = create_mappings(tree)


class TreeMHNBackendCode:
    def __init__(self, jitter: float = 1e-10) -> None:
        self._jitter: float = jitter

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
        subclone_lamb_map = {}

        for i, subclone in enumerate(tree_wrapper._subclone_index_map.keys()):
            lamb = 0
            last_mut = subclone[-1]
            for mutation in subclone[1:]:
                lamb += theta[last_mut - 1][mutation - 1]
            lamb = np.exp(lamb)
            subclone_lamb_map[i] = lamb
        exit_lamb_map = {}
        for i, (node, val) in enumerate(
            list(tree_wrapper._genotype_subtree_node_map.keys())[-1]
        ):
            lineage = tree_wrapper._index_subclone_map[i]
            lineage = list(lineage)
            tree_mutations = set(lineage + [c.name for c in node.children])

            exit_mutations = all_mut.difference(tree_mutations)

            for mutation in exit_mutations:
                lamb = 0
                lamb += theta[mutation - 1][mutation - 1]
                for j in lineage[1:]:
                    lamb += theta[mutation - 1][j - 1]
                lamb = np.exp(lamb)
                exit_lamb_map[tuple(lineage + [mutation])] = lamb

        V_old = np.zeros(subtrees_size)
        V_old[0] = -1.0
        V_new = np.zeros(subtrees_size)
        for genotype, index in tree_wrapper._genotype_subtree_node_map.items():
            x = 0.0
            genotype_list = [item[1] for item in genotype]
            for i, (node, val) in enumerate(genotype):
                if val:
                    lineage = tree_wrapper._index_subclone_map[i]
                    lineage = list(lineage)
                    tree_mutations = set(lineage + [c.name for c in node.children])

                    exit_mutations = all_mut.difference(tree_mutations)

                    for mutation in exit_mutations:
                        subclone_index = tree_wrapper._subclone_index_map.get(
                            tuple(lineage + [mutation])
                        )
                        if subclone_index is None:
                            lamb = exit_lamb_map[tuple(lineage + [mutation])]
                            V_new[index] += lamb
                        else:
                            genotype_list[subclone_index] = 1
                            ind = tree_wrapper._genotype_list_subtree_map.get(
                                tuple(genotype_list)
                            )

                            lamb = subclone_lamb_map[subclone_index]
                            V_new[ind] = -lamb
                            V_new[index] += lamb
                            genotype_list[subclone_index] = 0
            V_new[index] += sampling_rate

            x = -V_old[index] / V_new[index]
            if index == subtrees_size - 1:
                return np.log(x + self._jitter) + np.log(sampling_rate)
            V_old += V_new * x
            V_new = np.zeros_like(V_new)

        return 0.0

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
        loglikelihoods = []
        for i, tree in enumerate(trees):
            loglikelihoods.append(
                self.loglikelihood(tree, theta, sampling_rate, all_mut)
            )
        return loglikelihoods
