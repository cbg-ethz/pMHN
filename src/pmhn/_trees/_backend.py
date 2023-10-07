from typing import Protocol


import numpy as np
from scipy.sparse.linalg import spsolve_triangular
from scipy.sparse import csr_matrix
from pmhn._trees._interfaces import Tree
from pmhn._trees._tree_utils import create_all_subtrees, bfs_compare
from anytree import Node


class IndividualTreeMHNBackendInterface(Protocol):
    def loglikelihood(
        self,
        tree: Tree,
        theta: np.ndarray,
    ) -> float:
        """Calculates loglikelihood `log P(tree | theta)`.

        Args:
            tree: a tree
            theta: real-valued (i.e., log-theta) matrix,
              shape (n_mutations, n_mutations)

        Returns:
            loglikelihood of the tree
        """
        raise NotImplementedError

    def gradient(
        self,
        tree: Tree,
        theta: np.ndarray,
    ) -> np.ndarray:
        """Calculates the partial derivatives of `log P(tree | theta)`
        with respect to `theta`.

        Args:
            tree: a tree
            theta: real-valued matrix,
              shape (n_mutations, n_mutatations)

        Returns:
            gradient `d log P(tree | theta) / d theta`,
              shape (n_mutations, n_mutations)
        """
        raise NotImplementedError

    def gradient_and_loglikelihood(
        self, tree: Tree, theta: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Returns the gradient and the loglikelihood.

        Note:
            This function may be faster than calling `gradient` and `loglikelihood`
            separately.
        """
        return self.gradient(tree, theta), self.loglikelihood(tree, theta)


class OriginalTreeMHNBackend(IndividualTreeMHNBackendInterface):
    def create_V_Mat(
        self, tree: Node, theta: np.ndarray, sampling_rate: float
    ) -> np.ndarray:
        """Calculates the V matrix.

        Args:
            tree: a tree
            theta: real-valued (i.e., log-theta) matrix,
              shape (n_mutations, n_mutations)
            sampling_rate: a scalar of type float
        Returns:
            the V matrix.
        """

        subtrees = create_all_subtrees(tree)
        subtrees_size = len(subtrees)
        Q = np.zeros((subtrees_size, subtrees_size))
        for i in range(subtrees_size):
            for j in range(subtrees_size):
                if i == j:
                    Q[i][j] = self.diag_entry(subtrees[i], theta)
                else:
                    Q[i][j] = self.off_diag_entry(subtrees[i], subtrees[j], theta)
        V = np.eye(subtrees_size) * sampling_rate - Q
        return V

    def diag_entry(self, tree: Node, theta: np.ndarray) -> float:
        """
        Calculates a diagonal entry of the V matrix.

        Args:
            tree: a tree
            theta: real-valued (i.e., log-theta) matrix,
              shape (n_mutations, n_mutations)

        Returns:
            the diagonal entry of the V matrix corresponding to tree
        """
        lamb_sum = 0
        n_mutations = len(theta)
        current_nodes = [tree]
        while len(current_nodes) != 0:
            next_nodes = []
            for node in current_nodes:
                tree_mutations = list(node.path) + list(node.children)
                exit_mutations = list(
                    set([i + 1 for i in range(n_mutations)]).difference(
                        set(
                            [
                                tree_mutation.name  # type: ignore
                                for tree_mutation in tree_mutations
                            ]
                        )
                    )
                )
                for mutation in exit_mutations:
                    lamb = 0
                    exit_subclone = [
                        anc.name  # type: ignore
                        for anc in node.path
                        if anc.parent is not None
                    ] + [mutation]
                    for j in exit_subclone:
                        lamb += theta[mutation - 1][j - 1]
                    lamb = np.exp(lamb)
                    lamb_sum -= lamb
                for child in node.children:
                    next_nodes.append(child)
            current_nodes = next_nodes
        return lamb_sum

    def off_diag_entry(self, tree1: Node, tree2: Node, theta: np.ndarray) -> float:
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
        self, tree: Node, theta: np.ndarray, sampling_rate: float
    ) -> float:
        """
        Calculates loglikelihood `log P(tree | theta)`.

        Args:
            tree: a tree
            theta: real-valued (i.e., log-theta) matrix,
              shape (n_mutations, n_mutations)
            sampling_rate: a scalar of type float
        Returns:
            loglikelihood of the tree
        """
        # TODO(Pawel): this is part of https://github.com/cbg-ethz/pMHN/issues/15
        #   It can be implemented in any way.
        V = self.create_V_Mat(tree=tree, theta=theta, sampling_rate=sampling_rate)
        V_size = V.shape[0]
        b = np.zeros(V_size)
        b[0] = 1
        V_transposed = V.transpose()
        V_csr = csr_matrix(V_transposed)
        x = spsolve_triangular(V_csr, b, lower=True)

        return np.log(x[V_size - 1] + 1e-10) + np.log(sampling_rate)

    def gradient(self, tree: Node, theta: np.ndarray) -> np.ndarray:
        """Calculates the partial derivatives of `log P(tree | theta)`
        with respect to `theta`.

        Args:
            tree: a tree
            theta: real-valued matrix, shape (n_mutations, n_mutatations)

        Returns:
            gradient `d log P(tree | theta) / d theta`,
              shape (n_mutations, n_mutations)
        """
        # TODO(Pawel): This is part of
        #    https://github.com/cbg-ethz/pMHN/issues/18,
        #    but it is *not* a priority.
        #    We will try to do the modelling as soon as possible,
        #    starting with a sequential Monte Carlo sampler
        #    and Metropolis transitions.
        #    Only after initial experiments
        #    (we will probably see that it's not scalable),
        #    we'll consider switching to Hamiltonian Monte Carlo,
        #    which requires gradients.
        raise NotImplementedError
