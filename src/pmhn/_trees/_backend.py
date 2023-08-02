from typing import Protocol


import numpy as np


from pmhn._trees._interfaces import Tree


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
        # TODO(Pawel): this is part of https://github.com/cbg-ethz/pMHN/issues/15
        #   It can be implemented in any way.
        raise NotImplementedError

    def gradient(self, tree: Tree, theta: np.ndarray) -> np.ndarray:
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
