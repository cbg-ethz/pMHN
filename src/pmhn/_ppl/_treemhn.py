from pmhn._trees._backend_geno import OriginalTreeMHNBackend, LoglikelihoodSingleTree

import numpy as np
import pytensor.tensor as pt

Op = pt.Op  # type: ignore


class TreeMHNLoglikelihood(Op):
    """A wrapper around the TreeMHN loglikelihood, so that
    it can be used in PyMC models.

    This operation expects the (unconstrained/log) MHN matrix
    of shape (n_genes, n_genes).
    """

    itypes = [pt.dmatrix]  # (n_genes, n_genes)
    otypes = [pt.dscalar]  # scalar, the loglikelihood

    def __init__(
        self,
        data: list[LoglikelihoodSingleTree],
        mean_sampling_time: float,
        all_mut: set[int],
        backend: OriginalTreeMHNBackend,
    ) -> None:
        print("constructor called!")
        self._data = data
        self._mean_sampling_time = mean_sampling_time
        self._all_mut = all_mut
        self._backend = backend

    def perform(self, node, inputs, outputs):
        """This is the method which is called by the operation.

        It calculates the loglikelihood.

        Note:
            The arguments and the output are PyTensor variables.
        """
        (theta,) = inputs  # Unwrap the inputs
        print(self._mean_sampling_time)
        loglikelihoods = self._backend.loglikelihood_tree_list(
            trees=self._data,
            theta=theta,
            sampling_rate=self._mean_sampling_time,
            all_mut=self._all_mut,
        )
        total_loglikelihood = np.sum(loglikelihoods)
        outputs[0][0] = np.array(
            total_loglikelihood
        )  # Wrap the log-likelihood into output
