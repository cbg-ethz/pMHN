from typing import Optional

import numpy as np
import pytensor.tensor as pt

import pmhn._backend._learnmhn as lmhn

# TODO(Pawel): pyright reports a false positive
#   that `pt.Op` does not exist.
Op = pt.Op  # type: ignore


class _MHNLoglikelihoodGrad(Op):
    """This is a wrapper around the gradient of the loglikelihood
    with respect to the parameters.
    """

    itypes = [pt.dmatrix]
    otypes = [pt.dmatrix]

    def __init__(self, data: np.ndarray, backend: lmhn.MHNBackend) -> None:
        self._data = np.asarray(data, dtype=np.int32)
        self._backend = backend

        self._n_points = data.shape[0]

    def perform(self, node, inputs, outputs):
        (theta,) = inputs

        cast_theta = np.asarray(theta, dtype=np.float64)
        grads, _ = self._backend.gradient_and_loglikelihood(
            mutations=self._data, theta=cast_theta
        )

        # Rescale the gradients to have the gradients of the total loglikelihood
        # rather than average
        outputs[0][0] = grads * self._n_points


class MHNLoglikelihood(Op):
    """A wrapper around the MHN loglikelihood, so that
    it can be used in PyMC models.

    This operation expects the (unconstrained/log) MHN matrix
    of shape (n_genes, n_genes).
    """

    itypes = [pt.dmatrix]  # (n_genes, n_genes)
    otypes = [pt.dscalar]  # scalar, the loglikelihood

    def __init__(
        self, data: np.ndarray, backend: Optional[lmhn.MHNBackend] = None
    ) -> None:
        self._data = np.asarray(data, dtype=np.int32)
        self._backend = backend or lmhn.MHNJoblibBackend(n_jobs=-4)

        self._n_points = data.shape[0]
        self._gradop = _MHNLoglikelihoodGrad(data=data, backend=self._backend)

    def perform(self, node, inputs, outputs):
        """This is the method which is called by the operation.

        It calculates the loglikelihood.

        Note:
            The arguments and the output are PyTensor variables.
        """
        (theta,) = inputs  # Unwrap the inputs

        # Call the log-likelihood function
        _, loglike = self._backend.gradient_and_loglikelihood(
            mutations=self._data, theta=theta
        )

        outputs[0][0] = np.array(loglike)  # Wrap the log-likelihood into output

    def grad(self, inputs, g):
        (theta,) = inputs
        tangent_vector = g[0]

        return [tangent_vector * self._gradop(theta)]
