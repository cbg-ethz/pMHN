import numpy as np
import pytensor.tensor as pt

import pmhn._backend._learnmhn as lmhn

# TODO(Pawel): pyright reports a false positive
#   that `pt.Op` does not exist.
Op = pt.Op  # type: ignore


class _PersonalisedMHNLoglikelihoodGrad(Op):
    """This is a wrapper around the gradient of the loglikelihood
    with respect to the parameters.
    """

    itypes = [pt.dtensor3]
    otypes = [pt.dtensor3]

    def __init__(
        self, data: np.ndarray, backend: lmhn._PersonalisedMHNJoblibBackend
    ) -> None:
        self._data = np.asarray(data, dtype=np.int32)
        self._backend = backend

    def perform(self, node, inputs, outputs):
        (thetas,) = inputs

        cast_thetas = np.asarray(thetas, dtype=np.float64)
        grads, _ = self._backend.gradient_and_loglikelihood(
            mutations=self._data, thetas=cast_thetas
        )

        outputs[0][0] = grads


class PersonalisedMHNLoglikelihood(Op):
    """A wrapper around the MHN loglikelihood, so that
    it can be used in PyMC models.

    This operation expects the (unconstrained/log) MHN matrix
    of shape (n_genes, n_genes).
    """

    itypes = [pt.dtensor3]  # (n_patients, n_genes, n_genes)
    otypes = [pt.dscalar]  # scalar, the loglikelihood

    def __init__(self, data: np.ndarray, n_jobs: int = 4) -> None:
        self._data = np.asarray(data, dtype=np.int32)
        self._backend = lmhn._PersonalisedMHNJoblibBackend(n_jobs=n_jobs)

        self._gradop = _PersonalisedMHNLoglikelihoodGrad(
            data=data, backend=self._backend
        )

    def perform(self, node, inputs, outputs):
        """This is the method which is called by the operation.

        It calculates the loglikelihood.

        Note:
            The arguments and the output are PyTensor variables.
        """
        (thetas,) = inputs  # Unwrap the inputs

        # Call the log-likelihood function
        _, loglike = self._backend.gradient_and_loglikelihood(
            mutations=self._data, thetas=thetas
        )
        outputs[0][0] = np.array(loglike)  # Wrap the log-likelihood into output

    def grad(self, inputs, g):
        (thetas,) = inputs
        tangent_vector = g[0]

        return [tangent_vector * self._gradop(thetas)]
