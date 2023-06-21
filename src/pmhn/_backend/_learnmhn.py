from typing import cast, Callable, Protocol

import mhn
import numpy as np
import joblib


class MHNBackend(Protocol):
    """A backend for learning the MHN model.

    All implementations of this interface must be able to compute
    the gradient and the loglikelihood of a given set of mutations
    and theta (log-MHN) matrix.
    """

    def gradient_and_loglikelihood(
        self, mutations: np.ndarray, theta: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Compute the gradient and the loglikelihood of a given set of mutations

        Args:
            mutations, shape (n_patients, n_genes)
            theta: log-MHN matrix, shape (n_genes, n_genes)

        Returns:
            gradient of the loglikelihood with respect to theta,
              shape (n_genes, n_genes)
            loglikelihood of the given mutations, float
        """
        ...


def _cast_theta(theta: np.ndarray) -> np.ndarray:
    return np.asarray(theta, dtype=np.float64)


def _cast_mutations(mutations: np.ndarray) -> np.ndarray:
    return np.asarray(mutations, dtype=np.int32)


DEFAULT_N_JOBS: int = 4


def _get_function_for_theta(
    theta: np.ndarray,
) -> Callable[[np.ndarray], tuple[np.ndarray, float]]:
    def helper(mvec: np.ndarray) -> tuple[np.ndarray, float]:
        container = mhn.ssr.state_containers.StateContainer(mvec.reshape((1, -1)))
        return mhn.ssr.state_space_restriction.cython_gradient_and_score(
            theta, container
        )

    return helper


class MHNJoblibBackend(MHNBackend):
    """Calculates the gradient and the loglikelihood
    by using multiple processes via Joblib, sending
    them individual patient data."""

    def __init__(self, n_jobs: int = DEFAULT_N_JOBS) -> None:
        self._pool = joblib.Parallel(n_jobs=n_jobs)

    def gradient_and_loglikelihood(
        self, mutations: np.ndarray, theta: np.ndarray
    ) -> tuple[np.ndarray, float]:
        theta = _cast_theta(theta)
        mutations = _cast_mutations(mutations)

        fn = _get_function_for_theta(theta)

        grads_and_scores = self._pool(joblib.delayed(fn)(state) for state in mutations)
        grads_and_scores = cast(list, grads_and_scores)

        grads = [x[0] for x in grads_and_scores]
        scores = [x[1] for x in grads_and_scores]

        return np.sum(grads, axis=0), np.sum(scores)


class MHNCythonBackend(MHNBackend):
    """A simple wrapper around the Cython implementation
    of the gradient and loglikelihood."""

    @staticmethod
    def gradient_and_loglikelihood(
        mutations: np.ndarray, theta: np.ndarray
    ) -> tuple[np.ndarray, float]:
        theta = _cast_theta(theta)
        mutations = _cast_mutations(mutations)

        container = mhn.ssr.state_containers.StateContainer(mutations)
        grad_, s_ = mhn.ssr.state_space_restriction.cython_gradient_and_score(
            theta, container
        )
        # Note that s is the total loglikelihood *divided by* the number of patients,
        # so we multiply it again
        n = len(mutations)
        return n * grad_, n * s_
