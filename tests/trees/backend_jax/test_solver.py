import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pmhn._trees._backend_jax._private_api as api


def test_solve_forward_substitution() -> None:
    # Define an upper triangular rate matrix Q:
    Q = 0.1 * np.asarray(
        [
            [0, 7, 5, 0],
            [0, 0, 2, 1],
            [0, 0, 0, 2],
            [0, 0, 0, 0],
        ]
    )
    Q -= Q.sum(axis=1) * np.eye(Q.shape[0])
    npt.assert_allclose(Q.sum(axis=1), np.zeros(4), atol=1e-8)

    # Now define the V matrix and solve for probabilities
    V = np.eye(Q.shape[0]) - Q
    probs_expected = np.linalg.inv(V)[0, :]

    # Now we want to have the matrix V with non-negative
    # entries and use log V:
    log_magic = jnp.log(jnp.abs(V))

    log_probs = api.logprob_forward_substitution(api.coo_matrix_from_array(log_magic))

    npt.assert_allclose(jnp.exp(log_probs), probs_expected, atol=1e-5)
