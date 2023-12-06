import jax
import jax.numpy as jnp
import numpy.testing as npt
import pmhn._trees._backend_jax._rates as rates
import pytest


@pytest.mark.parametrize("log_tau", [0.0, -0.3, 0.5])
@pytest.mark.parametrize("seed", [0, 1])
def test_construct_log_U(log_tau: float, seed: int) -> None:
    omega = jax.random.normal(key=jax.random.PRNGKey(seed), shape=(3 + seed,))
    paths = jnp.asarray(
        [
            [0, 0, 1],
            [0, 0, 2],
            [0, 2, 1],
            [0, 1, 2],
            [0, 1, 3],
            [1, 3, 2],
        ]
    )

    expected = (
        jnp.asarray(
            [
                omega[0],
                omega[1],
                omega[1] + omega[0],
                omega[0] + omega[1],
                omega[0] + omega[2],
                omega[0] + omega[2] + omega[1],
            ]
        )
        - log_tau
    )

    npt.assert_allclose(
        expected,
        rates._construct_log_U(
            paths, extended_omega=rates._extend_omega(omega), log_tau=log_tau
        ),
    )
