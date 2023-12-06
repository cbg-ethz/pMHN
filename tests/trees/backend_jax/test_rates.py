import jax
import jax.numpy as jnp
import numpy.testing as npt
import pmhn._trees._backend_jax._rates as rates
import pytest

# *** _construct_log_transtion_rate ***


def test_construct_log_transition_rate_2() -> None:
    theta = 1.0 * jnp.asarray(
        [
            [1, 10],
            [100, 1000],
        ]
    )
    extended_theta = rates._extend_theta(theta)

    # We expect 1 -> 2, that is theta[2-1, 1-1] + theta[2-1, 2-1]
    path = jnp.asarray([0, 0, 1, 2])
    expected = theta[1, 0] + theta[1, 1]
    npt.assert_allclose(
        expected, rates._construct_log_transtion_rate(path, extended_theta)
    )

    path = jnp.asarray([0, 0, 2, 1])
    expected = theta[0, 0] + theta[0, 1]
    npt.assert_allclose(
        expected, rates._construct_log_transtion_rate(path, extended_theta)
    )

    path = jnp.asarray([1])
    expected = theta[0, 0]
    npt.assert_allclose(
        expected, rates._construct_log_transtion_rate(path, extended_theta)
    )

    path = jnp.asarray([2])
    expected = theta[1, 1]
    npt.assert_allclose(
        expected, rates._construct_log_transtion_rate(path, extended_theta)
    )

    path = jnp.asarray([0, 0], dtype=jnp.int32)
    expected = 0.0
    npt.assert_allclose(
        expected, rates._construct_log_transtion_rate(path, extended_theta)
    )


# *** _construct_log_U ***


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
