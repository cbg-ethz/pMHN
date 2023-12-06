import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pmhn._trees._backend_jax._private_api as api
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


# *** _construct_log_exit_rate ***


def test_construct_log_exit_rate_3() -> None:
    omega = jnp.asarray([1, 10, 100])
    extended_omega = rates._extend_omega(omega)

    path = jnp.asarray([0, 0, 1, 2])
    expected = omega[0] + omega[1]
    npt.assert_allclose(expected, rates._construct_log_exit_rate(path, extended_omega))

    path = jnp.asarray([0, 3, 2, 1])
    expected = omega[0] + omega[1] + omega[2]
    npt.assert_allclose(expected, rates._construct_log_exit_rate(path, extended_omega))

    path = jnp.asarray([0])
    npt.assert_allclose(0.0, rates._construct_log_exit_rate(path, extended_omega))


# *** _construct_log_Q_offdiag ***


@pytest.mark.parametrize("n_genes", [4, 5, 21])
def test_construct_log_Q_offdiag(n_genes: int, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    paths = api.DoublyIndexedPaths(
        start=jnp.arange(7),
        end=jnp.arange(8, 35, 2)[:7],
        path=jnp.asarray(
            [
                [0, 0, rng.integers(1, n_genes)],
                [0, 0, 2],
                [0, 2, 1],
                [0, 1, 1 + rng.integers(1, n_genes - 1)],
                [0, 1, 3],
                [1, 3, 2],
                [0, 1, 3],
            ]
        ),
    )
    theta = jnp.asarray(rng.normal(size=(n_genes, n_genes)))
    extended_theta = rates._extend_theta(theta)

    offdiag = rates._construct_log_Q_offdiag(paths=paths, extended_theta=extended_theta)

    npt.assert_allclose(offdiag.start, paths.start)
    npt.assert_allclose(offdiag.end, paths.end)
    expected = jnp.asarray(
        [
            rates._construct_log_transtion_rate(
                traj=traj, extended_theta=extended_theta
            )
            for traj in paths.path
        ]
    )
    npt.assert_allclose(offdiag.value, expected)


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


# *** segment_logsumexp ***


def test_segment_logsumexp() -> None:
    indices = jnp.asarray([1, 0, 0, 2, 2, 2])
    exp_values = jnp.asarray([0.1, 2, 3, 10, 11, 12])
    obtained = rates.segment_logsumexp(
        jnp.log(exp_values),
        segment_ids=indices,
        num_segments=3,
    )
    expected = jnp.asarray(
        [
            jnp.log(exp_values[1] + exp_values[2]),
            jnp.log(exp_values[0]),
            jnp.log(exp_values[3] + exp_values[4] + exp_values[5]),
        ]
    )

    npt.assert_allclose(expected, obtained)


# *** _log_neg_Q_to_log_V ***


@pytest.mark.parametrize("size", [2, 5])
@pytest.mark.parametrize("seed", [42, 111])
def test_log_neg_Q_to_log_V(size: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    # The diagonal entries of Q are non-positive
    Q = jnp.append(rng.uniform(low=-5.0, high=0.0, size=size), 0.0)

    log_neg_Q = jnp.log(-Q)
    logV = jnp.log1p(-Q)  # log(1 - Q)

    npt.assert_allclose(logV, rates._log_neg_Q_to_log_V(log_neg_Q))
