import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import numpy.testing as nptest
import numpyro.distributions as distrib
import pmhn._theta as th
import pytest


def test_construct_matrix() -> None:
    generated = th.construct_matrix(
        np.asarray([-7, -8, -9]), np.arange(1, 7).reshape((3, 2))
    )

    nptest.assert_allclose(
        generated,
        np.asarray(
            [
                [-7, 1, 2],
                [3, -8, 4],
                [5, 6, -9],
            ]
        ),
    )


@pytest.mark.parametrize("n_mutations", [2, 5, 10])
def test_construct_and_decompose_are_inverse1(n_mutations: int) -> None:
    rng = np.random.default_rng(12)
    theta = rng.normal(size=(n_mutations, n_mutations))

    diag, offdiag = th.decompose_matrix(theta)

    nptest.assert_allclose(theta, th.construct_matrix(diag, offdiag))


@pytest.mark.parametrize("n_mutations", [2, 5, 10])
def test_construct_and_decompose_are_inverse2(n_mutations: int) -> None:
    rng = np.random.default_rng(21)
    diag = rng.normal(size=(n_mutations,))
    offdiag = rng.normal(size=(n_mutations, n_mutations - 1))

    theta = th.construct_matrix(diag, offdiag)

    diag_, offdiag_ = th.decompose_matrix(theta)

    nptest.assert_allclose(diag, diag_)
    nptest.assert_allclose(offdiag, offdiag_)


@pytest.mark.parametrize("n_entries", [10, 20])
@pytest.mark.parametrize("w", [0.2, 0.4])
def test_spike_and_slab_distribution_sample(
    n_entries: int, w: float, _n_points: int = 1000
):
    key = jrandom.PRNGKey(101)

    dist = th.SpikeAndSlab(w=w, scale_slab=1, scale_spike=0.00001, features=n_entries)

    sample = dist.sample(key, (_n_points,))
    assert sample.shape == (_n_points, n_entries)

    # Check if the mean is fine
    nptest.assert_allclose(sample.mean(axis=0), np.zeros(n_entries), atol=0.1)

    # Number of successes (slab component). It should be distributed as Binomial(n_entries, w)
    n_success = (jnp.abs(sample) > 0.01).sum(axis=1)
    binomial_mean = w * n_entries
    binomial_var = w * (1 - w) * n_entries
    assert n_success.mean() == pytest.approx(
        binomial_mean, abs=1.96 * (binomial_var / _n_points) ** 0.5
    )


@pytest.mark.parametrize(
    "x0",
    [
        [0, 0],
        [1, 0],
        [1, 1],
        [2, 2],
    ],
)
def test_spike_and_slab_distribution_logp(
    x0: list,
    spike_width: float = 0.1,
    slab_width: float = 2.0,
    n_features: int = 2,
    w: float = 0.1,
) -> None:
    dist_mixt = th.SpikeAndSlab(
        w=w, scale_slab=slab_width, scale_spike=spike_width, features=n_features
    )
    dist_slab = distrib.Normal(loc=jnp.zeros(n_features), scale=slab_width).to_event(1)
    dist_spike = distrib.Normal(loc=jnp.zeros(n_features), scale=spike_width).to_event(
        1
    )

    x0 = jnp.asarray(x0, dtype=float)

    logp_mixt = dist_mixt.log_prob(x0)
    logp_slab = dist_slab.log_prob(x0)
    logp_spike = dist_spike.log_prob(x0)

    p_expected = w * jnp.exp(logp_slab) + (1 - w) * jnp.exp(logp_spike)
    assert jnp.exp(logp_mixt) == pytest.approx(p_expected, rel=0.01)


@pytest.mark.parametrize("n_mutations", [2, 5])
@pytest.mark.parametrize("p_offdiag", [0.2, 0.5])
@pytest.mark.parametrize("diag_mean", [-1.0])
@pytest.mark.parametrize("diag_std", [0.3])
def test_sample_spike_and_slab_function(
    n_mutations: int,
    p_offdiag: float,
    diag_mean: float,
    diag_std: float,
    n_samples: int = 100,
) -> None:
    generated = np.zeros((n_samples, n_mutations, n_mutations))
    rng = np.random.default_rng(12)

    for i in range(n_samples):
        sample = th.sample_spike_and_slab(
            rng,
            n_mutations=n_mutations,
            p_offdiag=p_offdiag,
            diag_mean=diag_mean,
            diag_sigma=diag_std,
        )
        assert sample.shape == (n_mutations, n_mutations)
        generated[i] = sample

    # Check whether the diagonal mean and standard deviation are right
    assert pytest.approx(diag_mean, rel=0.05) == np.mean(
        generated[:, np.arange(n_mutations), np.arange(n_mutations)]
    )
    assert pytest.approx(diag_std, rel=0.1) == np.std(
        generated[:, np.arange(n_mutations), np.arange(n_mutations)]
    )

    # Check whether the offdiagonal terms are sparse enough
    for i in range(n_samples):
        np.fill_diagonal(generated[i], 0.0)

    assert pytest.approx(p_offdiag, rel=0.1) == np.sum(generated != 0.0) / (
        n_samples * n_mutations * (n_mutations - 1)
    )
