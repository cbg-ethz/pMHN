import numpy as np
import numpy.testing as nptest
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


@pytest.mark.parametrize("n_mutations", [2, 5])
@pytest.mark.parametrize("p_offdiag", [0.2, 0.5])
@pytest.mark.parametrize("diag_mean", [-1.0])
@pytest.mark.parametrize("diag_std", [0.3])
def test_sample_spike_and_slab(
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
