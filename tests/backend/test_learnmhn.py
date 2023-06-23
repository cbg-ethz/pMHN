import numpy as np
import pytest

import pmhn._backend._learnmhn as lmhn


def backends() -> list[lmhn.MHNBackend]:
    return [
        lmhn.MHNJoblibBackend(n_jobs=1),
        lmhn.MHNJoblibBackend(n_jobs=-1),
        lmhn.MHNCythonBackend(),
    ]


@pytest.mark.parametrize("backend", backends())
@pytest.mark.parametrize("n_patients", [3, 5])
@pytest.mark.parametrize("n_mutations", [4, 8])
def test_shapes_work(
    backend: lmhn.MHNBackend, n_patients: int, n_mutations: int
) -> None:
    rng = np.random.default_rng(42)

    mutations = rng.binomial(1, 0.2, size=(n_patients, n_mutations))
    theta = rng.normal(size=(n_mutations, n_mutations))

    grad, ll = backend.gradient_and_loglikelihood(mutations=mutations, theta=theta)

    assert grad.shape == (n_mutations, n_mutations)
    assert isinstance(ll, float)


@pytest.mark.parametrize("backend", backends())
@pytest.mark.parametrize("ns_patients", [(2, 3), (1, 4), (2, 2)])
@pytest.mark.parametrize("n_mutations", [4, 8])
def test_additivity(
    backend: lmhn.MHNBackend, ns_patients: tuple[int, int], n_mutations: int
) -> None:
    """Check whether the loglikelihood and gradient are additive in patients."""
    n1, n2 = ns_patients
    n_patients = n1 + n2

    rng = np.random.default_rng(120)

    mutations = rng.binomial(1, 0.2, size=(n_patients, n_mutations))
    theta = rng.normal(size=(n_mutations, n_mutations))

    grad1, ll1 = backend.gradient_and_loglikelihood(
        mutations=mutations[:n1], theta=theta
    )
    grad2, ll2 = backend.gradient_and_loglikelihood(
        mutations=mutations[n1:], theta=theta
    )

    grad_all, ll_all = backend.gradient_and_loglikelihood(
        mutations=mutations, theta=theta
    )

    assert np.allclose(grad_all, grad1 + grad2)
    assert pytest.approx(ll_all) == ll1 + ll2
