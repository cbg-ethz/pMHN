import pytensor
import pytensor.tensor as pt
from pytensor.gradient import verify_grad

import numpy as np
import pytest

import pmhn._ppl._singlemhn as smhn
import pmhn._backend._learnmhn as lmhn


@pytest.mark.parametrize("n_patients", (3, 4))
@pytest.mark.parametrize("n_genes", (2, 5))
def test_loglikelihood(n_patients: int, n_genes: int) -> None:
    rng = np.random.default_rng(42)
    mutations = rng.binomial(1, 0.5, size=(n_patients, n_genes))
    theta = rng.normal(size=(n_genes, n_genes))

    _, loglike_backend = lmhn.MHNCythonBackend().gradient_and_loglikelihood(
        mutations=mutations, theta=theta
    )

    loglike_op = smhn.MHNLoglikelihood(data=mutations, backend=lmhn.MHNCythonBackend())

    x = pt.matrix()
    # TODO(Pawel): Remove the type annotation when Pyright starts
    #   locating `function` in the Pytensor library
    f = pytensor.function([x], [loglike_op(x)])  # type: ignore

    loglike_calc = f(theta)
    loglike_calc = loglike_calc[0]  # TODO(Pawel): Check why this is a list

    assert pytest.approx(loglike_backend, rel=0.01) == loglike_calc


@pytest.mark.parametrize("n_patients", (3, 4))
@pytest.mark.parametrize("n_genes", (2, 5))
def test_loglikelihood_grad(n_patients: int, n_genes: int) -> None:
    """Test the gradient of the loglikelihood."""
    rng = np.random.default_rng(42)
    mutations = rng.binomial(1, 0.5, size=(n_patients, n_genes))
    theta = rng.normal(size=(n_genes, n_genes))

    loglike_op = smhn.MHNLoglikelihood(data=mutations, backend=lmhn.MHNCythonBackend())

    verify_grad(loglike_op, [theta], rng=rng)
