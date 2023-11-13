import pymc as pm
import pytest

import pmhn._ppl._priors as priors


@pytest.mark.parametrize(
    "model_factory",
    [
        priors.prior_only_baseline_rates,
        priors.prior_normal,
        priors.prior_regularized_horseshoe,
        priors.prior_offdiagonal_laplace,
        priors.prior_horseshoe,
        priors.prior_spike_and_slab_marginalized,
    ],
)
def test_basic_prior_test(model_factory, n_mutations: int = 5) -> None:
    """Tests whether `model.theta` exists
    and has shape (n_mutations, n_mutations)."""
    model = model_factory(n_mutations=n_mutations)

    assert hasattr(model, "theta")

    with model:
        idata = pm.sample_prior_predictive(samples=1)

    theta_sample = idata.prior["theta"].values.squeeze()  # type: ignore
    assert theta_sample.shape == (n_mutations, n_mutations)
