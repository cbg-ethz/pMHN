import pymc as pm

import pmhn._ppl._priors as priors


def test_regularized_horseshoe(n_mutations: int = 5) -> None:
    model = priors.prior_regularized_horseshoe(n_mutations=n_mutations)

    assert hasattr(model, "theta")

    with model:
        idata = pm.sample_prior_predictive(samples=1)

    theta_sample = idata.prior["theta"].values.squeeze()  # type: ignore
    assert theta_sample.shape == (n_mutations, n_mutations)
