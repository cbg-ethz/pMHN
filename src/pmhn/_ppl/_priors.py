"""Priors for (log-)mutual hazard network parameters."""
from typing import Optional


import pymc as pm
import pytensor.tensor as pt


def prior_regularized_horseshoe(
    n_mutations: int,
    baselines_mean: float = 0,
    baselines_sigma: float = 3.0,
    sparsity_sigma: float = 0.3,
    c2: Optional[float] = None,
    tau: Optional[float] = None,
    lambdas_dof: int = 5,
) -> pm.Model:
    """Constructs PyMC model for regularized horseshoe prior.
    To access the (log-)mutual hazard network parameters, use the `theta` variable.

    Args:
        n_mutations: number of mutations
        baselines_mean: prior mean of the baseline rates
        sigma: prior standard deviation of the baseline rates
        sparsity_sigma: sparsity parameter, controls the prior on `tau`.
          Ignored if `tau` is provided.
        tau: if provided, will be used as the value of `tau` in the model

    Returns:
        PyMC model. Use `model.theta` to
           access the (log-)mutual hazard network variable,
           which has shape (n_mutations, n_mutations)

    Example:
        >>> model = prior_regularized_horseshoe(n_mutations=10)
        >>> with model:
        >>>     theta = model.theta
        >>>     pm.Potential("potential", some_function_of(theta))
    """
    if sparsity_sigma <= 0:
        raise ValueError("sparsity_sigma must be positive")
    if baselines_sigma <= 0:
        raise ValueError("baselines_sigma must be positive")
    if c2 is not None:
        if c2 <= 0:
            raise ValueError("c2 must be positive")

    # Below we ignore the type of some variables because Pyright
    # is not fully compatible with PyMC type annotations.
    with pm.Model() as model:  # type: ignore
        tau = pm.HalfStudentT("tau", 2, sparsity_sigma, observed=tau)  # type: ignore
        lambdas = pm.HalfStudentT(
            "lambdas_raw", lambdas_dof, shape=(n_mutations, n_mutations)
        )
        c2 = pm.InverseGamma("c2", 1, 1, observed=c2)  # type: ignore

        lambdas_ = pm.Deterministic(
            "lambdas_tilde",
            lambdas * pt.sqrt(c2 / (c2 + tau**2 * lambdas**2)),  # type: ignore
        )

        # Reparametrization trick for efficiency
        z = pm.Normal("z", 0.0, 1.0, shape=(n_mutations, n_mutations))
        betas = pm.Deterministic("betas", z * tau * lambdas_)

        # Now sample baseline rates
        baselines = pm.Normal(
            "baseline_rates",
            mu=baselines_mean,
            sigma=baselines_sigma,
            size=(n_mutations,),
        )

        # We need to construct the theta matrix out of `betas` and `baselines`
        # Note that we will effectively drop the diagonal of `betas`
        mask = pt.eye(n_mutations)
        pm.Deterministic("theta", mask * baselines + betas * (1 - mask))  # type: ignore

    return model
