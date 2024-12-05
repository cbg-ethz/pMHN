# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from metmhn import pmhn

# +
rng = np.random.default_rng(42)

n_samples = 400
n_genes = 10

Y = rng.binomial(1, p=0.3, size=(n_samples, n_genes))
X = rng.normal(size=(n_samples, 2))


# +
class StratifiedDataSet(NamedTuple):
    """Data set stratified by number of mutations.

    Attrs:
        n_genes: number of all loci considered
        covariates_zeros: represents covariates of the patients
            with no mutation, shape (n_patients_zero, n_features)
        genotypes_nonzero: list of arrays stratified by the (non-zero)
            number of occurred mutations. Arrays are of shape
                (n_patients_in_strata[i], n_genes)
        covariates_nonzero: covariates associated with each
            `genotypes_nonzero` strata. Arrays are of shape
                (n_patients_in_strata[i], n_features)
        n_mutations: number of mutations occurred in the strata
        n_mutation_shapes: template arrays controlling the shapes,
            the `n_mutations_shapes[i]` has shape `(2**n_mutations[i],)`
    """

    n_genes: int
    covariates_zeros: np.ndarray

    genotypes_nonzero: list[np.ndarray]
    covariates_nonzero: list[np.ndarray]
    n_mutations: list[int]
    n_mutation_shapes: list[np.ndarray]


def stratify_dataset(Y, X=None) -> StratifiedDataSet:
    Y = np.asarray(Y)
    if X is None:
        X = np.zeros((Y.shape[0], 1))

    ns, Ys, Xs = [], [], []

    n_genes = Y.shape[1]
    for n in range(1, n_genes + 1):
        idx = Y.sum(axis=1) == n

        if idx.sum() > 0:
            ns.append(n)
            Ys.append(jnp.asarray(Y[idx, ...]))
            Xs.append(jnp.asarray(X[idx, ...]))

    idx0 = Y.sum(axis=1) == 0

    return StratifiedDataSet(
        n_genes=n_genes,
        covariates_zeros=jnp.asarray(X[idx0, ...]),
        genotypes_nonzero=Ys,
        covariates_nonzero=Xs,
        n_mutations=ns,
        n_mutation_shapes=[jnp.zeros(2**n) for n in ns],
    )


# -

dataset = stratify_dataset(Y, X)

# +
rng = np.random.default_rng(101)

theta = np.zeros((n_genes, n_genes))
omega = np.zeros(n_genes)


theta = rng.normal(size=theta.shape)


theta = jnp.asarray(theta)
omega = jnp.asarray(omega)


# +
def theta_link(params, x):
    return jnp.eye(n_genes, n_genes)


def omega_link(params, x):
    return jnp.zeros(n_genes)


# +
def _default_theta_link(n_genes):
    def fn(params, x):
        return jnp.eye(n_genes)

    return fn


def _default_omega_link(n_genes):
    def fn(params, x):
        return jnp.zeros(n_genes)

    return fn


@jax.custom_jvp
def _loglike(theta, omega, state, n):
    return pmhn._lp_prim_obs(
        theta,
        omega,
        state,
        n,
    )


@_loglike.defjvp
def _loglike_jvp(primals, tangents):
    theta, omega, state, n = primals
    theta_dot, omega_dot, _, _ = tangents

    primal_out, grad_theta, grad_omega = pmhn._grad_prim_obs(
        theta,
        omega,
        state,
        n,
    )

    tangent_out = jnp.sum(grad_theta * theta_dot) + jnp.sum(grad_omega * omega_dot)
    return primal_out, tangent_out


def generate_loglikelihood(
    dataset: StratifiedDataSet,
    theta_link_fn=None,
    omega_link_fn=None,
):
    if theta_link_fn is None:
        theta_link_fn = _default_theta_link(dataset.n_genes)
    if omega_link_fn is None:
        omega_link_fn = _default_omega_link(dataset.n_genes)

    def loglikelihood(params):
        def adjusted_loglike(x, state, n):
            theta = theta_link_fn(params, x)
            omega = omega_link_fn(params, x)
            return _loglike(theta, omega, state, n)

        def adjusted_loglike_zero(x):
            theta = theta_link_fn(params, x)
            return pmhn._lp_prim_obs_az(theta)

        loglikelihood_nonzero_n = jnp.array(
            [
                jax.vmap(adjusted_loglike, in_axes=(0, 0, None))(xs, ys, ns_shape).sum()
                for xs, ys, ns_shape in zip(
                    dataset.covariates_nonzero,
                    dataset.genotypes_nonzero,
                    dataset.n_mutation_shapes,
                )
            ]
        ).sum()

        loglikelihood_zero_n = jax.vmap(adjusted_loglike_zero)(
            dataset.covariates_zeros
        ).sum()

        return loglikelihood_nonzero_n + loglikelihood_zero_n

    return loglikelihood


# -


class FullParams(NamedTuple):
    theta: jnp.ndarray
    omega: jnp.ndarray

    @staticmethod
    def theta_link(params, x):
        return params.theta

    @staticmethod
    def omega_link(params, x):
        return params.omega


ll_fn = generate_loglikelihood(
    dataset,
    theta_link_fn=FullParams.theta_link,
    omega_link_fn=FullParams.omega_link,
)

ll_fn(FullParams(jnp.eye(n_genes) + 0.5, jnp.zeros(n_genes)))

jax.grad(ll_fn)(FullParams(jnp.eye(n_genes) + 0.3, jnp.zeros(n_genes)))


@jax.jit
def loglikelihood(theta, omega):
    loglikelihood_nonzero_n = jnp.array(
        [
            jax.vmap(f, in_axes=(None, None, 0, None))(
                theta,
                omega,
                ys,
                ns_shape,
            ).sum()
            for ys, ns_shape in zip(dataset.genotypes, dataset.n_mutation_shapes)
        ]
    ).sum()

    loglikelihood_zero_n = dataset.n_zeros * pmhn._lp_prim_obs_az(theta)

    return loglikelihood_nonzero_n + loglikelihood_zero_n


# %timeit loglikelihood(theta, omega + 0.3).block_until_ready()

loglikelihood(theta, omega)

# %timeit jax.grad(loglikelihood, 0)(theta + 0.5, omega + 0.9).block_until_ready()


# +


def f_fwd(theta, omega, state, n):
    primal_out, grad_theta, grad_omega = pmhn._grad_prim_obs(
        theta,
        omega,
        state,
        n,
    )
    return primal_out, (grad_theta, grad_omega, state, n)


def f_bwd(res, g):
    grad_theta, grad_omega, state, n = res
    grad_state = jnp.zeros_like(state, dtype=float)
    grad_n = jnp.zeros_like(n)
    return (g * grad_theta, g * grad_omega, grad_state, grad_n)


# f.defvjp(f_fwd, f_bwd)


# -

f(theta, omega, jnp.eye(n_genes)[0], jnp.zeros(2**1))
jax.jacfwd(f)(theta, omega, jnp.eye(n_genes)[0], jnp.zeros(2**1))

jax.jacrev(f)(theta, omega, jnp.eye(n_genes)[0], jnp.zeros(2**1))

f_val, df_theta, df_omega = pmhn._grad_prim_obs(
    theta, omega, jnp.eye(n_genes)[0], jnp.zeros(2**1)
)
df_theta


jax.grad(f)(theta, omega, jnp.eye(n_genes)[0], 1)

jax.grad(f, argnums=1)(theta, omega, jnp.eye(n_genes)[0], 1)

pmhn._lp_prim_obs(
    log_theta=jnp.asarray(theta),
    log_d_p=jnp.zeros(n_genes),
    state_pt=jnp.eye(n_genes)[1],  # jnp.zeros(n_genes, dtype=int),
    n_prim=1,
)

pmhn._lp_prim_obs(
    jnp.asarray(theta),
    jnp.zeros(n_genes),
    jnp.eye(n_genes)[0],  # jnp.zeros(n_genes, dtype=int),
    1,
)

stratify_dataset(Y).n_zeros

stratify_dataset(Y).genotypes
