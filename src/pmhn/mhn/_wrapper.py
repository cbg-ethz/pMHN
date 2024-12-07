from typing import Callable, NamedTuple, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int  # type: ignore

from pmhn.mhn._backend import (
    grad_loglikelihood_nonzero,
    loglikelihood_nonzero,
    loglikelihood_zero,
)

_MutationShapePlaceholder = Float[Array, " two_to_power_n_mutated"]


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
    covariates_zeros: Float[Array, "n_patients_zero n_features"]

    genotypes_nonzero: list[Int[Array, "n_patients_strata n_genes"]]
    covariates_nonzero: list[Float[Array, "n_patients_strata n_features"]]
    n_mutations: list[int]
    n_mutation_shapes: list[_MutationShapePlaceholder]


def stratify_dataset(
    Y: Int[Array, "n_patients n_genes"],
    X: Float[Array, "n_patients n_features"] | None = None,
) -> StratifiedDataSet:
    Y = np.asarray(Y)  # type: ignore
    if X is None:
        X = np.zeros((Y.shape[0], 1))  # type: ignore
    assert X is not None

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


_ThetaMatrix = Float[Array, "n_genes n_genes"]
_OmegaVector = Float[Array, " n_genes"]
_State = Int[Array, " n_genes"]
_Float = Float[Array, " "]


@jax.custom_jvp
def _loglike(
    theta: _ThetaMatrix,
    omega: _OmegaVector,
    state: _State,
    n: _MutationShapePlaceholder,
) -> _Float:
    return loglikelihood_nonzero(
        theta,
        omega,
        state,
        n,
    )


@_loglike.defjvp
def _loglike_jvp(
    primals: tuple[_ThetaMatrix, _OmegaVector, _State, _MutationShapePlaceholder],
    tangents,
) -> tuple:
    theta, omega, state, n = primals
    theta_dot, omega_dot, _, _ = tangents

    primal_out, grad_theta, grad_omega = grad_loglikelihood_nonzero(
        theta,
        omega,
        state,
        n,
    )

    tangent_out = jnp.sum(grad_theta * theta_dot) + jnp.sum(grad_omega * omega_dot)
    return primal_out, tangent_out


Params = TypeVar("Params")
_ThetaLinkFn = Callable[
    [Params, Float[Array, " n_features"]], Float[Array, "n_genes n_genes"]
]

_OmegaLinkFn = Callable[[Params, Float[Array, " n_features"]], Float[Array, " n_genes"]]


def _default_theta_link(n_genes: int) -> _ThetaLinkFn:
    def fn(params: Params, x: Float[Array, " n_features"]) -> _ThetaMatrix:  # type: ignore
        return jnp.eye(n_genes)

    return fn


def _default_omega_link(n_genes: int) -> _OmegaLinkFn:
    def fn(params: Params, x: Float[Array, " n_features"]) -> Float[Array, " n_genes"]:  # type: ignore
        return jnp.zeros(n_genes)

    return fn


_LoglikelihoodFn = Callable[[Params], _Float]


def _generate_loglikelihood_from_dataset(
    dataset: StratifiedDataSet,
    theta_link_fn: _ThetaLinkFn | None,
    omega_link_fn: _OmegaLinkFn | None,
) -> _LoglikelihoodFn:
    if theta_link_fn is None:
        theta_link_fn = _default_theta_link(dataset.n_genes)
    if omega_link_fn is None:
        omega_link_fn = _default_omega_link(dataset.n_genes)

    def loglikelihood(params: Params) -> _Float:  # type: ignore
        def adjusted_loglike(x, state: _State, n: _MutationShapePlaceholder) -> _Float:
            theta = theta_link_fn(params, x)
            omega = omega_link_fn(params, x)
            return _loglike(theta, omega, state, n)

        def adjusted_loglike_zero(x) -> _Float:  # type: ignore
            theta = theta_link_fn(params, x)
            return loglikelihood_zero(theta)

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


def generate_loglikelihood(
    Y: Int[Array, "n_patients n_genes"],
    X: Float[Array, "n_patients n_features"] | None = None,
    theta_link_fn: _ThetaLinkFn | None = None,
    omega_link_fn: _OmegaLinkFn | None = None,
) -> _LoglikelihoodFn:
    dataset = stratify_dataset(Y, X)
    return _generate_loglikelihood_from_dataset(
        dataset=dataset,
        theta_link_fn=theta_link_fn,
        omega_link_fn=omega_link_fn,
    )
