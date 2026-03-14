from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from pmhn._trees._backend_jax_ragged._rates import (
    _construct_log_magic_matrix,
    _construct_log_magic_matrix_with_rate_model,
)
from pmhn._trees._backend_jax_ragged._solver import (
    logprob_forward_substitution_layerwise,
)
from pmhn._trees._backend_jax_ragged._wrapper import RaggedTree


def loglikelihood(
    theta: Float[Array, "G G"],
    omega: Float[Array, " G"],
    tree: RaggedTree,
    log_tau: float | Float = 0.0,
) -> Float:
    log_magic = _construct_log_magic_matrix(
        tree=tree,
        theta=theta,
        omega=omega,
        log_tau=log_tau,
    )
    log_probs = logprob_forward_substitution_layerwise(
        log_magic=log_magic,
        node_layer=tree.node_layer,
        layer_ptr=tree.layer_ptr,
    )
    return log_probs[-1]


def loglikelihood_many(
    theta: Float[Array, "G G"],
    omega: Float[Array, " G"],
    trees: list[RaggedTree],
    log_tau: float | Float = 0.0,
) -> Float[Array, " n_trees"]:
    """Evaluates likelihood on a heterogeneous list of wrapped trees.

    Note:
        Trees with different sizes/shapes cannot be vmap'ed directly. A Python loop
        over trees is the intended baseline pattern.
    """
    return jnp.asarray([loglikelihood(theta, omega, tree, log_tau) for tree in trees])


def loglikelihood_with_rate_model(
    log_rate_model: Callable[[Float[Array, " G"], Int[Array, ""]], Float],
    omega: Float[Array, " G"],
    tree: RaggedTree,
    log_tau: float | Float = 0.0,
) -> Float:
    """Likelihood using a generic log-rate model instead of bilinear theta."""
    log_magic = _construct_log_magic_matrix_with_rate_model(
        tree=tree,
        omega=omega,
        log_tau=log_tau,
        log_rate_model=log_rate_model,
    )
    log_probs = logprob_forward_substitution_layerwise(
        log_magic=log_magic,
        node_layer=tree.node_layer,
        layer_ptr=tree.layer_ptr,
    )
    return log_probs[-1]


def loglikelihood_many_with_rate_model(
    log_rate_model: Callable[[Float[Array, " G"], Int[Array, ""]], Float],
    omega: Float[Array, " G"],
    trees: list[RaggedTree],
    log_tau: float | Float = 0.0,
) -> Float[Array, " n_trees"]:
    """Evaluates model-based likelihood on a heterogeneous list of wrapped trees."""
    return jnp.asarray(
        [
            loglikelihood_with_rate_model(log_rate_model, omega, tree, log_tau)
            for tree in trees
        ]
    )
