"""Submodule used to construct the rates matrices from wrapped tree."""
from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Float, Array, Int

from pmhn._trees._backend_jax._wrapper import WrappedTree, IndexedPaths, DoublyIndexedPaths, ExitPathsArray
from pmhn._trees._backend_jax._sparse import Values, COOMatrix
from pmhn._trees._backend_jax._const import PADDING

# We require PADDING to be -1 because of `_extend_theta` and `_extend_omega`
if PADDING != -1:
    raise ValueError("PADDING has to be -1 for this to work")


def _extend_theta(theta: Float[Array, "G G"]) -> Float[Array, "G+1 G+1"]:
    """Adds a new row and column to theta, filled with zeros,
    to represent a mock gene not affecting any rates.
    
    Note:
        `PADDING` has to be -1 for this to work
        (as we add the last = -1th gene)
    """
    n = theta.shape[0]
    ret = jnp.zeros((n + 1, n + 1))
    return ret.at[:n, :n].set(theta)


def _extend_omega(omega: Float[Array, " G"]) -> Float[Array, " G+1"]:
    """Adds a 0 entry to the omega vector to represent a mock
    gene not affecting any rates.
    
    Note:
        `PADDING` has to be -1 for this to work
        (as we add the last = -1th gene)
    """
    return jnp.append(omega, 0.0)


def _construct_log_transtion_rate(
    traj: Int[Array, " n_events"],
    extended_theta: Float[Array, "n+1 n+1"],
) -> Float:
    new_mut = traj[-1]
    return jnp.sum(extended_theta[new_mut - 1, traj - 1])


def _construct_log_exit_rate(
    traj: Int[Array, " n_events"],
    extended_omega: Float[Array, " n+1"],
) -> Float:
    return jnp.sum(extended_omega[traj - 1])


def _construct_log_Q_offdiag(paths: DoublyIndexedPaths, extended_theta: Float[Array, "G+1 G+1"]) -> Values:
    return Values(
        start=paths.start,
        end=paths.end,
        value=jnp.apply_along_axis(
            func1d=_construct_log_transtion_rate,
            axis=1,
            extended_theta=extended_theta,
        )
    )


def _construct_log_neg_Q_diag(paths: IndexedPaths, extended_theta: Float[Array, "G+1 G+1"]) -> Float[Array, " n_subtrees"]:
    pass
    

def _construct_log_U(paths: ExitPathsArray, extended_omega: Float[Array, " G+1"], log_tau: float | Float) -> Float[Array, " n_subtrees"]:
    return jnp.apply_along_axis(
        func1d=_construct_log_exit_rate,
        axis=1,
        extended_omega=extended_omega,
    ) - log_tau  # Note that we *subtract* log_tau, because we have U as a rate matrix, i.e, 1/tau


def _construct_log_magic_matrix(
    tree: WrappedTree,
    theta: Float[Array, "G G"],
    omega: Float[Array, " G"],
    log_tau: float | Float,
) -> COOMatrix:
    """Constructs the rates matrix from the wrapped tree."""
    pass
