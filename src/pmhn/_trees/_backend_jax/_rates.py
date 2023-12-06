"""Submodule used to construct the rates matrices from wrapped tree."""
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from pmhn._trees._backend_jax._const import PADDING
from pmhn._trees._backend_jax._sparse import COOMatrix, Values
from pmhn._trees._backend_jax._wrapper import (
    DoublyIndexedPaths,
    ExitPathsArray,
    IndexedPaths,
    WrappedTree,
)

# We require PADDING to be 0 because of `_extend_theta` and `_extend_omega`
if PADDING != 0:
    raise ValueError("PADDING has to be 0 for this to work")


def _extend_theta(theta: Float[Array, "G G"]) -> Float[Array, "G+1 G+1"]:
    """Adds a new row and column to theta, filled with zeros,
    to represent a mock gene not affecting any rates.

    Note:
        `PADDING` has to be -1 for this to work
        (as we add the last = -1th gene)
    """
    n = theta.shape[0]
    ret = jnp.zeros((n + 1, n + 1), dtype=theta.dtype)
    return ret.at[:n, :n].set(theta)


def _extend_omega(omega: Float[Array, " G"]) -> Float[Array, " G+1"]:
    """Adds a 0 entry to the omega vector to represent a mock
    gene not affecting any rates.

    Note:
        `PADDING` has to be 0 for this to work
        The trick is that we use `gene-1`, so that
        `PADDING` will access the -1th entry, which
        is the last entry.
    """
    return jnp.append(omega, 0.0)


def _construct_log_transtion_rate(
    traj: Int[Array, " n_events"],
    extended_theta: Float[Array, "n+1 n+1"],
) -> Float:
    new_mut = traj[-1]  # The added mutation is the last one
    return jnp.sum(extended_theta[new_mut - 1, traj - 1])


def _construct_log_exit_rate(
    traj: Int[Array, " n_events"],
    extended_omega: Float[Array, " G+1"],
) -> Float:
    return jnp.sum(extended_omega[traj - 1])


def _construct_log_Q_offdiag(
    paths: DoublyIndexedPaths, extended_theta: Float[Array, "G+1 G+1"]
) -> Values:
    return Values(
        start=paths.start,
        end=paths.end,
        value=jnp.apply_along_axis(
            func1d=_construct_log_transtion_rate,
            axis=1,
            extended_theta=extended_theta,
            arr=paths.path,
        ),
    )


def segment_logsumexp(
    data: Float[Array, " n"],
    segment_ids: Int[Array, " n"],
    num_segments: int,
) -> Float[Array, " num_segments"]:
    """logsumexp operation applied to each segment

    Args:
        values: values to which the logsumexp is applied
        segment_ids: indices of the segments
        num_segments: number of segments

    See
    https://jax.readthedocs.io/en/latest/_autosummary/jax.ops.segment_sum.html
    """
    max_per_segment = jax.ops.segment_max(
        data, segment_ids=segment_ids, num_segments=num_segments
    )
    adjusted_values = jnp.exp(data - max_per_segment[segment_ids])
    summed_exp_values = jax.ops.segment_sum(
        adjusted_values, segment_ids=segment_ids, num_segments=num_segments
    )
    return jnp.log(summed_exp_values) + max_per_segment


def _construct_log_neg_Q_diag(
    paths: IndexedPaths, extended_theta: Float[Array, "G+1 G+1"], n_subtrees: int
) -> Float[Array, " n_subtrees"]:
    """Constructs the log (lambda_1 + ... + lambda_k) entries.

    Note:
        This is the same as log(-Q_{ii}) as Q_{ii} <= 0
    """
    # TODO(Pawel): UNTESTED
    log_rates = jnp.apply_along_axis(
        func1d=_construct_log_transtion_rate,
        axis=1,
        arr=paths.path,
        extended_theta=extended_theta,
    )
    return segment_logsumexp(
        log_rates,
        segment_ids=paths.index,
        num_segments=n_subtrees,
    )


def _log_neg_Q_to_log_V(
    log_neg_Q: Float[Array, " n_subtrees"]
) -> Float[Array, " n_subtrees"]:
    """Converts the log(-Q_{ii}) entries to log(V_{ii}) entries.

    We have
        V_{ii} = 1 - Q_{ii}
    so that
        log(V_{ii}) = log(1 - Q_{ii}) = log(1 + exp(log(-Q_{ii})))
                    = log(1 + exp(input_i))
    """
    # TODO(Pawel): UNTESTED
    return jax.nn.softplus(log_neg_Q)


def _construct_log_U(
    paths: ExitPathsArray, extended_omega: Float[Array, " G+1"], log_tau: float | Float
) -> Float[Array, " n_subtrees"]:
    # TODO(Pawel): UNTESTED
    return (
        jnp.apply_along_axis(
            func1d=_construct_log_exit_rate,
            axis=1,
            extended_omega=extended_omega,
            arr=paths,
        )
        - log_tau
    )  # Note that we *subtract* log_tau, because we have U as a rate matrix, i.e, 1/tau


def _construct_log_magic_matrix(
    tree: WrappedTree,
    theta: Float[Array, "G G"],
    omega: Float[Array, " G"],
    log_tau: float | Float,
) -> COOMatrix:
    # TODO(Pawel): UNTESTED
    """Constructs the rates matrix from the wrapped tree."""
    # Construct the Q matrix
    extended_theta = _extend_theta(theta)
    log_neg_Q_diag = _construct_log_neg_Q_diag(
        paths=tree.diag_paths,
        extended_theta=extended_theta,
        n_subtrees=tree.n_subtrees,
    )
    log_Q_offdiag = _construct_log_Q_offdiag(
        paths=tree.offdiag_paths,
        extended_theta=extended_theta,
    )

    # Construct the exit rates
    log_U = _construct_log_U(
        paths=tree.exit_paths,
        extended_omega=_extend_omega(omega),
        log_tau=log_tau,
    )
    # Construct the Q' matrix, which is Q matrix adjusted by U.
    # We want to adjust log (-Q_{ii}) to
    # log( -Q_{ii} / U_{ii} ) = log(-Q_{ii}) - log(U_{ii})
    # Similarly we want adjust log Q_{start, end} to
    # log (Q_{start, end} / U_{start, start})
    log_neg_Q_diag_adjusted = log_neg_Q_diag - log_U
    log_Q_offdiag_adjusted = Values(
        start=log_Q_offdiag.start,
        end=log_Q_offdiag.end,
        value=log_Q_offdiag.value - log_U[log_Q_offdiag.start],
    )

    # Construct the magic log V matrix
    return COOMatrix(
        diagonal=_log_neg_Q_to_log_V(log_neg_Q_diag_adjusted),
        offdiagonal=log_Q_offdiag_adjusted,
        fill_value=-jnp.inf,
    )
