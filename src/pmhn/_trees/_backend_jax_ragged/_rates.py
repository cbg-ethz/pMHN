"""Rate construction for ragged tree representation."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from pmhn._trees._backend_jax._sparse import COOMatrix, Values
from pmhn._trees._backend_jax_ragged._wrapper import RaggedPaths, RaggedTree


def segment_logsumexp(
    data: Float[Array, " n"],
    segment_ids: Int[Array, " n"],
    num_segments: int,
) -> Float[Array, " num_segments"]:
    """Segment-wise logsumexp."""
    max_per_segment = jax.ops.segment_max(
        data, segment_ids=segment_ids, num_segments=num_segments
    )
    adjusted_values = jnp.exp(data - max_per_segment[segment_ids])
    summed_exp_values = jax.ops.segment_sum(
        adjusted_values, segment_ids=segment_ids, num_segments=num_segments
    )
    return jnp.log(summed_exp_values) + max_per_segment


def _construct_path_log_transition_rates(
    paths: RaggedPaths,
    theta: Float[Array, "G G"],
) -> Float[Array, " n_paths"]:
    """Computes transition log-rates for all paths."""
    path_ids = paths.event_path_id
    new_mut = paths.path_last_event[path_ids]
    event_mut = paths.events_flat

    per_event = theta[new_mut - 1, event_mut - 1]
    return jax.ops.segment_sum(
        per_event, segment_ids=path_ids, num_segments=paths.n_paths
    )


def _construct_path_log_exit_rates(
    paths: RaggedPaths,
    omega: Float[Array, " G"],
) -> Float[Array, " n_paths"]:
    """Computes exit log-rates for all paths."""
    path_ids = paths.event_path_id
    per_event = omega[paths.events_flat - 1]
    return jax.ops.segment_sum(
        per_event, segment_ids=path_ids, num_segments=paths.n_paths
    )


def _construct_log_Q_offdiag(
    tree: RaggedTree,
    path_log_transition_rates: Float[Array, " n_paths"],
) -> Values:
    return Values(
        start=tree.edge_start,
        end=tree.edge_end,
        value=path_log_transition_rates[tree.edge_path_id],
    )


def _construct_log_neg_Q_diag(
    tree: RaggedTree,
    path_log_transition_rates: Float[Array, " n_paths"],
) -> Float[Array, " n_subtrees"]:
    return segment_logsumexp(
        path_log_transition_rates[tree.diag_path_id],
        segment_ids=tree.diag_subtree_id,
        num_segments=tree.n_subtrees,
    )


def _construct_log_U(
    tree: RaggedTree,
    path_log_exit_rates: Float[Array, " n_paths"],
    log_tau: float | Float,
) -> Float[Array, " n_subtrees"]:
    # For parity with the current padded backend implementation, use the same
    # placeholder exit behavior: every subtree gets log_U = -log_tau.
    _ = path_log_exit_rates
    return jnp.full((tree.n_subtrees,), fill_value=-log_tau)


def _log_neg_Q_to_log_V(
    log_neg_Q: Float[Array, " n_subtrees"],
) -> Float[Array, " n_subtrees"]:
    return jax.nn.softplus(log_neg_Q)


def _construct_log_magic_matrix(
    tree: RaggedTree,
    theta: Float[Array, "G G"],
    omega: Float[Array, " G"],
    log_tau: float | Float,
) -> COOMatrix:
    """Constructs log-magic matrix from ragged tree representation."""
    path_log_transition_rates = _construct_path_log_transition_rates(
        paths=tree.paths, theta=theta
    )
    path_log_exit_rates = _construct_path_log_exit_rates(paths=tree.paths, omega=omega)

    log_neg_Q_diag = _construct_log_neg_Q_diag(
        tree=tree, path_log_transition_rates=path_log_transition_rates
    )
    log_Q_offdiag = _construct_log_Q_offdiag(
        tree=tree, path_log_transition_rates=path_log_transition_rates
    )
    log_U = _construct_log_U(
        tree=tree,
        path_log_exit_rates=path_log_exit_rates,
        log_tau=log_tau,
    )

    log_neg_Q_diag_adjusted = log_neg_Q_diag - log_U
    log_Q_offdiag_adjusted = Values(
        start=log_Q_offdiag.start,
        end=log_Q_offdiag.end,
        value=log_Q_offdiag.value - log_U[log_Q_offdiag.start],
    )

    return COOMatrix(
        diagonal=_log_neg_Q_to_log_V(log_neg_Q_diag_adjusted),
        offdiagonal=log_Q_offdiag_adjusted,
        fill_value=-jnp.inf,
    )
