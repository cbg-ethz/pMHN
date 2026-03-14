import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from pmhn._trees._backend_jax._sparse import COOMatrix


def _segment_logsumexp_safe(
    data: Float[Array, " n"],
    segment_ids: Int[Array, " n"],
    num_segments: int,
) -> Float[Array, " num_segments"]:
    """Segment-wise logsumexp robust to all--inf segments."""
    max_per_segment = jax.ops.segment_max(
        data, segment_ids=segment_ids, num_segments=num_segments
    )
    finite_mask = jnp.isfinite(max_per_segment)
    safe_max_per_segment = jnp.where(finite_mask, max_per_segment, 0.0)

    adjusted = jnp.exp(data - safe_max_per_segment[segment_ids])
    adjusted = jnp.where(jnp.isfinite(data), adjusted, 0.0)

    summed = jax.ops.segment_sum(
        adjusted, segment_ids=segment_ids, num_segments=num_segments
    )

    out = jnp.log(summed) + safe_max_per_segment
    return jnp.where(finite_mask, out, -jnp.inf)


def logprob_forward_substitution_layerwise(
    log_magic: COOMatrix,
    node_layer: Int[Array, " S"],
    layer_ptr: Int[Array, " Lp1"],
) -> Float[Array, " S"]:
    """Layer-parallel forward substitution in log-space.

    Note:
        This assumes subtree indices are grouped by layer and `node_layer`
        is aligned with matrix indices.
    """
    size = log_magic.size
    n_layers = layer_ptr.shape[0] - 1

    log_x0 = jnp.full(size, fill_value=-jnp.inf).at[0].set(-log_magic.diagonal[0])

    if n_layers <= 1:
        return log_x0

    def single_layer(logx, layer):
        # Gather contributions into nodes in the selected layer.
        edge_end_layer = node_layer[log_magic.offdiagonal.end]
        active_edge = edge_end_layer == layer
        log_contrib = jnp.where(
            active_edge,
            log_magic.offdiagonal.value + logx[log_magic.offdiagonal.start],
            -jnp.inf,
        )

        incoming = _segment_logsumexp_safe(
            log_contrib,
            segment_ids=log_magic.offdiagonal.end,
            num_segments=size,
        )
        updates = incoming - log_magic.diagonal

        target_nodes = node_layer == layer
        new_logx = jnp.where(target_nodes, updates, logx)
        return new_logx, None

    layers = jnp.arange(1, n_layers, dtype=int)
    log_x, _ = jax.lax.scan(single_layer, log_x0, layers)
    return log_x
