import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from pmhn._trees._backend_jax._sparse import COOMatrix


def logprob_forward_substitution(
    log_magic: COOMatrix,
) -> Float[Array, " S"]:
    """Uses forward/back substitution to compute the log probability
    of a tree.

    Works in the log-space for numerical stability.

    Args:
        log_magic: the matrix with log(V) on the diagonal
          and log(-V)=log(Q) off the diagonal.

    Returns:
        The log probability of the trees.

    Note:
        Assumes that `start < end` for all off-diagonal entries.
    """

    def single_iter(i, logx):
        logQ_row = log_magic.get_vector_fixed_end(end=i)
        log_numerator = jax.scipy.special.logsumexp(logQ_row + logx)

        value = log_numerator - log_magic.diagonal[i]

        return logx.at[i].set(value)

    size = log_magic.size
    logV0 = log_magic.diagonal[0]

    log_x = jax.lax.fori_loop(
        lower=1,
        upper=size,
        body_fun=single_iter,
        init_val=jnp.full(size, fill_value=-jnp.inf).at[0].set(-logV0),
    )
    return log_x
