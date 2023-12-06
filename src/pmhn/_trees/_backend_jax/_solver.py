import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def logQending(
    size: int,
    logQ: None,
    m: int,
    fill_value: float | Float = -jnp.inf,
) -> Float[Array, " size"]:
    # TODO(Pawel): UNTESTED
    # TODO(Pawel): NOT-IMPLEMENTED
    def body_fun(i, carry):
        # Unwrap the right entry
        start = logQ.start[i]
        end = logQ.end[i]
        value = logQ.value[i]

        # If end == m, we want to overwrite carry[start].
        # Otherwise we leave it untouched
        set_value = jax.lax.cond(end == m, lambda: value, lambda: carry[start])
        return carry.at[start].set(set_value)

    return jax.lax.fori_loop(
        lower=0,
        upper=_offdiag_length(logQ),
        body_fun=body_fun,
        init_val=jnp.full(shape=(size,), fill_value=fill_value),
    )


def logprob_forward_substitution(
    logV_diag: Float[Array, " S"],
    logQ: None,
) -> Float:
    # TODO(Pawel): UNTESTED
    # TODO(Pawel): NOT-IMPLEMENTED
    S = logV_diag.shape[0]

    def single_iter(i, logx):
        logQ_row = logQending(size=S, logQ=logQ, m=i, fill_value=-jnp.inf)
        log_numerator = jax.scipy.special.logsumexp(logQ_row + logx)

        value = log_numerator - logV_diag[i]

        return logx.at[i].set(value)

    log_x = jax.lax.fori_loop(
        lower=1,
        upper=S,
        body_fun=single_iter,
        init_val=jnp.full_like(logV_diag, fill_value=-jnp.inf).at[0].set(-logV_diag[0]),
    )
    return log_x
