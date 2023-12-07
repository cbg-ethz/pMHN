from jaxtyping import Array, Float

from pmhn._trees._backend_jax._rates import _construct_log_magic_matrix
from pmhn._trees._backend_jax._solver import logprob_forward_substitution
from pmhn._trees._backend_jax._wrapper import WrappedTree


def loglikelihood(
    theta: Float[Array, "G G"],
    omega: Float[Array, " G"],
    tree: WrappedTree,
    log_tau: float | Float = 0.0,
) -> Float:
    # Construct the "magic matrix":
    # log(V) on the diagonal,
    # log(-V) = log(Q) off the diagonal.
    log_magic = _construct_log_magic_matrix(
        tree=tree,
        theta=theta,
        omega=omega,
        log_tau=log_tau,
    )
    # Calculate the log-probability of all trees
    log_probs = logprob_forward_substitution(log_magic)
    # Return the log-probability of the last tree
    return log_probs[-1]
