from jaxtyping import Array, Float

from pmhn._trees._backend_jax._rates import _construct_log_magic_matrix
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
    _construct_log_magic_matrix(
        tree=tree,
        theta=theta,
        omega=omega,
        log_tau=log_tau,
    )

    # TODO(Pawel): UNTESTED
    # TODO(Pawel): NOT-IMPLEMENTED
    pass
