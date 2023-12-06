from jaxtyping import Array, Float

from pmhn._trees._backend_jax._wrapper import WrappedTree


def loglikelihood(
    theta: Float[Array, "G G"],
    omega: Float[Array, " G"],
    tree: WrappedTree,
    log_tau: float | Float = 0.0,
) -> Float:
    # TODO(Pawel): UNTESTED
    # TODO(Pawel): NOT-IMPLEMENTED
    pass
