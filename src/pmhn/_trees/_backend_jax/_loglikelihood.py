from jaxtyping import Float, Array, Int




def loglikelihood(
    theta: Float[Array, "G G"],
    omega: Float[Array, " G"],
    tree: WrappedTree,
    log_tau: float | Float = 0.0,
) -> Float:
    pass
