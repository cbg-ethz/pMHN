import jax.numpy as jnp
import numpy.testing as npt
import pmhn._trees._backend_jax._private_api as api


def test_to_dense() -> None:
    mat = api.COOMatrix(
        diagonal=1.0 * jnp.arange(1, 4),
        offdiagonal=api.Values(
            start=jnp.asarray([0, 1]),
            end=jnp.asarray([1, 2]),
            value=jnp.asarray([8.0, 11.0]),
        ),
        fill_value=0.0,
    )

    npt.assert_allclose(
        mat.to_dense(),
        jnp.asarray([[1.0, 8.0, 0.0], [0.0, 2.0, 11.0], [0.0, 0.0, 3.0]]),
    )
