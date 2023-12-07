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


def test_get_vector_fixed_end() -> None:
    """We will check whether the right vector is retrieved.

    Consider a matrix

          --- end ---
    s  [ 3  5  0  0  0 ]
    t  [ 0  0  2  0  0 ]
    a  [ 1  0  5  7  9 ]
    r  [ 0  0  0  4  0 ]
    t  [ 0  0  0  2  1 ]
    """
    _offdiag = api.values_from_dict(
        {
            (0, 1): 5,
            (1, 2): 2,
            (2, 0): 1,
            (2, 3): 7,
            (2, 4): 9,
            (4, 3): 2,
        }
    )

    mat = api.COOMatrix(
        diagonal=jnp.asarray([3, 0, 5, 4, 1]),
        offdiagonal=_offdiag,
        fill_value=0,
    )

    _expected_dense = jnp.asarray(
        [
            [3, 5, 0, 0, 0],
            [0, 0, 2, 0, 0],
            [1, 0, 5, 7, 9],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 2, 1],
        ]
    )
    npt.assert_allclose(
        mat.to_dense(),
        _expected_dense,
    )

    npt.assert_allclose(mat.get_vector_fixed_end(end=0), jnp.asarray([3, 0, 1, 0, 0]))
    npt.assert_allclose(mat.get_vector_fixed_end(end=1), jnp.asarray([5, 0, 0, 0, 0]))
    npt.assert_allclose(mat.get_vector_fixed_end(end=2), jnp.asarray([0, 2, 5, 0, 0]))
    npt.assert_allclose(mat.get_vector_fixed_end(end=3), jnp.asarray([0, 0, 7, 4, 2]))
    npt.assert_allclose(mat.get_vector_fixed_end(end=4), jnp.asarray([0, 0, 9, 0, 1]))
