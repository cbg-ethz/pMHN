from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


class Values(NamedTuple):
    start: Int[Array, " K"]
    end: Int[Array, " K"] | None
    value: Float[Array, " K"]

    def size(self) -> int:
        return self.start.shape[0]


class COOMatrix(NamedTuple):
    diagonal: Float[Array, " n_subtrees"]
    offdiagonal: Values
    size: int
    fill_value: float | Float

    def to_dense(self) -> Float[Array, "size size"]:
        """Converts a COO matrix to a dense matrix.

        Note:
            We parametrize this matrix as Q[start, end].
            Depending on the convention used, you may prefer
            to transpose it.
        """
        # TODO(Pawel): UNTESTED

        # Fill the matrix with the fill value
        a = jnp.full((self.size, self.size), fill_value=self.fill_value)

        # Iterate over the diagonal terms
        def _diag_loop_body(
            i: int, a: Float[Array, "size size"]
        ) -> Float[Array, "size size"]:
            return a.at[i, i].set(self.diagonal[i])

        a = jax.lax.fori_loop(0, self.diagonal.shape[0], _diag_loop_body, a)

        # Iterate over the off-diagonal terms
        def _offdiag_loop_body(
            i: int, a: Float[Array, "size size"]
        ) -> Float[Array, "size size"]:
            return a.at[self.offdiagonal.start[i], self.offdiagonal.end[i]].set(
                self.offdiagonal.value[i]
            )

        a = jax.lax.fori_loop(0, self.offdiagonal.size(), _offdiag_loop_body, a)
        return a
