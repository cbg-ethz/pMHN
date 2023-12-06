from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


class Values(NamedTuple):
    start: Int[Array, " K"]
    end: Int[Array, " K"]
    value: Float[Array, " K"]

    @property
    def size(self) -> int:
        return self.start.shape[0]


def values_from_dict(d: dict[tuple[int, int], float]) -> Values:
    """Constructs a `Values` object from a dictionary.

    Note:
        This function is NOT JIT-compatible.
    """
    starts = []
    ends = []
    values = []

    for (s, e), v in d.items():
        starts.append(s)
        ends.append(e)
        values.append(v)

    return Values(
        start=jnp.asarray(starts), end=jnp.asarray(ends), value=jnp.asarray(values)
    )


class COOMatrix(NamedTuple):
    diagonal: Float[Array, " size"]
    offdiagonal: Values
    fill_value: float | Float

    @property
    def size(self) -> int:
        return self.diagonal.shape[0]

    def to_dense(self) -> Float[Array, "size size"]:
        """Converts a COO matrix to a dense matrix.

        Note:
            We parametrize this matrix as Q[start, end].
            Depending on the convention used, you may prefer
            to transpose it.
        """
        # Fill the matrix with the fill value
        a = jnp.full((self.size, self.size), fill_value=self.fill_value)

        # Iterate over the diagonal terms
        def _diag_loop_body(
            i: int, a: Float[Array, "size size"]
        ) -> Float[Array, "size size"]:
            return a.at[i, i].set(self.diagonal[i])

        a = jax.lax.fori_loop(0, self.size, _diag_loop_body, a)

        # Iterate over the off-diagonal terms
        def _offdiag_loop_body(
            i: int, a: Float[Array, "size size"]
        ) -> Float[Array, "size size"]:
            return a.at[self.offdiagonal.start[i], self.offdiagonal.end[i]].set(
                self.offdiagonal.value[i]
            )

        a = jax.lax.fori_loop(0, self.offdiagonal.size, _offdiag_loop_body, a)
        return a

    def get_vector_fixed_end(self, end: int) -> Float[Array, " size"]:
        """Returns the vector Q[:, end] with fixed `end` and varying `start`."""

        # First, we will go over the off-diagonal entries
        def body_fun(i, carry):
            # Unwrap the right entry
            start = self.offdiagonal.start[i]

            # If `end`s match, we want to overwrite carry[start].
            # Otherwise we leave it untouched
            set_value = jax.lax.select(
                end == self.offdiagonal.end[i],
                self.offdiagonal.value[i],
                carry[start],
            )
            return carry.at[start].set(set_value)

        a = jax.lax.fori_loop(
            lower=0,
            upper=self.offdiagonal.size,
            body_fun=body_fun,
            init_val=jnp.full(shape=(self.size,), fill_value=self.fill_value),
        )

        # Finally, we will update the diagonal entry
        return a.at[end].set(self.diagonal[end])
