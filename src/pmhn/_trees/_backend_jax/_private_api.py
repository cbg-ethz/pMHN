"""This is *private* API, which is convenient for testing and experimenting."""
import pmhn._trees._backend_jax._rates as rates
from pmhn._trees._backend_jax._loglikelihood import loglikelihood
from pmhn._trees._backend_jax._solver import logprob_forward_substitution
from pmhn._trees._backend_jax._sparse import (
    COOMatrix,
    Values,
    coo_matrix_from_array,
    values_from_dict,
)
from pmhn._trees._backend_jax._wrapper import (
    DoublyIndexedPaths,
    ExitPathsArray,
    IndexedPaths,
    WrappedTree,
    wrap_tree,
)

__all__ = [
    "WrappedTree",
    "wrap_tree",
    "IndexedPaths",
    "DoublyIndexedPaths",
    "ExitPathsArray",
    "COOMatrix",
    "Values",
    "rates",
    "logprob_forward_substitution",
    "loglikelihood",
    "values_from_dict",
    "coo_matrix_from_array",
]
