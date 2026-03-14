"""JAX backend for tree likelihoods.

Default exports are ragged-first (no path padding).
Legacy padded API remains available via `legacy_*` names.
"""

from pmhn._trees._backend_jax._loglikelihood import (
    loglikelihood as legacy_loglikelihood,
)
from pmhn._trees._backend_jax._sparse import COOMatrix, Values
from pmhn._trees._backend_jax._wrapper import (
    DoublyIndexedPaths as LegacyDoublyIndexedPaths,
)
from pmhn._trees._backend_jax._wrapper import ExitPathsArray as LegacyExitPathsArray
from pmhn._trees._backend_jax._wrapper import IndexedPaths as LegacyIndexedPaths
from pmhn._trees._backend_jax._wrapper import WrappedTree as LegacyWrappedTree
from pmhn._trees._backend_jax._wrapper import wrap_tree as legacy_wrap_tree
from pmhn._trees._backend_jax_ragged import (
    BilinearLogRateModel,
    MLPLogRateModel,
    RaggedPaths,
    RaggedTree,
    loglikelihood,
    loglikelihood_many,
    loglikelihood_many_with_rate_model,
    loglikelihood_with_rate_model,
    logprob_forward_substitution_layerwise,
    wrap_tree_ragged,
)

# Default migration aliases.
WrappedTree = RaggedTree
wrap_tree = wrap_tree_ragged
IndexedPaths = LegacyIndexedPaths
DoublyIndexedPaths = LegacyDoublyIndexedPaths
ExitPathsArray = LegacyExitPathsArray

__all__ = [
    # Default ragged API
    "RaggedPaths",
    "RaggedTree",
    "WrappedTree",
    "wrap_tree",
    "COOMatrix",
    "Values",
    "loglikelihood",
    "loglikelihood_many",
    "loglikelihood_with_rate_model",
    "loglikelihood_many_with_rate_model",
    "logprob_forward_substitution_layerwise",
    "BilinearLogRateModel",
    "MLPLogRateModel",
    # Legacy compatibility API
    "LegacyWrappedTree",
    "LegacyIndexedPaths",
    "LegacyDoublyIndexedPaths",
    "LegacyExitPathsArray",
    "legacy_wrap_tree",
    "legacy_loglikelihood",
    # Kept names for compatibility with existing type imports
    "IndexedPaths",
    "DoublyIndexedPaths",
    "ExitPathsArray",
]
