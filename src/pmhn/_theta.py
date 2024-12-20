import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist


def spike_and_slab(P, pi: float = 0.5, spike_std: float = 0.01, slab_std: float = 1.0):
    """Spike and slab prior over a vector with P components.

    Args:
        pi: probability of the feature being active (sampled from the slab component)
    """
    mixture_dist = dist.Categorical(
        probs=jnp.stack([pi * jnp.ones(P), (1 - pi) * jnp.ones(P)], axis=-1)
    )
    component_dist = dist.Normal(
        loc=jnp.zeros((P, 2)),
        scale=jnp.stack([slab_std * jnp.ones(P), spike_std * jnp.ones(P)], axis=-1),
    )
    return dist.MixtureSameFamily(mixture_dist, component_dist)


def construct_matrix(diag: jnp.ndarray, offdiag: jnp.ndarray) -> jnp.ndarray:
    """Constructs a square matrix from diagonal and offdiagonal terms.

    Args:
        diag: array of shape (n,)
        offdiag: array of shape (n, n-1)

    Returns:
        array of shape (n, n)
          with the diagonal `diag`
          with the offdiagonal term at (k, i)
            given by offdiag[k, j(i)],
          where j(i) = i if i < diagonal_index
            and then skips it for i > diagonal_index

    See Also:
        decompose_matrix, the inverse function.
    """
    n = diag.shape[0]
    matrix = jnp.diag(diag)  # Shape: (n, n)

    # Compute row indices: shape (n, 1) broadcasted to (n, n-1)
    row_indices = jnp.arange(n).reshape(n, 1)
    row_indices_broadcasted = jnp.broadcast_to(row_indices, (n, n - 1))

    # Compute column indices based on the row number
    # For each row i, columns are [0, 1, ..., i-1, i+1, ..., n-1]
    # This is achieved by:
    # - For each row i, elements in offdiag[i] correspond to columns:
    #   j < i => j
    #   j >= i => j + 1
    j_indices = jnp.arange(n - 1)
    col_indices = jnp.where(
        j_indices < row_indices, j_indices, j_indices + 1
    )  # Shape: (n, n-1)

    # Flatten the indices and offdiag for vectorized assignment
    row_indices_flat = row_indices_broadcasted.flatten()  # Shape: (n*(n-1),)
    col_indices_flat = col_indices.flatten()  # Shape: (n*(n-1),)
    offdiag_flat = offdiag.flatten()  # Shape: (n*(n-1),)

    # Assign off-diagonal values
    matrix = matrix.at[row_indices_flat, col_indices_flat].set(offdiag_flat)

    return matrix


def decompose_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Splits an (n, n) matrix into diagonal and offdiagonal terms.

    Args:
        matrix: array of shape (n, n)

    Returns:
        diag, diagonal terms, shape (n,)
        offdiag, offdiagonal terms, shape (n, n-1)

    See Also:
        construct_matrix, the inverse function.
    """
    n = matrix.shape[0]
    assert matrix.shape == (n, n), "Input matrix must be square."

    # Extract the diagonal elements
    diag = jnp.diag(matrix)

    # Create a boolean mask to exclude the diagonal elements
    mask_offdiag = ~jnp.eye(n, dtype=bool)

    # Apply the mask and reshape to (n, n-1)
    offdiag = matrix[mask_offdiag].reshape(n, n - 1)

    return diag, offdiag


def sample_spike_and_slab(
    rng,
    n_mutations: int,
    diag_mean: float = 0.0,
    diag_sigma: float = 1.0,
    offdiag_effect: float = 1.0,
    p_offdiag: float = 0.2,
) -> np.ndarray:
    """Samples a matrix using diagonal terms from a normal
    distribution and offdiagonal terms sampled from spike and slab
    distribution.

    Args:
        rng: NumPy random number generator.
        n_mutations: number of mutations.
        diag_mean: mean of the normal distribution used to sample
            diagonal terms.
        diag_scale: standard deviation of the normal distribution
            used to sample diagonal terms.
        offdiag_effect: the standard deviation of the slab used
            to sample non-zero offdiagonal terms
        p_offdiag: the probability of sampling a non-zero offdiagonal
            term.
    """
    assert n_mutations > 0, "n_mutations should be positive."
    assert 0.0 <= p_offdiag <= 1.0, "p_offdiag should be between 0 and 1."

    diag = rng.normal(loc=diag_mean, scale=diag_sigma, size=n_mutations)
    diag = np.sort(diag)[::-1]  # Sort from highest baseline effects to the smallest
    offdiag = rng.normal(
        loc=0.0, scale=offdiag_effect, size=(n_mutations, n_mutations - 1)
    )
    offdiag = np.where(rng.uniform(size=offdiag.shape) < p_offdiag, offdiag, 0.0)
    return construct_matrix(diag=diag, offdiag=offdiag)
