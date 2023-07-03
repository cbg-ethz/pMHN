import numpy as np


def construct_matrix(diag: np.ndarray, offdiag: np.ndarray) -> np.ndarray:
    """Constructs a square matrix from diagonal and offdiagonal terms.

    Args:
        diag: array of shape (n,)
        offdiag: array of shape (n, n-1)

    Returns:
        embedding, array of shape (n, n)
          with the diagonal `diag`
          with the offdiagonal term at (k, i)
            given by offdiag[k, j(i)],
          where j(i) = i if i < diagonal_index
            and then skips it for i > diagonal_index
    """
    n = len(diag)
    assert offdiag.shape == (n, n - 1)

    embedding = np.zeros((n, n), dtype=diag.dtype)

    for i in range(n):
        # Before the diagonal
        if i > 0:
            embedding[i, :i] = offdiag[i, :i]
        # After the diagonal
        if i < n - 1:
            embedding[i, i + 1 :] = offdiag[i, i:]

    np.fill_diagonal(embedding, diag)

    return embedding


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
    offdiag = rng.normal(
        loc=0.0, scale=offdiag_effect, size=(n_mutations, n_mutations - 1)
    )
    offdiag = np.where(rng.uniform(size=offdiag.shape) < p_offdiag, offdiag, 0.0)
    return construct_matrix(diag=diag, offdiag=offdiag)
