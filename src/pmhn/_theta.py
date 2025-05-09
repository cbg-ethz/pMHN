import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax import random
from numpyro.distributions import constraints


class SpikeAndSlab(dist.Distribution):
    arg_constraints = {
        "w": constraints.unit_interval,
        "scale_spike": constraints.positive,
        "scale_slab": constraints.positive,
        "features": constraints.nonnegative_integer,
    }
    support = constraints.real_vector
    # no .has_rsample because sample is not reparameterized in JAX sense

    def __init__(
        self,
        w: float,
        scale_spike: float,
        scale_slab: float,
        features: int,
        validate_args=None,
    ):
        """
        w:   probability of slab component (active)
        scale_spike: std of spike
        scale_slab:  std of slab
        features:    dimensionality (n_features)
        """
        self.w = w
        self.scale_spike = scale_spike
        self.scale_slab = scale_slab
        self.features = features
        batch_shape = jnp.shape(w)
        event_shape = (features,)

        self._spike_dist = dist.Independent(
            dist.Normal(jnp.zeros(features), scale_spike), 1
        )
        print(self._spike_dist.batch_shape, self._spike_dist.event_shape)
        self._slab_dist = dist.Independent(
            dist.Normal(jnp.zeros(features), scale_slab), 1
        )

        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        """
        Draws x ~ mixture, but does NOT expose the indicators to NumPyro.
        """
        # total shape = sample_shape + batch_shape + event_shape
        shape = sample_shape + self.batch_shape + (self.features,)
        # draw indicators z_ij ∈ {0,1}
        key_z, key_eps = random.split(key)
        z = random.bernoulli(key_z, p=self.w, shape=shape)
        # draw standard normals
        eps = random.normal(key_eps, shape=shape)
        # where z==1 use slab, else use spike
        return jnp.where(z, eps * self.scale_slab, eps * self.scale_spike)

    def log_prob(self, x):
        """
        log p(x) = sum_i log[ w * N(0,scale_slab^2)(x_i)
                           + (1-w) * N(0,scale_spike^2)(x_i) ]
        """
        # shape: batch_shape + (features,)
        logp_slab = self._slab_dist.log_prob(x) + jnp.log(self.w)
        logp_spike = self._spike_dist.log_prob(x) + jnp.log1p(-self.w)
        # log-sum-exp over the two mixture components, then sum over features
        log_mix = jnp.logaddexp(logp_spike, logp_slab)
        return log_mix

    @property
    def mean(self):
        return jnp.zeros(self.batch_shape + (self.features,))

    @property
    def variance(self):
        # Var[X] = w*σ_slab^2 + (1-w)*σ_spike^2
        v = self.w * self.scale_slab**2 + (1 - self.w) * self.scale_spike**2
        return jnp.broadcast_to(v, self.batch_shape + (self.features,))


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
