# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from pmhn import mhn


# +
class FullParams(NamedTuple):
    theta: jnp.ndarray
    omega: jnp.ndarray

    @staticmethod
    def theta_link(params, x):
        return params.theta

    @staticmethod
    def omega_link(params, x):
        return params.omega


rng = np.random.default_rng(42)

n_samples = 400
n_genes = 10

Y = rng.binomial(1, p=0.3, size=(n_samples, n_genes))
X = rng.normal(size=(n_samples, 2))

# +
rng = np.random.default_rng(101)

theta = np.zeros((n_genes, n_genes))
omega = np.zeros(n_genes)


theta = rng.normal(size=theta.shape)


theta = jnp.asarray(theta)
omega = jnp.asarray(omega)
# -

ll_fn = mhn.generate_loglikelihood(
    Y,
    X,
    theta_link_fn=FullParams.theta_link,
    omega_link_fn=FullParams.omega_link,
)

ll_fn(FullParams(jnp.eye(n_genes) + 0.5, jnp.zeros(n_genes)))

jax.grad(ll_fn)(FullParams(jnp.eye(n_genes) + 0.3, jnp.zeros(n_genes)))
