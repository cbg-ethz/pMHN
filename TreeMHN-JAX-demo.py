from collections import Counter

import jax
import jax.numpy as jnp
import joblib
import numpy as np
import pmhn._trees._backend_jax as bj
import pmhn._trees._simulate as sim

dim = 30
theta = -3 * np.eye(dim)
n_trees: int = 100
min_size: int = 10
max_size: int = 15

dim = 3
theta = -3 * np.eye(dim)
n_trees: int = 100
min_size: int = 3
max_size: int = 5


def sample_trees():
    trees = []
    rng = np.random.default_rng(42)
    for i in range(n_trees):
        tree, _ = sim.generate_valid_tree(
            rng,
            theta=theta,
            sampling_time=1,
            mean_sampling_time=1,
            min_tree_size=min_size,
            max_tree_size=max_size,
        )
        trees.append(tree)
    return trees


trees = sample_trees()

print(Counter([a.size for a in trees]))


def _wrap(tree):
    return bj.wrap_tree(tree, n_genes=dim)[0]


import time

t0 = time.time()
wrapped_trees = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(_wrap)(tree) for tree in trees
)
t1 = time.time()

print(f"Wrapping time: {t1-t0:.2f}")


def logp(theta, tree):
    return bj.loglikelihood(theta, omega=jnp.zeros(dim), tree=tree)


jit_grad = jax.jit(jax.grad(logp))
jit_logp = jax.jit(logp)


def logp_total(theta):
    return jnp.sum(jnp.asarray([jit_logp(theta, wrap) for wrap in wrapped_trees]))


def grad_total(theta):
    return jnp.sum(jnp.asarray([jit_grad(theta, wrap) for wrap in wrapped_trees]))


logp_total = jax.jit(logp_total)


# Call for the first time for JIT compilation
t0 = time.time()

_ = logp_total(jnp.eye(dim))
_ = grad_total(jnp.eye(dim))

t1 = time.time()

print(f"Compilation time: {t1-t0:.2f}")

# See how quickly JAX finds likelihood
n_reps: int = 3

t0 = time.time()

ps = []
for i in range(n_reps):
    theta = jnp.eye(dim) + i / n_reps
    ps.append(logp_total(theta))

repr(ps)

t1 = time.time()

average_time = (t1 - t0) / n_reps
print(f"JAX logp evaluation time: {average_time:.2f}")

t0 = time.time()
gs = []
for i in range(n_reps):
    theta = jnp.eye(dim) + i / n_reps
    gs.append(grad_total(theta))

repr(gs)
t1 = time.time()
print(f"JAX grad evaluation time: {average_time:.2f}")

# Compare with original implementation

import pmhn._trees._backend_code as bc

wrappers = [bc.TreeWrapperCode(t) for t in trees]
backend = bc.TreeMHNBackendCode()


def get_loglike(theta):
    return np.sum(
        [
            backend.loglikelihood(
                w, theta, sampling_rate=1.0, all_mut=set(range(1, dim + 1))
            )
            for w in wrappers
        ]
    )


n_reps: int = 3

t0 = time.time()

ps = []
for i in range(n_reps):
    theta = np.eye(dim) + i / n_reps
    ps.append(get_loglike(theta))


t1 = time.time()

average_time = (t1 - t0) / n_reps
print(f"Python evaluation time: {average_time:.2f}")

# See if both versions agree

import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(1, 2, figsize=(8, 3), dpi=250)

rng = np.random.default_rng(21)

thetas = [
    -np.eye(dim, dtype=np.float64),
    rng.normal(size=(dim, dim)),
    3 * rng.normal(size=(dim, dim)),
]


backend = bc.TreeMHNBackendCode(jitter=1e-20)
backend2 = bc.TreeMHNBackendCode(jitter=1e-10)

for i, theta in enumerate(thetas):
    jax_vals = np.asarray([jit_logp(theta, t) for t in wrapped_trees])

    pyt_vals = np.asarray(
        [
            backend.loglikelihood(
                t, theta, sampling_rate=1.0, all_mut=set(range(1, dim + 1))
            )
            for t in wrappers
        ]
    )

    pyt2_vals = np.asarray(
        [
            backend2.loglikelihood(
                t, theta, sampling_rate=1.0, all_mut=set(range(1, dim + 1))
            )
            for t in wrappers
        ]
    )

    ax = axs[0]
    ax.plot(
        pyt_vals,
        pyt_vals,
        c="k",
        alpha=1.0,
        rasterized=True,
        linestyle="--",
        linewidth=0.1,
    )
    ax.scatter(pyt_vals, jax_vals, s=2, rasterized=True, c=f"C{i}")

    ax = axs[1]
    ax.plot(
        pyt_vals,
        pyt_vals,
        c="k",
        alpha=1.0,
        rasterized=True,
        linestyle="--",
        linewidth=0.1,
    )
    ax.scatter(pyt_vals, pyt2_vals, s=2, rasterized=True, c=f"C{i}")

ax = axs[0]
ax.set_xlabel("$\\log P_\\text{Python}$\njitter=$10^{-20}$")
ax.set_ylabel("$\\log P_\\text{JAX}$")

ax = axs[1]
ax.set_xlabel("$\\log P_\\text{Python}$\njitter=$10^{-20}$")
ax.set_ylabel("$\\log P_\\text{Python}$\njitter=$10^{-10}$")

# ax.set_aspect('equal', 'box')
fig.tight_layout()
fig.savefig("jax_vs_python.pdf")
