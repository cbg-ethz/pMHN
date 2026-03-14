import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from anytree import Node
from pmhn._trees._backend_jax._loglikelihood import loglikelihood as loglikelihood_old
from pmhn._trees._backend_jax._wrapper import wrap_tree as wrap_tree_old
from pmhn._trees._backend_jax_ragged import loglikelihood as loglikelihood_new
from pmhn._trees._backend_jax_ragged import (
    loglikelihood_many,
    wrap_tree_ragged,
)


def _tree_small() -> Node:
    root = Node(0)
    Node(1, parent=root)
    return root


def _tree_medium() -> Node:
    root = Node(0)
    left = Node(1, parent=root)
    Node(2, parent=root)
    Node(3, parent=left)
    return root


def _tree_large() -> Node:
    root = Node(0)
    left = Node(1, parent=root)
    right = Node(2, parent=root)
    Node(3, parent=left)
    Node(4, parent=right)
    Node(5, parent=right)
    return root


def test_loglikelihood_matches_old_implementation(seed: int = 123) -> None:
    n_genes = 6
    tree = _tree_medium()
    wrapped_old, _ = wrap_tree_old(tree, n_genes=n_genes)
    wrapped_new, _ = wrap_tree_ragged(tree, n_genes=n_genes)

    rng = np.random.default_rng(seed)
    theta = jnp.asarray(rng.normal(size=(n_genes, n_genes)))
    omega = jnp.asarray(rng.normal(size=(n_genes,)))
    log_tau = -0.37

    old_value = loglikelihood_old(
        theta=theta,
        omega=omega,
        tree=wrapped_old,
        log_tau=log_tau,
    )
    new_value = loglikelihood_new(
        theta=theta,
        omega=omega,
        tree=wrapped_new,
        log_tau=log_tau,
    )

    npt.assert_allclose(new_value, old_value, atol=1e-6)


def test_loglikelihood_gradient_matches_old_implementation(seed: int = 7) -> None:
    n_genes = 5
    tree = _tree_medium()
    wrapped_old, _ = wrap_tree_old(tree, n_genes=n_genes)
    wrapped_new, _ = wrap_tree_ragged(tree, n_genes=n_genes)

    rng = np.random.default_rng(seed)
    theta = jnp.asarray(rng.normal(size=(n_genes, n_genes)))
    omega = jnp.asarray(rng.normal(size=(n_genes,)))
    log_tau = jnp.asarray(0.11)

    def old_fn(th, om, lt):
        return loglikelihood_old(theta=th, omega=om, tree=wrapped_old, log_tau=lt)

    def new_fn(th, om, lt):
        return loglikelihood_new(theta=th, omega=om, tree=wrapped_new, log_tau=lt)

    grad_old = jax.grad(old_fn, argnums=(0, 1, 2))(theta, omega, log_tau)
    grad_new = jax.grad(new_fn, argnums=(0, 1, 2))(theta, omega, log_tau)

    npt.assert_allclose(grad_new[0], grad_old[0], atol=1e-6)
    npt.assert_allclose(grad_new[1], grad_old[1], atol=1e-6)
    npt.assert_allclose(grad_new[2], grad_old[2], atol=1e-6)


def test_new_loglikelihood_is_jittable(seed: int = 22) -> None:
    n_genes = 5
    tree = _tree_medium()
    wrapped_new, _ = wrap_tree_ragged(tree, n_genes=n_genes)

    rng = np.random.default_rng(seed)
    theta = jnp.asarray(rng.normal(size=(n_genes, n_genes)))
    omega = jnp.asarray(rng.normal(size=(n_genes,)))
    log_tau = jnp.asarray(-0.2)

    def fn(th, om, lt):
        return loglikelihood_new(theta=th, omega=om, tree=wrapped_new, log_tau=lt)

    value_eager = fn(theta, omega, log_tau)
    value_jit = jax.jit(fn)(theta, omega, log_tau)
    grad_jit = jax.jit(jax.grad(fn, argnums=0))(theta, omega, log_tau)

    npt.assert_allclose(value_jit, value_eager, atol=1e-6)
    assert grad_jit.shape == theta.shape
    assert jnp.all(jnp.isfinite(grad_jit))


def test_apply_new_implementation_to_three_trees_of_different_sizes(
    seed: int = 5,
) -> None:
    n_genes = 6
    trees = [_tree_small(), _tree_medium(), _tree_large()]
    wrapped_old = [wrap_tree_old(tree, n_genes=n_genes)[0] for tree in trees]
    wrapped_new = [wrap_tree_ragged(tree, n_genes=n_genes)[0] for tree in trees]

    rng = np.random.default_rng(seed)
    theta = jnp.asarray(rng.normal(size=(n_genes, n_genes)))
    omega = jnp.asarray(rng.normal(size=(n_genes,)))
    log_tau = 0.09

    values_old = jnp.asarray(
        [
            loglikelihood_old(theta=theta, omega=omega, tree=tree, log_tau=log_tau)
            for tree in wrapped_old
        ]
    )

    # Heterogeneous tree sizes: evaluate in a Python loop via helper API.
    values_new = loglikelihood_many(
        theta=theta,
        omega=omega,
        trees=wrapped_new,
        log_tau=log_tau,
    )

    assert values_new.shape == (3,)
    npt.assert_allclose(values_new, values_old, atol=1e-6)
