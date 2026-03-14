import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from anytree import Node
from pmhn._trees._backend_jax._loglikelihood import loglikelihood as loglikelihood_old
from pmhn._trees._backend_jax._wrapper import wrap_tree as wrap_tree_old
from pmhn._trees._backend_jax_ragged import loglikelihood as loglikelihood_new
from pmhn._trees._backend_jax_ragged import (
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


def test_ragged_path_payload_is_not_larger_than_padded_payload() -> None:
    trees = [_tree_small(), _tree_medium(), _tree_large()]
    n_genes = 8

    saw_strict_improvement = False

    for tree in trees:
        wrapped_old, _ = wrap_tree_old(tree, n_genes=n_genes)
        wrapped_new, _ = wrap_tree_ragged(tree, n_genes=n_genes)

        padded_payload = int(
            wrapped_old.diag_paths.path.size + wrapped_old.offdiag_paths.path.size
        )
        ragged_payload = int(wrapped_new.paths.events_flat.size)

        assert ragged_payload <= padded_payload
        if ragged_payload < padded_payload:
            saw_strict_improvement = True

    # On realistic trees with heterogeneous trajectories, ragged should improve.
    assert saw_strict_improvement


def test_optional_runtime_benchmark_smoke(seed: int = 42) -> None:
    if os.getenv("PMHN_RUN_BENCHMARKS") != "1":
        return

    n_genes = 8
    tree = _tree_large()
    wrapped_old, _ = wrap_tree_old(tree, n_genes=n_genes)
    wrapped_new, _ = wrap_tree_ragged(tree, n_genes=n_genes)

    rng = np.random.default_rng(seed)
    theta = jnp.asarray(rng.normal(size=(n_genes, n_genes)))
    omega = jnp.asarray(rng.normal(size=(n_genes,)))
    log_tau = jnp.asarray(0.0)

    old_fn = jax.jit(
        lambda th, om, lt: loglikelihood_old(th, om, wrapped_old, lt),  # noqa: E731
    )
    new_fn = jax.jit(
        lambda th, om, lt: loglikelihood_new(th, om, wrapped_new, lt),  # noqa: E731
    )

    # Warm-up (includes compilation).
    old_value = old_fn(theta, omega, log_tau)
    new_value = new_fn(theta, omega, log_tau)
    _ = old_value.block_until_ready()
    _ = new_value.block_until_ready()
    npt.assert_allclose(old_value, new_value, atol=1e-6)

    n_runs = 100

    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = old_fn(theta, omega, log_tau).block_until_ready()
    old_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = new_fn(theta, omega, log_tau).block_until_ready()
    new_time = time.perf_counter() - t0

    assert old_time > 0.0
    assert new_time > 0.0
    print(f"[benchmark] old_time={old_time:.6f}s new_time={new_time:.6f}s")
