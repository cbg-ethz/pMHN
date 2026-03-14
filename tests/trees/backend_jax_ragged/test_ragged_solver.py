import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pmhn._trees._backend_jax._rates as rates_padded
from anytree import Node
from pmhn._trees._backend_jax._solver import logprob_forward_substitution as solver_old
from pmhn._trees._backend_jax._sparse import coo_matrix_from_array
from pmhn._trees._backend_jax._wrapper import wrap_tree as wrap_tree_old
from pmhn._trees._backend_jax_ragged._rates import _construct_log_magic_matrix
from pmhn._trees._backend_jax_ragged._solver import (
    logprob_forward_substitution_layerwise as solver_new,
)
from pmhn._trees._backend_jax_ragged._wrapper import wrap_tree_ragged


def _build_tree() -> Node:
    root = Node(0)
    left = Node(1, parent=root)
    Node(2, parent=root)
    Node(3, parent=left)
    return root


def _solve_dense_reference(log_magic_dense: np.ndarray) -> np.ndarray:
    """Dense reference solver for the transformed triangular system."""
    a = np.exp(log_magic_dense)
    size = a.shape[0]

    # Build B x = e0 where:
    #   B[i, i] = V_ii
    #   B[i, s] = -Q_{s, i} for s != i
    b = np.diag(np.diag(a))
    for s in range(size):
        for e in range(size):
            if s == e:
                continue
            b[e, s] = -a[s, e]

    rhs = np.zeros(size)
    rhs[0] = 1.0
    x = np.linalg.solve(b, rhs)
    return np.log(x)


def test_layerwise_solver_matches_old_solver(seed: int = 12) -> None:
    tree = _build_tree()
    n_genes = 5

    wrapped_old, _ = wrap_tree_old(tree, n_genes=n_genes)
    wrapped_new, _ = wrap_tree_ragged(tree, n_genes=n_genes)

    rng = np.random.default_rng(seed)
    theta = jnp.asarray(rng.normal(size=(n_genes, n_genes)))
    omega = jnp.asarray(rng.normal(size=(n_genes,)))
    log_tau = -0.21

    log_magic_old = rates_padded._construct_log_magic_matrix(
        tree=wrapped_old,
        theta=theta,
        omega=omega,
        log_tau=log_tau,
    )
    log_magic_new = _construct_log_magic_matrix(
        tree=wrapped_new,
        theta=theta,
        omega=omega,
        log_tau=log_tau,
    )

    old_values = solver_old(log_magic_old)
    new_values = solver_new(
        log_magic=log_magic_new,
        node_layer=wrapped_new.node_layer,
        layer_ptr=wrapped_new.layer_ptr,
    )
    npt.assert_allclose(new_values, old_values, atol=1e-6)


def test_layerwise_solver_matches_dense_reference() -> None:
    # Same synthetic system as the legacy solver test.
    q = 0.1 * np.asarray(
        [
            [0, 7, 5, 0],
            [0, 0, 2, 1],
            [0, 0, 0, 2],
            [0, 0, 0, 0],
        ]
    )
    q -= q.sum(axis=1) * np.eye(q.shape[0])

    v = np.eye(q.shape[0]) - q
    log_magic_dense = np.log(np.abs(v))
    log_magic = coo_matrix_from_array(jnp.asarray(log_magic_dense))

    # Nodes are ordered by increasing subtree size in practice; here use 1-node layers.
    node_layer = jnp.arange(log_magic.size, dtype=int)
    layer_ptr = jnp.arange(log_magic.size + 1, dtype=int)

    obtained = solver_new(
        log_magic=log_magic,
        node_layer=node_layer,
        layer_ptr=layer_ptr,
    )
    expected = _solve_dense_reference(log_magic_dense)

    npt.assert_allclose(obtained, expected, atol=1e-6)
