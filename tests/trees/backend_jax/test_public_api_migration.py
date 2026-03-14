import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pmhn._trees._backend_jax as backend_default
import pmhn._trees._backend_jax_ragged as backend_ragged
from anytree import Node


def _build_tree() -> Node:
    root = Node(0)
    left = Node(1, parent=root)
    Node(2, parent=root)
    Node(3, parent=left)
    return root


def test_default_backend_jax_wrap_tree_is_ragged() -> None:
    wrapped, _ = backend_default.wrap_tree(_build_tree(), n_genes=5)
    assert isinstance(wrapped, backend_ragged.RaggedTree)
    assert hasattr(wrapped.paths, "event_path_id")


def test_default_backend_jax_loglikelihood_matches_ragged(seed: int = 1) -> None:
    tree = _build_tree()
    wrapped_default, _ = backend_default.wrap_tree(tree, n_genes=5)
    wrapped_ragged, _ = backend_ragged.wrap_tree_ragged(tree, n_genes=5)

    rng = np.random.default_rng(seed)
    theta = jnp.asarray(rng.normal(size=(5, 5)))
    omega = jnp.asarray(rng.normal(size=(5,)))
    log_tau = -0.23

    val_default = backend_default.loglikelihood(
        theta=theta, omega=omega, tree=wrapped_default, log_tau=log_tau
    )
    val_ragged = backend_ragged.loglikelihood(
        theta=theta, omega=omega, tree=wrapped_ragged, log_tau=log_tau
    )
    npt.assert_allclose(val_default, val_ragged, atol=1e-6)


def test_legacy_backend_jax_api_still_available(seed: int = 3) -> None:
    tree = _build_tree()
    wrapped_legacy, _ = backend_default.legacy_wrap_tree(tree, n_genes=5)
    wrapped_default, _ = backend_default.wrap_tree(tree, n_genes=5)

    # Legacy wrapper keeps padded-path structure.
    assert hasattr(wrapped_legacy, "diag_paths")
    assert not hasattr(wrapped_default, "diag_paths")

    rng = np.random.default_rng(seed)
    theta = jnp.asarray(rng.normal(size=(5, 5)))
    omega = jnp.asarray(rng.normal(size=(5,)))
    log_tau = 0.1

    val_legacy = backend_default.legacy_loglikelihood(
        theta=theta, omega=omega, tree=wrapped_legacy, log_tau=log_tau
    )
    assert jnp.isfinite(val_legacy)
