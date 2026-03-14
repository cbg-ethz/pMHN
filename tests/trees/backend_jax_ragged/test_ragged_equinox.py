import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from anytree import Node

eqx = pytest.importorskip("equinox")
optax = pytest.importorskip("optax")

from pmhn._trees._backend_jax_ragged import (  # noqa: E402
    BilinearLogRateModel,
    MLPLogRateModel,
    loglikelihood_many_with_rate_model,
    loglikelihood_with_rate_model,
    wrap_tree_ragged,
)
from pmhn._trees._backend_jax_ragged._rates import (  # noqa: E402
    _construct_path_lineage_multihot,
    _construct_path_log_transition_rates,
    _construct_path_log_transition_rates_model,
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


def test_mlp_rate_model_output_shape_and_dtype() -> None:
    model = MLPLogRateModel(
        n_genes=6, key=jax.random.PRNGKey(0), width_size=16, depth=1
    )
    lineage = jnp.asarray([1, 0, 1, 0, 0, 1], dtype=jnp.float32)
    out = model(lineage, jnp.asarray(3))

    assert out.shape == ()
    assert out.dtype == jnp.float32


def test_bilinear_model_matches_bilinear_path_rates(seed: int = 1) -> None:
    n_genes = 6
    wrapped, _ = wrap_tree_ragged(_tree_medium(), n_genes=n_genes)

    rng = np.random.default_rng(seed)
    theta = jnp.asarray(rng.normal(size=(n_genes, n_genes)))

    model = BilinearLogRateModel(theta=theta)
    expected = _construct_path_log_transition_rates(wrapped.paths, theta=theta)
    obtained = _construct_path_log_transition_rates_model(
        paths=wrapped.paths,
        n_genes=n_genes,
        log_rate_model=model,
    )

    npt.assert_allclose(obtained, expected, atol=1e-6)


def test_model_likelihood_is_jittable_and_gradients_finite(seed: int = 2) -> None:
    n_genes = 6
    wrapped, _ = wrap_tree_ragged(_tree_medium(), n_genes=n_genes)
    model = MLPLogRateModel(n_genes=n_genes, key=jax.random.PRNGKey(11))

    rng = np.random.default_rng(seed)
    omega = jnp.asarray(rng.normal(size=(n_genes,)))
    log_tau = jnp.asarray(-0.15)

    @eqx.filter_jit
    def eval_ll(m, om):
        return loglikelihood_with_rate_model(m, om, wrapped, log_tau=log_tau)

    value = eval_ll(model, omega)
    assert jnp.isfinite(value)

    def loss_fn(params):
        m, om = params
        return -loglikelihood_with_rate_model(m, om, wrapped, log_tau=log_tau)

    loss_val, grads = eqx.filter_value_and_grad(loss_fn)((model, omega))
    assert jnp.isfinite(loss_val)

    model_grads, omega_grad = grads
    leaves = jax.tree_util.tree_leaves(eqx.filter(model_grads, eqx.is_inexact_array))
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in leaves)
    assert jnp.all(jnp.isfinite(omega_grad))


def test_optax_smoke_one_dataset(seed: int = 3) -> None:
    n_genes = 6
    trees = [_tree_small(), _tree_medium(), _tree_large()]
    wrapped_trees = [wrap_tree_ragged(tree, n_genes=n_genes)[0] for tree in trees]

    model = MLPLogRateModel(n_genes=n_genes, key=jax.random.PRNGKey(31))
    rng = np.random.default_rng(seed)
    omega = jnp.asarray(rng.normal(size=(n_genes,)))

    def loss_fn(params):
        m, om = params
        ll = loglikelihood_many_with_rate_model(
            log_rate_model=m,
            omega=om,
            trees=wrapped_trees,
            log_tau=0.0,
        )
        return -jnp.mean(ll)

    params = (model, omega)
    opt = optax.adam(1e-2)
    opt_state = opt.init(eqx.filter(params, eqx.is_inexact_array))

    loss_initial = loss_fn(params)
    for _ in range(10):
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(
            grads,
            opt_state,
            params=eqx.filter(params, eqx.is_inexact_array),
        )
        params = eqx.apply_updates(params, updates)
        assert jnp.isfinite(loss_val)

    loss_final = loss_fn(params)
    assert jnp.isfinite(loss_final)
    assert loss_final <= loss_initial


def test_lineage_features_are_binary() -> None:
    wrapped, _ = wrap_tree_ragged(_tree_large(), n_genes=6)
    lineage = _construct_path_lineage_multihot(wrapped.paths, n_genes=6)
    assert jnp.all((lineage == 0.0) | (lineage == 1.0))
