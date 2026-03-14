import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pmhn._trees._backend_jax._rates as rates_padded
import pmhn._trees._backend_jax_ragged._rates as rates_ragged
from anytree import Node
from pmhn._trees._backend_jax._wrapper import wrap_tree
from pmhn._trees._backend_jax_ragged._wrapper import wrap_tree_ragged


def _decode_path(wrapped_ragged, path_id: int) -> tuple[int, ...]:
    ptr = np.asarray(wrapped_ragged.paths.path_ptr)
    events = np.asarray(wrapped_ragged.paths.events_flat)
    return tuple(events[ptr[path_id] : ptr[path_id + 1]].tolist())


def _build_tree() -> Node:
    root = Node(0)
    left = Node(1, parent=root)
    Node(2, parent=root)
    Node(3, parent=left)
    return root


def test_path_log_transition_rates_match_padded_reference(seed: int = 7) -> None:
    tree = _build_tree()
    n_genes = 5
    wrapped_ragged, _ = wrap_tree_ragged(tree, n_genes=n_genes)

    rng = np.random.default_rng(seed)
    theta = jnp.asarray(rng.normal(size=(n_genes, n_genes)))
    extended_theta = rates_padded._extend_theta(theta)

    obtained = rates_ragged._construct_path_log_transition_rates(
        paths=wrapped_ragged.paths, theta=theta
    )
    expected = jnp.asarray(
        [
            rates_padded._construct_log_transtion_rate(
                jnp.asarray(_decode_path(wrapped_ragged, path_id), dtype=int),
                extended_theta,
            )
            for path_id in range(wrapped_ragged.paths.n_paths)
        ]
    )

    npt.assert_allclose(obtained, expected, atol=1e-6)


def test_path_log_exit_rates_match_padded_reference(seed: int = 9) -> None:
    tree = _build_tree()
    n_genes = 5
    wrapped_ragged, _ = wrap_tree_ragged(tree, n_genes=n_genes)

    rng = np.random.default_rng(seed)
    omega = jnp.asarray(rng.normal(size=(n_genes,)))
    extended_omega = rates_padded._extend_omega(omega)

    obtained = rates_ragged._construct_path_log_exit_rates(
        paths=wrapped_ragged.paths, omega=omega
    )
    expected = jnp.asarray(
        [
            rates_padded._construct_log_exit_rate(
                jnp.asarray(_decode_path(wrapped_ragged, path_id), dtype=int),
                extended_omega,
            )
            for path_id in range(wrapped_ragged.paths.n_paths)
        ]
    )

    npt.assert_allclose(obtained, expected, atol=1e-6)


def test_construct_log_magic_matrix_parity_with_padded_backend(
    seed: int = 123,
) -> None:
    tree = _build_tree()
    n_genes = 5
    wrapped_padded, _ = wrap_tree(tree, n_genes=n_genes)
    wrapped_ragged, _ = wrap_tree_ragged(tree, n_genes=n_genes)

    rng = np.random.default_rng(seed)
    theta = jnp.asarray(rng.normal(size=(n_genes, n_genes)))
    omega = jnp.asarray(rng.normal(size=(n_genes,)))
    log_tau = 0.27

    magic_padded = rates_padded._construct_log_magic_matrix(
        tree=wrapped_padded,
        theta=theta,
        omega=omega,
        log_tau=log_tau,
    )
    magic_ragged = rates_ragged._construct_log_magic_matrix(
        tree=wrapped_ragged,
        theta=theta,
        omega=omega,
        log_tau=log_tau,
    )

    npt.assert_allclose(magic_ragged.to_dense(), magic_padded.to_dense(), atol=1e-6)


def test_segment_logsumexp_safe_for_all_inf_and_empty_segments() -> None:
    data = jnp.asarray([-jnp.inf, -jnp.inf, 0.0, -3.0])
    segment_ids = jnp.asarray([0, 0, 2, 2])

    obtained = rates_ragged.segment_logsumexp(
        data=data,
        segment_ids=segment_ids,
        num_segments=4,
    )

    expected = jnp.asarray(
        [
            -jnp.inf,  # all -inf
            -jnp.inf,  # empty segment
            jnp.log(jnp.exp(0.0) + jnp.exp(-3.0)),
            -jnp.inf,  # empty segment
        ]
    )

    assert not jnp.any(jnp.isnan(obtained))
    npt.assert_allclose(obtained, expected, atol=1e-6)


def test_segment_logsumexp_gradients_are_finite() -> None:
    segment_ids = jnp.asarray([0, 0, 1, 2])

    def objective(x):
        data = jnp.asarray([x[0], x[1], -jnp.inf, x[2]])
        y = rates_ragged.segment_logsumexp(
            data=data,
            segment_ids=segment_ids,
            num_segments=3,
        )
        return jnp.where(jnp.isfinite(y), y, 0.0).sum()

    x = jnp.asarray([-1000.0, -1001.0, -999.0])
    g = jax.grad(objective)(x)
    assert jnp.all(jnp.isfinite(g))
