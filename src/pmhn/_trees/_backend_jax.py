"""JAX version of TreeMHN"""
from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp
from anytree import Node
from jaxtyping import Array, Float, Int

from pmhn._trees._tree_utils import (
    _OffdiagDict,
    _OndiagList,
    _RawTraj,
    construct_paths_matrix,
)


def _extend_theta(theta: Float[Array, "n n"]) -> Float[Array, "n+1 n+1"]:
    """Adds a new row and column to theta, filled with zeros,
    to represent a mock gene not affecting any rates."""
    n = theta.shape[0]
    ret = jnp.zeros((n + 1, n + 1))
    return ret.at[:n, :n].set(theta)


def _get_pad_value(n_genes: int) -> int:
    """Returns the value to use for padding the trajectories.
    As `_extend_theta` adds the mock gene as the last row and column,
    we should pad with the `n_genes+1` value.
    """
    return n_genes + 1


_OffdiagAuxArray = Int[Array, "n_entries n_events"]


def _pad_trajectory(
    traj: _RawTraj,
    total_length: int,
    pad_value: int,
) -> list[int]:
    """Pads a trajectory from the left with `pad_value`
    so that `total_length` is reached."""
    n_add = total_length - len(traj)
    # Pad with additional values and remember to remove the 0 (wildtype)
    return [pad_value] * n_add + list(traj)[1:]


def _prepare_offdiag(
    offdiag: _OffdiagDict,
    pad_value: int,
) -> _OffdiagAuxArray:
    """Prepares a JAX array suitable for
    calculation of the off-diagonal terms from `theta`.

    Returns:
        array of shape (n_entries, 2 + n_events)
          where each row is of equal length 2 + n_events
          and has the format (i, j, trajectory)
          where (i, j) is the offdiagonal index
          and trajectory is the trajectory used to calculate the
          rate.

          This trajectory contains the mutation names (and excludes)
          the wildtype (0) and is padded from the *left* with `pad_value`
    """
    max_traj_length = max(len(v) for v in offdiag.values())

    ret = []

    for (i, j), traj in offdiag.items():
        # Pad the trajectory
        padded_traj = _pad_trajectory(
            traj, total_length=max_traj_length, pad_value=pad_value
        )
        # Add a row containing the indices and the padded trajectory
        entry = [i, j] + list(padded_traj)
        ret.append(entry)

    return jnp.asarray(ret, dtype=int)


def _construct_log_lambda(
    traj: Int[Array, " n_events"],
    extended_theta: Float[Array, "n+1 n+1"],
) -> Float:
    new_mut = traj[-1]
    return jnp.sum(extended_theta[new_mut - 1, traj - 1])


def _construct_offdiag_lambda(
    traj: Int[Array, " n_events"],
    extended_theta: Float[Array, "n+1 n+1"],
) -> Float:
    """Constructs the off-diagonal lambda for a given trajectory."""
    return jnp.exp(_construct_log_lambda(traj, extended_theta=extended_theta))


def _construct_offdiag_q_matrix(
    n_subtrees: int,
    offdiag: _OffdiagAuxArray,
    extended_theta: Float[Array, "n+1 n+1"],
) -> Float[Array, "n_subtrees n_subtrees"]:
    """Constructs the off-diagonal terms
    of the $Q$ matrix.
    The diagonal terms are zero.

    Note:
        Because `n_subtrees` is an argument, this function cannot be JITted.
        If needed, you can mark it as static using `functools.partial`:
        ```{python}
        from functools import partial

        new_func = partial(jax.jit, static_argnums=0)(_construct_offdiag_q_matrix)
        ```
    """

    def f(q: jax.Array, ext_traj: jax.Array) -> tuple[jax.Array, None]:
        i = ext_traj[0]
        j = ext_traj[1]
        traj = ext_traj[2:]
        return (
            q.at[i, j].set(
                _construct_offdiag_lambda(traj, extended_theta=extended_theta)
            ),
            None,
        )

    ret, _ = jax.lax.scan(
        f,
        jnp.zeros((n_subtrees, n_subtrees)),
        offdiag,
    )

    return ret


_OndiagAuxArray = Int[Array, "n_subtrees n_traj n_events"]


def _fake_traj(length: int, value: int) -> _RawTraj:
    """Prepares a trajectory of a given `length`
    with constant `value`.
    """
    return [value] * length


def _prepare_ondiag(
    ondiag: _OndiagList,
    pad_value: int,
) -> _OndiagAuxArray:
    max_n_traj = max(len(traj_list) for traj_list in ondiag)
    max_length = max(max(len(traj) for traj in traj_list) for traj_list in ondiag)

    ret = []
    for traj_list in ondiag:
        entry = []
        for traj in traj_list:
            # Add padded trajectories
            entry.append(
                _pad_trajectory(traj, total_length=max_length, pad_value=pad_value)
            )
        # Pad with fake trajectories.
        # Note that we drop 0, so we have to subtract 1 from the length
        entry.extend(
            [_fake_traj(length=max_length - 1, value=pad_value)]
            * (max_n_traj - len(traj_list))
        )

        ret.append(entry)

    return jnp.asarray(ret, dtype=int)


def _construct_diag_q_matrix(
    ondiag: _OndiagAuxArray, extended_theta: Float[Array, "n+1 n+1"]
) -> Float[Array, "n_subtrees n_subtrees"]:
    """Constructs the diagonal terms of the $Q$ matrix."""
    _log_lambd_diag = jnp.apply_along_axis(
        partial(_construct_log_lambda, extended_theta=extended_theta),
        -1,
        ondiag,
    )
    n_subtrees = ondiag.shape[0]
    return -jnp.exp(jax.scipy.special.logsumexp(_log_lambd_diag, axis=-1)) * jnp.eye(
        n_subtrees
    )


WrappedTree = namedtuple("WrappedTree", ["ondiag", "offdiag"])


def get_q_matrix(
    theta: Float[Array, "n n"],
    wrapped_tree: WrappedTree,
) -> Float[Array, "n_subtrees n_subtrees"]:
    """Creates the $Q$ matrix from the parameters."""
    extended_theta = _extend_theta(theta)

    ondiag = wrapped_tree.ondiag
    offdiag = wrapped_tree.offdiag

    n_subtrees = ondiag.shape[0]

    q_offdiag = _construct_offdiag_q_matrix(
        n_subtrees=n_subtrees, offdiag=offdiag, extended_theta=extended_theta
    )
    q_ondiag = _construct_diag_q_matrix(ondiag=ondiag, extended_theta=extended_theta)

    return q_ondiag + q_offdiag


def _logp_from_q_mat(
    q_mat: Float[Array, "n_subtrees n_subtrees"],
    jitter: float = 1e-10,
) -> Float:
    n_subtrees = q_mat.shape[0]

    i_mat = jnp.eye(n_subtrees)
    v_mat = i_mat - q_mat

    e = jnp.zeros(n_subtrees, dtype=q_mat.dtype)
    e = e.at[0].set(1.0)

    x = jax.scipy.linalg.solve_triangular(v_mat, e, trans="T")

    return jnp.log(x[-1] + jitter)


def logp(
    theta: Float[Array, "n n"],
    wrapped_tree: WrappedTree,
    jitter: float = 1e-10,
) -> float:
    q_mat = get_q_matrix(theta=theta, wrapped_tree=wrapped_tree)
    return _logp_from_q_mat(q_mat, jitter)


def wrap_tree(tree: Node, n_genes: int) -> WrappedTree:
    """Wraps the tree into a format suitable for JAX.

    Returns:
        WrappedTree(ondiag, offdiag)
    """
    offdiag, diag = construct_paths_matrix(tree, n_genes=n_genes)

    pad_value = _get_pad_value(n_genes)

    return WrappedTree(
        ondiag=_prepare_ondiag(ondiag=diag, pad_value=pad_value),
        offdiag=_prepare_offdiag(offdiag=offdiag, pad_value=pad_value),
    )
