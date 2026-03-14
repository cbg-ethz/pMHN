from collections.abc import Iterable
from typing import NamedTuple

import jax.numpy as jnp
from anytree import Node
from jaxtyping import Array, Int

from pmhn._trees._tree_utils import construct_paths_matrix

_RawTrajectory = tuple[int, ...] | list[int]


class RaggedPaths(NamedTuple):
    """Ragged representation of a collection of mutation trajectories."""

    events_flat: Int[Array, " n_events"]
    path_ptr: Int[Array, " n_paths_plus_one"]
    path_last_event: Int[Array, " n_paths"]

    @property
    def n_paths(self) -> int:
        return self.path_ptr.shape[0] - 1


class RaggedTree(NamedTuple):
    """Ragged tree wrapper used by the JAX backend."""

    paths: RaggedPaths
    edge_start: Int[Array, " n_edges"]
    edge_end: Int[Array, " n_edges"]
    edge_path_id: Int[Array, " n_edges"]
    diag_subtree_id: Int[Array, " n_diag"]
    diag_path_id: Int[Array, " n_diag"]
    exit_subtree_id: Int[Array, " n_exit"]
    exit_path_id: Int[Array, " n_exit"]
    node_layer: Int[Array, " n_subtrees"]
    layer_ptr: Int[Array, " n_layers_plus_one"]
    n_genes: int

    @property
    def n_subtrees(self) -> int:
        return self.node_layer.shape[0]

    @property
    def n_layers(self) -> int:
        return self.layer_ptr.shape[0] - 1


def _strip_root(traj: _RawTrajectory) -> tuple[int, ...]:
    stripped = tuple(traj)[1:]
    if len(stripped) == 0:
        raise ValueError(
            "Encountered an empty trajectory after removing root. "
            "All encoded trajectories must include at least one mutation."
        )
    return stripped


def _layer_ptr_from_layers(node_layer: list[int]) -> list[int]:
    if len(node_layer) == 0:
        return [0]

    max_layer = max(node_layer)
    counts = [0 for _ in range(max_layer + 1)]

    for layer_idx in node_layer:
        if layer_idx < 0:
            raise ValueError(f"Layer index has to be non-negative, got {layer_idx}.")
        counts[layer_idx] += 1

    layer_ptr = [0]
    running = 0
    for c in counts:
        running += c
        layer_ptr.append(running)
    return layer_ptr


def _iterate_diag_paths(
    diag_paths: list[list[_RawTrajectory]],
) -> Iterable[tuple[int, tuple[int, ...]]]:
    for subtree_id, traj_list in enumerate(diag_paths):
        for traj in sorted(map(tuple, traj_list)):
            yield subtree_id, _strip_root(traj)


def wrap_tree_ragged(tree: Node, n_genes: int) -> tuple[RaggedTree, list[Node]]:
    """Wraps a tree into a ragged representation with no path padding."""
    paths_object = construct_paths_matrix(tree, n_genes=n_genes)

    path_to_id: dict[tuple[int, ...], int] = {}
    events_flat: list[int] = []
    path_ptr: list[int] = [0]
    path_last_event: list[int] = []

    def get_path_id(traj: tuple[int, ...]) -> int:
        pid = path_to_id.get(traj)
        if pid is not None:
            return pid

        pid = len(path_to_id)
        path_to_id[traj] = pid
        events_flat.extend(traj)
        path_ptr.append(len(events_flat))
        path_last_event.append(traj[-1])
        return pid

    edge_start: list[int] = []
    edge_end: list[int] = []
    edge_path_id: list[int] = []

    for (start, end), traj in sorted(paths_object.offdiag.items()):
        if start >= end:
            raise ValueError(f"Expected start < end, got ({start}, {end}).")
        traj_wo_root = _strip_root(traj)
        edge_start.append(start)
        edge_end.append(end)
        edge_path_id.append(get_path_id(traj_wo_root))

    diag_subtree_id: list[int] = []
    diag_path_id: list[int] = []

    for subtree_id, traj in _iterate_diag_paths(paths_object.diag):
        diag_subtree_id.append(subtree_id)
        diag_path_id.append(get_path_id(traj))

    # Exit terms mirror the diagonal path specification.
    exit_subtree_id = list(diag_subtree_id)
    exit_path_id = list(diag_path_id)

    node_layer = [subtree.size - 1 for subtree in paths_object.indices]
    if any(node_layer[i] > node_layer[i + 1] for i in range(len(node_layer) - 1)):
        raise ValueError("Subtrees are expected to be sorted by non-decreasing size.")
    layer_ptr = _layer_ptr_from_layers(node_layer)

    wrapped = RaggedTree(
        paths=RaggedPaths(
            events_flat=jnp.asarray(events_flat, dtype=int),
            path_ptr=jnp.asarray(path_ptr, dtype=int),
            path_last_event=jnp.asarray(path_last_event, dtype=int),
        ),
        edge_start=jnp.asarray(edge_start, dtype=int),
        edge_end=jnp.asarray(edge_end, dtype=int),
        edge_path_id=jnp.asarray(edge_path_id, dtype=int),
        diag_subtree_id=jnp.asarray(diag_subtree_id, dtype=int),
        diag_path_id=jnp.asarray(diag_path_id, dtype=int),
        exit_subtree_id=jnp.asarray(exit_subtree_id, dtype=int),
        exit_path_id=jnp.asarray(exit_path_id, dtype=int),
        node_layer=jnp.asarray(node_layer, dtype=int),
        layer_ptr=jnp.asarray(layer_ptr, dtype=int),
        n_genes=n_genes,
    )

    return wrapped, paths_object.indices
