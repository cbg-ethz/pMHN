import numpy as np
from anytree import Node
from pmhn._trees._backend_jax_ragged import wrap_tree_ragged
from pmhn._trees._tree_utils import construct_paths_matrix


def _decode_path(wrapped, path_id: int) -> tuple[int, ...]:
    events = np.asarray(wrapped.paths.events_flat)
    ptr = np.asarray(wrapped.paths.path_ptr)
    start = int(ptr[path_id])
    end = int(ptr[path_id + 1])
    return tuple(events[start:end].tolist())


def _build_tree() -> Node:
    root = Node(0)
    left = Node(1, parent=root)
    Node(2, parent=root)
    Node(3, parent=left)
    return root


def test_wrap_tree_ragged_offdiag_paths_match_original() -> None:
    tree = _build_tree()
    n_genes = 4

    wrapped, _ = wrap_tree_ragged(tree, n_genes=n_genes)
    paths_matrix = construct_paths_matrix(tree, n_genes=n_genes)

    expected = {
        (start, end): tuple(traj)[1:]
        for (start, end), traj in paths_matrix.offdiag.items()
    }
    obtained = {
        (int(start), int(end)): _decode_path(wrapped, int(path_id))
        for start, end, path_id in zip(
            np.asarray(wrapped.edge_start),
            np.asarray(wrapped.edge_end),
            np.asarray(wrapped.edge_path_id),
        )
    }

    assert expected == obtained


def test_wrap_tree_ragged_diag_and_exit_paths_match_original_multiset() -> None:
    tree = _build_tree()
    n_genes = 4

    wrapped, _ = wrap_tree_ragged(tree, n_genes=n_genes)
    paths_matrix = construct_paths_matrix(tree, n_genes=n_genes)

    expected = sorted(
        (subtree_id, tuple(traj)[1:])
        for subtree_id, trajs in enumerate(paths_matrix.diag)
        for traj in trajs
    )
    obtained_diag = sorted(
        (int(subtree_id), _decode_path(wrapped, int(path_id)))
        for subtree_id, path_id in zip(
            np.asarray(wrapped.diag_subtree_id),
            np.asarray(wrapped.diag_path_id),
        )
    )
    obtained_exit = sorted(
        (int(subtree_id), _decode_path(wrapped, int(path_id)))
        for subtree_id, path_id in zip(
            np.asarray(wrapped.exit_subtree_id),
            np.asarray(wrapped.exit_path_id),
        )
    )

    assert expected == obtained_diag
    assert expected == obtained_exit


def test_wrap_tree_ragged_layer_metadata_is_valid() -> None:
    tree = _build_tree()
    wrapped, _ = wrap_tree_ragged(tree, n_genes=4)

    node_layer = np.asarray(wrapped.node_layer)
    layer_ptr = np.asarray(wrapped.layer_ptr)
    edge_start = np.asarray(wrapped.edge_start)
    edge_end = np.asarray(wrapped.edge_end)

    assert layer_ptr[0] == 0
    assert layer_ptr[-1] == wrapped.n_subtrees
    assert wrapped.n_layers == int(node_layer.max()) + 1

    for i in range(wrapped.n_layers):
        begin = int(layer_ptr[i])
        end = int(layer_ptr[i + 1])
        assert np.all(node_layer[begin:end] == i)

    assert np.all(node_layer[edge_end] == node_layer[edge_start] + 1)


def test_wrap_tree_ragged_pointer_and_last_event_consistency() -> None:
    tree = _build_tree()
    wrapped, _ = wrap_tree_ragged(tree, n_genes=4)

    ptr = np.asarray(wrapped.paths.path_ptr)
    events = np.asarray(wrapped.paths.events_flat)
    last = np.asarray(wrapped.paths.path_last_event)

    assert ptr[0] == 0
    assert ptr[-1] == events.shape[0]
    assert np.all(ptr[1:] > ptr[:-1])
    assert wrapped.paths.n_paths == last.shape[0]

    for path_id in range(wrapped.paths.n_paths):
        decoded = _decode_path(wrapped, path_id)
        assert len(decoded) > 0
        assert decoded[-1] == int(last[path_id])
