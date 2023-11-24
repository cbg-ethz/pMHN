from itertools import combinations, product
from typing import Optional

from anytree import LevelOrderGroupIter, Node, PreOrderIter, RenderTree


def all_combinations_of_elements(*lists):
    """
    Takes a variable number of lists as input and returns a generator that yields
    all possible combinations of the input lists. In our use case: It takes a list 
    of lists of subtrees as input where a subtree itself is a list of nodes and 
    outputs all possible combinations of the lists of subtrees.

    For instance, if we have the following tree:
      
        0 
     /  |  \
    1   3   2   
    |      
    2
      
    and assumed that we know the list of subtrees for the trees:
      
        1  
        |    -> list of subtrees: [[1], [1, 2]]
        2

        3    -> list of subtrees: [[3]]

    and 

        2    -> list of subtrees: [[2]]

    , we can find the subtrees of the original tree by looking at
    all possible combinations of the list of subtrees for the trees above
    and add the root node (0) to each combination (this is done in the
    get_subtrees function).
    
    So the input would be [[[1], [1, 2]],[[3]], [[2]]] 
    
    The generator would yield the following combinations one at a time:
    [[1]], [[1, 2]], [[3]], [[2]], [[1], [3]], [[1, 2], [3]], [[1], [2]],
    [[1, 2], [2]], [[3], [2]], [[1], [3], [2]], [[1, 2], [3], [2]]
    
    Args:
        *lists: any number of lists

    Returns:
        A generator that yields all combinations of the input lists. 

    """
    n = len(lists)
    for r in range(1, n + 1):
        for list_combination in combinations(lists, r):
            for element_combination in product(*list_combination):
                yield list(element_combination)


def create_subtree(original_root: Node, nodes_list: list[Node]) -> Node:
    """
    Creates a subtree given a list of nodes and the root node.

    Args:
        original_root: the root node
        nodes_list: a list of nodes
    Returns:
           a subtree
    """
    nodes_dict = {}

    for node in [original_root] + list(original_root.descendants):
        if node in nodes_list:
            parent_node = next((n for n in nodes_list if n is node.parent), None)
            nodes_dict[node] = Node(node.name, parent=nodes_dict.get(parent_node))
    return nodes_dict[original_root]


def get_subtrees(node: Node) -> list[list[Node]]:
    """
    Creates a list of all subtrees of a tree.
    A recursive approach is employed: If one knows the subtrees of the
    children of the root node, then one can find all combinations of
    the subtrees of the children and add the root node to each one
    of these combinations, this way one obtains all subtrees of the original tree.

    Args:
        node: the root node
    Returns:
        a list of subtrees

    """
    if not node.children:
        return [[node]]

    child_subtrees = [get_subtrees(child) for child in node.children]

    combined_subtrees = all_combinations_of_elements(*child_subtrees)

    result_subtrees = [[node]] + [
        [node] + [item for sublist in combination for item in sublist]
        for combination in combined_subtrees
    ]

    return result_subtrees


def create_all_subtrees(root: Node) -> dict[Node, int]:
    """
    Creates a dictionary where each key is a subtree,
    and each value is the size of that subtree.


    Args:
        root: the root node
    Returns:
        A dictionary mapping subtrees to their sizes
    """
    all_node_lists = get_subtrees(root)
    all_node_lists = sorted(all_node_lists, key=len)
    all_subtrees_dict = {
        create_subtree(root, node_list): len(node_list) for node_list in all_node_lists
    }
    return all_subtrees_dict


def get_lineage(node: Node) -> tuple[int]:
    """
    Creates a tuple of the names of the nodes that
    are in the lineage of the input node.

    Args:
        node: a node
    Returns:
            the lineage of a node
    """
    return tuple(ancestor.name for ancestor in node.path)  # type: ignore


def check_equality(tree1: Optional[Node], tree2: Optional[Node]) -> bool:
    """
    Checks if tree1 and tree2 are identical, note that direct
    comparison with == is not possible.

    Args:
        tree1: the first tree
        tree2: the second tree
    Returns:
           (in)equality of the trees
    """
    iter1 = list(LevelOrderGroupIter(tree1))
    iter2 = list(LevelOrderGroupIter(tree2))
    if tree1 is not None and tree2 is not None:
        if len(tree1.descendants) != len(tree2.descendants):
            return False
    for nodes1, nodes2 in zip(iter1, iter2):
        set_nodes1_lineages = {get_lineage(node) for node in nodes1}
        set_nodes2_lineages = {get_lineage(node) for node in nodes2}

        additional_nodes_lineages = set_nodes2_lineages ^ set_nodes1_lineages
        if len(additional_nodes_lineages) != 0:
            return False

    return True


def bfs_compare(tree1: Node, tree2: Node) -> Optional[Node]:
    """
    Checks if tree1 is a subtree of tree2 with the assumption
    that tree2 is larger than the first tree by one.


    Args:
        tree1: the first tree
        tree2: the second tree
    Returns:
           the additional node in the second tree if available, otherwise None.

    """

    diff_count = 0
    iter1 = list(LevelOrderGroupIter(tree1))
    iter2 = list(LevelOrderGroupIter(tree2))
    exit_node = None

    for level, (nodes1, nodes2) in enumerate(zip(iter1, iter2)):
        dict_nodes1_lineages = {node: get_lineage(node) for node in nodes1}
        dict_nodes2_lineages = {node: get_lineage(node) for node in nodes2}
        set_nodes1_lineages = set(dict_nodes1_lineages.values())
        set_nodes2_lineages = set(dict_nodes2_lineages.values())

        additional_nodes_lineages = set_nodes2_lineages ^ set_nodes1_lineages
        diff_count += len(additional_nodes_lineages)

        if diff_count == 1 and exit_node is None:
            additional_node_lineage = additional_nodes_lineages.pop()

            for node in nodes1:
                if dict_nodes1_lineages[node] == additional_node_lineage:
                    return None

            for node in nodes2:
                if dict_nodes2_lineages[node] == additional_node_lineage:
                    exit_node = node
                    break

    if diff_count > 1:
        return None

    if diff_count == 0:
        return iter2[-1][0]

    return exit_node


_RawTraj = tuple[int, ...] | list[int]
_OffdiagDict = dict[tuple[int, int], _RawTraj]


def _construct_offdiag_paths(subtrees: list[Node]) -> _OffdiagDict:
    offdiag = {}

    for i, i_tree in enumerate(subtrees):
        for j, j_tree in enumerate(subtrees):
            if i == j:
                continue
            comp = bfs_compare(i_tree, j_tree)
            if comp is not None:
                offdiag[(i, j)] = get_lineage(comp)

    return offdiag


_OndiagList = list[list[_RawTraj]]


def _get_exit_trajectories(root: Node, n_mutations: int) -> list[_RawTraj]:
    """Constructs a list of trajectories resulting
    in an exit from the tree considered."""
    exit_trajs = []

    # For each node we will consider all potential children
    for node in PreOrderIter(root):
        current_lineage = list(get_lineage(node))

        # We can add mutations such that:
        #   1. Are not in the lineage
        #   2. Are not already children in the tree

        children_mutations = set([ch.name for ch in node.children])

        available = set(range(1, n_mutations + 1)).difference(
            children_mutations.union(current_lineage)
        )

        for new_mut in available:
            traj = tuple(current_lineage + [new_mut])
            exit_trajs.append(traj)

    return exit_trajs


def construct_paths_matrix(
    root: Node, n_genes: int
) -> tuple[_OffdiagDict, _OndiagList]:
    subtrees_dict = create_all_subtrees(root)
    subtrees = [x[0] for x in sorted(subtrees_dict.items(), key=lambda x: x[1])]

    offdiag = _construct_offdiag_paths(subtrees)
    diag_terms = [
        _get_exit_trajectories(subtree, n_mutations=n_genes) for subtree in subtrees
    ]

    return offdiag, diag_terms


if __name__ == "__main__":
    A = Node("0")
    B = Node("1", parent=A)
    C = Node("3", parent=A)
    D = Node("3", parent=B)

    print(RenderTree(A))
    print("\n")
    all_subtrees = create_all_subtrees(A)
    i = 1
    for subtree in all_subtrees:
        print(f"{i}. ")
        print(RenderTree(subtree))
        i += 1
