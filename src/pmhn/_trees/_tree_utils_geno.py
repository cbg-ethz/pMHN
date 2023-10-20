from anytree import Node, LevelOrderGroupIter
from itertools import combinations, product
from typing import Optional


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


def create_subtree(
    original_root: Node, nodes_list: list[Node], all_nodes_root: list[Node]
) -> Node:
    nodes_dict = {}
    node_to_parent = {node: node.parent for node in nodes_list}

    for node in all_nodes_root:
        if node in nodes_list:
            parent_node = node_to_parent.get(node)
            nodes_dict[node] = Node(node.name, parent=nodes_dict.get(parent_node))

    return nodes_dict[original_root]


def get_subtrees(node: Node, memo: Optional[dict] = None) -> list[list[Node]]:
    """
    Creates a list of all subtrees of a tree.


    Args:
        node: the root node
    Returns:
           a list of subtrees where each subtree is a list of nodes
    """
    if memo is None:
        memo = {}

    if node in memo:
        return memo[node]

    if not node.children:
        memo[node] = [[node]]
        return [[node]]

    child_subtrees = [get_subtrees(child, memo) for child in node.children]

    combined_subtrees = all_combinations_of_elements(*child_subtrees)

    result_subtrees = [[node]] + [
        [node] + [item for sublist in combination for item in sublist]
        for combination in combined_subtrees
    ]

    memo[node] = result_subtrees

    return result_subtrees


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


def create_index_subclone_maps(
    tree: Node,
) -> tuple[dict[int, tuple[int]], dict[tuple[int], int]]:
    index_subclone_map = {}
    subclone_index_map = {}
    index = 0
    for level in LevelOrderGroupIter(tree):
        for node in level:
            index_subclone_map[index] = get_lineage(node)
            subclone_index_map[get_lineage(node)] = index
            index += 1
    return index_subclone_map, subclone_index_map


def create_genotype(
    size: int, subtree: Node, subclone_index_map: dict[tuple[int], int]
) -> tuple[tuple[Optional[Node], int], ...]:
    x = [(Node(None), int(0))] * size
    for level in LevelOrderGroupIter(subtree):
        for node in level:
            lineage = get_lineage(node)
            x[subclone_index_map[lineage]] = (node, 1)
    return tuple(x)


def create_genotype_subtree_map(
    root: Node,
) -> tuple[dict[tuple[tuple[Node, int]], tuple[int, int]], dict[int, tuple[int]]]:
    index_subclone_map, subclone_index_map = create_index_subclone_maps(root)
    subtree_genotype_node_map = {}
    all_node_lists = get_subtrees(root)
    all_nodes_root = all_node_lists[-1]
    all_node_lists_with_len = [
        (node_list, len(node_list)) for node_list in all_node_lists
    ]
    size = len(all_node_lists)
    for index, (node_list, node_list_len) in enumerate(all_node_lists_with_len):
        subtree = create_subtree(root, node_list, all_nodes_root)
        genotype_node = create_genotype(size, subtree, subclone_index_map)
        subtree_genotype_node_map[genotype_node] = (index, node_list_len)
    return subtree_genotype_node_map, index_subclone_map


if __name__ == "__main__":
    A = Node("0")
    B = Node("1", parent=A)
    C = Node("3", parent=A)
    D = Node("4", parent=B)
