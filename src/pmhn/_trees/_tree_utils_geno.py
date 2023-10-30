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


def create_subtree(subtree_nodes: list[Node], original_tree_nodes: list[Node]) -> Node:
    """
    Creates a certain subtree of the original tree.

    Args:
        subtree_nodes: the nodes that are contained in both
        the subtree and the original tree
        original_tree_nodes: all nodes of the original tree
    Returns:
           a subtree
    """
    nodes_dict = {}

    for node in subtree_nodes:
        parent_node = node.parent
        nodes_dict[node] = Node(node.name, parent=nodes_dict.get(parent_node))

    return nodes_dict[original_tree_nodes[0]]


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
    root: Node,
) -> tuple[dict[int, tuple[int, ...]], dict[tuple[int, ...], int]]:
    """
    Assigns a unique index to each subclone in the provided
    tree and generates two dictionaries: one mapping each unique
    index to its corresponding subclone, and the other inverting this relationship.
    Args:
        root: the root node of a tree
    Returns:
        two dictionaries that contain the mappings
    """
    index_subclone_map = {}
    subclone_index_map = {}
    index = 0
    for level in LevelOrderGroupIter(root):
        for node in level:
            index_subclone_map[index] = get_lineage(node)
            subclone_index_map[get_lineage(node)] = index
            index += 1
    return index_subclone_map, subclone_index_map


def create_genotype(
    size: int, root: Node, subclone_index_map: dict[tuple[int, ...], int]
) -> tuple[tuple[Optional[Node], int], ...]:
    """
    Creates the genotype of a given tree.

    Args:
        size: the size of the original tree
        root: the root node of a subtree of the original tree
        subclone_index_map: a dictionary that maps subclones to their indices
    Returns:
        a tuple of tuples, where each inner tuple represents a subclone from
        the original tree. For each subclone, if it exists in the subtree,
        the inner tuple contains the last node of that subclone and the value 1;
        if it doesn't exist, the tuple contains None and the value 0.
    """
    x = [(Node(None), int(0))] * size
    for level in LevelOrderGroupIter(root):
        for node in level:
            lineage = get_lineage(node)
            x[subclone_index_map[lineage]] = (node, 1)
    return tuple(x)


def create_mappings(
    root: Node,
) -> tuple[
    dict[tuple[tuple[Node, int], ...], tuple[int, int]], dict[int, tuple[int, ...]]
]:
    """
    Creates the required mappings to calculate the likelihood of a tree.

    Args:
        root: the root node of the original tree
    Returns:
        two dictionaries, one mapping genotypes to subtrees (here only the
        index and length of the subtrees are needed) and the other one
        mapping indices to subclones
    """
    index_subclone_map, subclone_index_map = create_index_subclone_maps(root)
    genotype_subtree_map = {}
    subtrees = get_subtrees(root)
    original_tree = subtrees[-1]
    all_node_lists_with_len = [(subtree, len(subtree)) for subtree in subtrees]
    size = len(subtrees)
    for index, (subtree, subtree_size) in enumerate(all_node_lists_with_len):
        subtree = create_subtree(subtree, original_tree)
        genotype = create_genotype(size, subtree, subclone_index_map)
        genotype_subtree_map[genotype] = (index, subtree_size)
    return genotype_subtree_map, index_subclone_map
