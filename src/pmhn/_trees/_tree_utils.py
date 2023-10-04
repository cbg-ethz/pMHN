from anytree import Node, RenderTree, LevelOrderGroupIter
from itertools import combinations, product
from typing import Optional


def all_combinations_of_elements(*lists):
    """
    Takes a variable number of lists as input and returns a list
    containing all possible combination of the input lists. In our
    use case: It takes a list of lists of subtrees (for each child
    one list of subtrees) as input where a subtree itself is a list
    of nodes and outputs all combinations of subtrees.

    Args:
        *lists: any number of lists

    Returns:
           all combinations of the input lists
    """
    n = len(lists)
    all_combinations = []

    for r in range(1, n + 1):
        for list_combination in combinations(lists, r):
            for element_combination in product(*list_combination):
                all_combinations.append(list(element_combination))

    return all_combinations


def create_subtree(original_root: Node, nodes_list: list[Node]) -> Optional[Node]:
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

    return nodes_dict.get(original_root)


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

    result_subtrees = []
    result_subtrees.append([node])
    for combination in combined_subtrees:
        subtree_with_root = [node] + [
            item for sublist in combination for item in sublist
        ]
        result_subtrees.append(subtree_with_root)

    return result_subtrees


def create_all_subtrees(root: Node) -> list[Node]:
    """
    Creates a list of subtrees and sorts the list in ascending subtree size.

    Args:
        root: the root node
    Returns:
            the final list of subtrees
    """
    all_node_lists = get_subtrees(root)
    all_node_lists = sorted(all_node_lists, key=len)
    all_subtrees = []
    for subtree in all_node_lists:
        all_subtrees.append(create_subtree(root, subtree))
    return all_subtrees


def get_lineage(node: Node) -> list[int]:
    """
    Creates a list of the names of the nodes that
    are in the lineage of input node.
    Args:
        node: a node
    Returns:
            the lineage of a node
    """
    return [ancestor.name for ancestor in node.path]  # type: ignore


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
        set_nodes1_lineages = {tuple(get_lineage(node)) for node in nodes1}
        set_nodes2_lineages = {tuple(get_lineage(node)) for node in nodes2}
        additional_nodes_lineages = set_nodes2_lineages ^ set_nodes1_lineages
        if len(additional_nodes_lineages) != 0:
            return False

    return True


def bfs_compare(tree1: Node, tree2: Node) -> Optional[Node]:
    """
    Checks if tree1 is a subtree of tree2 and is smaller in
    size by one.

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
    if len(list(tree2.descendants)) - len(list(tree1.descendants)) != 1:
        return None
    for nodes1, nodes2 in zip(iter1, iter2):
        set_nodes1_lineages = {tuple(get_lineage(node)) for node in nodes1}
        set_nodes2_lineages = {tuple(get_lineage(node)) for node in nodes2}
        additional_nodes_lineages = set_nodes2_lineages ^ set_nodes1_lineages
        diff_count += len(additional_nodes_lineages)

        if diff_count == 1 and exit_node is None:
            additional_node_lineage = additional_nodes_lineages.pop()
            for node in nodes1:
                if tuple(get_lineage(node)) == additional_node_lineage:
                    return None

            for node in nodes2:
                if tuple(get_lineage(node)) == additional_node_lineage:
                    exit_node = node
        if diff_count > 1:
            return None
    if len(iter1) < len(iter2):
        return iter2[-1][0]
    return exit_node


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
