from anytree import Node, RenderTree
from itertools import combinations, product

"""
takes a variable number of lists as input and returns a list containing all possible combination of the input lists
in this case: takes a list of lists of subtrees (for each child one list of subtrees) as input, a subtree itself is a list of nodes
outputs all combinations of subtrees 
"""


def all_combinations_of_elements(*lists):
    n = len(lists)
    all_combinations = []

    for r in range(1, n + 1):
        for list_combination in combinations(lists, r):
            for element_combination in product(*list_combination):
                all_combinations.append(list(element_combination))

    return all_combinations


"""creates a subtree given a subtree (nodes_list) and the root node"""


def create_subtree(original_root, nodes_list):
    nodes_dict = {}

    for node in [original_root] + list(original_root.descendants):
        if node in nodes_list:
            parent_node = next((n for n in nodes_list if n is node.parent), None)
            nodes_dict[node] = Node(node.name, parent=nodes_dict.get(parent_node))

    return nodes_dict.get(original_root)


"""returns a list of all subtrees of a tree, input is the root node
a recursive approach is used: if one knows the subtrees of the children of the root node,
then one can find all combinations of the subtrees of the children and add the root node to each one of these combinations,
this way one obtains all subtrees of the root node"""


def subtrees(node):
    if not node.children:
        return [[node]]

    child_subtrees = [subtrees(child) for child in node.children]

    combined_subtrees = all_combinations_of_elements(*child_subtrees)

    result_subtrees = []
    result_subtrees.append([node])
    for combination in combined_subtrees:
        subtree_with_root = [node] + [
            item for sublist in combination for item in sublist
        ]
        result_subtrees.append(subtree_with_root)

    return result_subtrees


A = Node("0")
B = Node("1", parent=A)
C = Node("3", parent=A)
D = Node("3", parent=B)

print(RenderTree(A))
print("\n")
all_node_lists = subtrees(A)
all_node_lists = sorted(all_node_lists, key=len)
print(all_node_lists)
print("\n")
all_subtrees = []

for nodes_list in all_node_lists:
    subtree = create_subtree(A, nodes_list)
    all_subtrees.append(subtree)
i = 1
for subtree in all_subtrees:
    print(f"{i}. ")
    print(RenderTree(subtree))
    i += 1
