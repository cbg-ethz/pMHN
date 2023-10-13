from anytree import Node, RenderTree, PreOrderIter

A = Node("A")
B = Node("B", parent=A)
C = Node("C", parent=A)
D = Node("D", parent=B)
E = Node("E", parent=C)
F = Node("F", parent=C)

all_nodes = list(PreOrderIter(A))
print(RenderTree(A))
children = {}
for node in all_nodes:
    children[node.name] = [child.name for child in node.children]
print(children)
