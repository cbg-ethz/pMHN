from pmhn._trees import Tree


def test_create_tree_mutation_occurs_twice() -> None:
    """
    We want to see if a tree with a mutation occuring twice can be created:
        root
        ├── 0
        │   └── 1
        └── 1
    """
    root = Tree("root")
    zero = Tree(0, parent=root)
    Tree(1, parent=zero)
    Tree(1, parent=root)

    assert [node.name for node in root.leaves] == [1, 1]
