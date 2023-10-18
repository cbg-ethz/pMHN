import anytree
import pandas as pd

import pmhn._trees._io as io


def test_parse_tree() -> None:
    """Parse a simple tree with root 1 and two branches:
     - 1 -> 2
     - 1 -> 5 -> 10
    Each node has two values: n**2 and 2*n, where n is its node id.
    """
    tree_df = pd.DataFrame(
        {
            "Node_ID": [1, 2, 3, 4],
            "Parent_ID": [1, 1, 1, 2],
            "Mutation_ID": [0, 5, 2, 10],
            "ValueSquare": [0, 25, 4, 100],
            "ValueDoubled": [0, 10, 4, 20],
        }
    )
    naming = io.TreeNaming(
        node="Node_ID",
        parent="Parent_ID",
        mutation="Mutation_ID",
        data={
            "ValueSquare": "square",
            "ValueDoubled": "doubled",
        },
    )
    root = io.parse_tree(df=tree_df, naming=naming)

    for node in anytree.PreOrderIter(root):
        n = node.name
        assert node.square == n**2
        assert node.doubled == 2 * n

        if n == 2 or n == 5:
            assert node.parent.name == 0
        elif n == 10:
            assert node.parent.name == 5


def test_parse_forest() -> None:
    forest = pd.DataFrame(
        {
            "Patient_ID": [1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
            "Tree_ID": [1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
            "Node_ID": [1, 2, 1, 2, 3, 1, 2, 3, 4, 5],
            "Mutation_ID": [0, 4, 0, 4, 3, 0, 2, 4, 3, 4],
            "Parent_ID": [1, 1, 1, 1, 2, 1, 1, 1, 2, 2],
        }
    )
    naming = io.ForestNaming(
        tree_name="Tree_ID",
        naming=io.TreeNaming(
            node="Node_ID",
            parent="Parent_ID",
            data={
                "Mutation_ID": "mutation",
            },
        ),
    )

    parsed = io.parse_forest(forest, naming=naming)
    assert len(parsed) == 3
    for root in parsed.values():
        assert isinstance(root, anytree.Node)
        assert hasattr(root, "mutation")
