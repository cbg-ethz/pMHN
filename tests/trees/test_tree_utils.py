from anytree import Node
from pmhn._trees._tree_utils import bfs_compare, check_equality, create_all_subtrees


def test_create_all_subtrees_simple():
    """
    Checks for a simple tree if all subtrees are found using create_all_subtrees.
    """
    A = Node(name=0)
    Node(name=1, parent=A)

    subtrees = create_all_subtrees(A)

    A_test_1 = Node(name=0)
    Node(name=1, parent=A_test_1)

    A_test_2 = Node(name=0)

    true_subtrees = [A_test_1, A_test_2]

    assert len(true_subtrees) == len(subtrees)

    for tree in true_subtrees:
        assert (
            len(
                [
                    subtree
                    for subtree in subtrees
                    if check_equality(subtree, tree) is True
                ]
            )
            == 1
        )


def test_create_all_subtrees_medium():
    """
    Checks for a tree if all subtrees are found using create_all_subtrees.
    """
    A = Node(0)
    B = Node(1, parent=A)
    Node(3, parent=A)
    Node(3, parent=B)
    subtrees = create_all_subtrees(A)

    A_test_1 = Node(0)
    Node(1, parent=A_test_1)

    A_test_2 = Node(0)

    A_test_3 = Node(0)
    Node(3, parent=A_test_3)

    A_test_4 = Node(0)
    B_test_4 = Node(1, parent=A_test_4)
    Node(3, parent=B_test_4)

    A_test_5 = Node(0)
    Node(1, parent=A_test_5)
    Node(3, parent=A_test_5)

    A_test_6 = Node(0)
    B_test_6 = Node(1, parent=A_test_6)
    Node(3, parent=A_test_6)
    Node(3, parent=B_test_6)

    true_subtrees = [A_test_1, A_test_2, A_test_3, A_test_4, A_test_5, A_test_6]
    assert len(true_subtrees) == len(subtrees)

    for tree in true_subtrees:
        assert (
            len(
                [
                    subtree
                    for subtree in subtrees
                    if check_equality(subtree, tree) is True
                ]
            )
            == 1
        )


def test_bfs_compare_long():
    r"""
    Tests if bfs_compare successfully returns the additional node
    if a tree is a subtree of another tree and is smaller in size by one.

    first tree:
        0
        |
        1
        |
        2
        |
        3
        |
        4

    second tree:

        0
        |
        1
        |
        2
        |
        3
        |
        4
        |
        5
    """
    # first tree
    A = Node(0)
    B = Node(1, parent=A)
    C = Node(2, parent=B)
    D = Node(3, parent=C)
    Node(4, parent=D)

    # second tree
    A_ = Node(0)
    B_ = Node(1, parent=A_)
    C_ = Node(2, parent=B_)
    D_ = Node(3, parent=C_)
    E_ = Node(4, parent=D_)
    F_ = Node(5, parent=E_)

    assert bfs_compare(A, A_) == F_


def test_bfs_compare_complex():
    r"""

    Tests if bfs_compare successfully returns the additional node
    if a tree is a subtree of another tree and is smaller in size by one.

    
    first tree:
        0
     /  |  \
    2   1   3   
    |   |   |
    3   2   1
        |   |
        3   2
        
    second tree:
        0
     /  |  \
    2   1   3   
    |   |\  | 
    3   2 3 1 
        |   |
        3   2
    """

    A = Node(0)
    B = Node(2, parent=A)
    C = Node(1, parent=A)
    D = Node(3, parent=A)
    Node(3, parent=B)
    F = Node(2, parent=C)
    G = Node(1, parent=D)
    Node(3, parent=F)
    Node(2, parent=G)

    A_ = Node(0)
    B_ = Node(2, parent=A_)
    C_ = Node(1, parent=A_)
    D_ = Node(3, parent=A_)
    Node(3, parent=B_)
    F_ = Node(2, parent=C_)
    G_ = Node(1, parent=D_)
    Node(3, parent=F_)
    Node(2, parent=G_)
    J_ = Node(3, parent=C_)

    assert bfs_compare(A, A_) == J_
