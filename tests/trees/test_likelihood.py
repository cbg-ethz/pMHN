from pmhn._trees._backend import OriginalTreeMHNBackend
from pmhn._trees._tree_utils import create_all_subtrees
from anytree import Node
import numpy as np


def test_create_V_Mat():
    """
    Checks if create_V_Mat is implemented correctly.
    """

    true_Q = np.array(
        [
            [
                -(np.exp(-1.41) + np.exp(-2.26) + np.exp(-2.55)),
                np.exp(-1.41),
                np.exp(-2.55),
                0,
                0,
                0,
            ],
            [
                0,
                -(
                    np.exp(-2.26)
                    + np.exp(-2.55)
                    + np.exp(-1.12 - 2.26)
                    + np.exp(1 - 2.55)
                ),
                0,
                np.exp(1 - 2.55),
                np.exp(-2.55),
                0,
            ],
            [
                0,
                0,
                -(
                    np.exp(-1.41)
                    + np.exp(-2.26)
                    + np.exp(-1.41 + 3)
                    + np.exp(-2.26 + 2)
                ),
                0,
                np.exp(-1.41),
                0,
            ],
            [
                0,
                0,
                0,
                -(
                    np.exp(-2.26)
                    + np.exp(-2.55)
                    + np.exp(-2.26 - 1.12)
                    + np.exp(-2.26 - 1.12 + 2)
                ),
                0,
                np.exp(-2.55),
            ],
            [
                0,
                0,
                0,
                0,
                -(
                    np.exp(-2.26)
                    + np.exp(-2.26 - 1.12)
                    + np.exp(-2.55 + 1)
                    + np.exp(-1.41 + 3)
                    + np.exp(-2.26 + 2)
                ),
                np.exp(-2.55 + 1),
            ],
            [
                0,
                0,
                0,
                0,
                0,
                -(
                    np.exp(-2.26)
                    + np.exp(-2.26 - 1.12)
                    + np.exp(-2.26 + 2 - 1.12)
                    + np.exp(-1.41 + 3)
                    + np.exp(-2.26 + 2)
                ),
            ],
        ]
    )
    A = Node(0)
    B = Node(1, parent=A)
    Node(3, parent=A)
    Node(3, parent=B)
    subtrees = create_all_subtrees(A)
    subtrees_size = len(subtrees)
    sampling_rate = 1.0
    true_V = np.eye(subtrees_size) * sampling_rate - true_Q
    backend = OriginalTreeMHNBackend()
    theta = np.array([[-1.41, 2, 3], [-1.12, -2.26, 2], [1, -0.86, -2.55]])

    V = backend.create_V_Mat(A, theta, sampling_rate)

    assert np.allclose(V, true_V, atol=1e-8)


def test_diag_entry():
    r"""
    
    Checks if the diagonal values of the V matrix are calculated correctly.

        0
     /  |  \
    2   1   3   
    |   |
    3   3


    augmented tree:


        0
     /  |  \
    2   1   3
    |\  |\  |\
    3 1 3 2 1 2
    |   |  
    1   2
        
    """
    A = Node(0)
    B = Node(2, parent=A)
    D = Node(1, parent=A)
    Node(3, parent=A)
    Node(3, parent=B)
    Node(3, parent=D)
    theta = np.array([[-1.41, 2, 3], [-1.12, -2.26, 2], [1, -0.86, -2.55]])
    true_diag_entry = -(
        np.exp(-1.41 + 2)
        + np.exp(-2.26 - 1.12)
        + np.exp(-1.41 + 3)
        + np.exp(-2.26 + 2)
        + np.exp(-1.41 + 2 + 3)
        + np.exp(-2.26 + 2 - 1.12)
    )
    backend = OriginalTreeMHNBackend()
    diag_entry = backend.diag_entry(A, theta)

    assert np.allclose(diag_entry, true_diag_entry, atol=1e-8)


def test_off_diag_entry_valid():
    r"""
    Checks if the off-diagonal entries of the V matrix
    are calculated correctly.
    
    first tree:
        0
     /  |  \
    2   1   3   
    |   
    3   

    second tree:
        0
     /  |  \
    2   1   3   
    |   |
    3   3 
       
        
    """

    theta = np.array([[-1.41, 2, 3], [-1.12, -2.26, 2], [1, -0.86, -2.55]])
    # first tree
    A_1 = Node(0)
    B_1 = Node(2, parent=A_1)
    Node(1, parent=A_1)
    Node(3, parent=A_1)
    Node(3, parent=B_1)

    # second tree
    A_2 = Node(0)
    B_2 = Node(2, parent=A_2)
    D_2 = Node(1, parent=A_2)
    Node(3, parent=A_2)
    Node(3, parent=B_2)
    Node(3, parent=D_2)

    true_off_diag_entry = np.exp(-2.55 + 1)
    backend = OriginalTreeMHNBackend()
    off_diag_entry = backend.off_diag_entry(A_1, A_2, theta)

    assert np.allclose(off_diag_entry, true_off_diag_entry, atol=1e-8)


def test_off_diag_entry_invalid_size():
    r"""

    Checks if off_diag_entry successfully returns 0 when the size is invalid
    (i.e the first tree is not smaller than the second tree by one).

    first tree:
        0
     /  |  \
    2   1   3   
    |   
    3   

    second tree:
        0
     /  |  \
    2   1   3   
    |   |   | 
    3   3   2 
       
        
    """

    # first tree
    A_1 = Node(0)
    B_1 = Node(2, parent=A_1)
    Node(1, parent=A_1)
    Node(3, parent=A_1)
    Node(3, parent=B_1)

    # second tree
    A_2 = Node(0)
    B_2 = Node(2, parent=A_2)
    C_2 = Node(1, parent=A_2)
    D_2 = Node(3, parent=A_2)
    Node(3, parent=B_2)
    Node(3, parent=C_2)
    Node(2, parent=D_2)

    theta = np.array([[-1.41, 2, 3], [-1.12, -2.26, 2], [1, -0.86, -2.55]])
    backend = OriginalTreeMHNBackend()

    assert backend.off_diag_entry(A_1, A_2, theta) == 0


def test_off_diag_entry_not_subset():
    r"""
    Checks if the off_diag_entry succesfully returns 0 when the
    first tree is not a subtree of the second tree.

    first tree:
        0
     /  |  \
    2   1   3   
    |   
    3   

    second tree:
        0
     /  |  \
    2   1   3   
        |   | 
        3   2 
       
        
    """
    # first tree
    A_1 = Node(0)
    B_1 = Node(2, parent=A_1)
    Node(1, parent=A_1)
    Node(3, parent=A_1)
    Node(3, parent=B_1)

    # second tree
    A_2 = Node(0)
    Node(2, parent=A_2)
    C_2 = Node(1, parent=A_2)
    D_2 = Node(3, parent=A_2)
    Node(3, parent=C_2)
    Node(2, parent=D_2)

    theta = np.array([[-1.41, 2, 3], [-1.12, -2.26, 2], [1, -0.86, -2.55]])
    backend = OriginalTreeMHNBackend()

    assert backend.off_diag_entry(A_1, A_2, theta) == 0
