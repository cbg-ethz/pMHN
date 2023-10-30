from pmhn._trees._backend import OriginalTreeMHNBackend, TreeWrapper
from anytree import Node
import numpy as np


def test_likelihood_small_tree():
    """
    Checks if the likelihood of a small tree is calculated
    correctly.

    tree:
        0
        |
        2
        |
        7
    """

    A = Node(0)
    B = Node(7, parent=A)
    Node(2, parent=B)
    theta = np.array(
        [
            [-1.41, 0.00, 0.00, 4.91, 1.03, 0.00, -1.91, -0.74, -1.35, 1.48],
            [-1.12, -2.26, 0.00, 0.82, 0.00, 0.00, 1.16, 0.00, -1.62, 0.00],
            [0.00, -0.86, -2.55, 1.58, 0.00, 0.00, 1.02, -2.70, 0.00, 0.68],
            [0.00, 0.00, 0.00, -3.69, 0.00, 0.00, -0.95, 1.42, 0.00, -1.01],
            [-3.08, -1.42, -3.14, 0.00, -3.95, 3.90, -1.46, -2.00, 0.00, 2.87],
            [-2.24, 0.00, 0.00, 0.00, 0.00, -2.38, -2.13, 1.50, 0.00, 1.35],
            [0.00, 0.00, 0.00, 0.00, 1.52, 0.00, -1.79, 0.00, 0.00, 0.00],
            [1.69, 0.76, 0.00, 1.29, 1.73, -0.82, -1.38, -4.65, 0.92, 0.00],
            [-1.22, 0.00, 0.00, 0.00, 0.65, -1.14, 0.00, 0.00, -3.25, 0.00],
            [0.97, 1.75, 0.00, -3.66, -1.28, 0.00, 1.66, 0.00, 0.00, -3.03],
        ]
    )
    sampling_rate = 1.0

    backend = OriginalTreeMHNBackend()
    log_value = backend.loglikelihood(TreeWrapper(A), theta, sampling_rate)

    assert np.allclose(log_value, -5.793104, atol=1e-5)


def test_likelihood_medium_tree():
    """ 
   Checks if the likelihood of a medium-sized tree is calculated
   correctly.
   
   tree:
        0
      /   \
     2     3   
     |     | 
     1     10 
     |    
     10
   """

    A = Node(0)
    B = Node(2, parent=A)
    C = Node(3, parent=A)
    D = Node(1, parent=B)
    Node(10, parent=C)
    Node(10, parent=D)
    theta = np.array(
        [
            [-1.41, 0.00, 0.00, 4.91, 1.03, 0.00, -1.91, -0.74, -1.35, 1.48],
            [-1.12, -2.26, 0.00, 0.82, 0.00, 0.00, 1.16, 0.00, -1.62, 0.00],
            [0.00, -0.86, -2.55, 1.58, 0.00, 0.00, 1.02, -2.70, 0.00, 0.68],
            [0.00, 0.00, 0.00, -3.69, 0.00, 0.00, -0.95, 1.42, 0.00, -1.01],
            [-3.08, -1.42, -3.14, 0.00, -3.95, 3.90, -1.46, -2.00, 0.00, 2.87],
            [-2.24, 0.00, 0.00, 0.00, 0.00, -2.38, -2.13, 1.50, 0.00, 1.35],
            [0.00, 0.00, 0.00, 0.00, 1.52, 0.00, -1.79, 0.00, 0.00, 0.00],
            [1.69, 0.76, 0.00, 1.29, 1.73, -0.82, -1.38, -4.65, 0.92, 0.00],
            [-1.22, 0.00, 0.00, 0.00, 0.65, -1.14, 0.00, 0.00, -3.25, 0.00],
            [0.97, 1.75, 0.00, -3.66, -1.28, 0.00, 1.66, 0.00, 0.00, -3.03],
        ]
    )
    sampling_rate = 1.0

    backend = OriginalTreeMHNBackend()
    log_value = backend.loglikelihood(TreeWrapper(A), theta, sampling_rate)

    assert np.allclose(log_value, -14.729560, atol=1e-5)


def test_likelihood_large_tree():
    """ 
   Checks if the likelihood of a large tree is calculated
   correctly.
   
   tree:
        0
      /   \
     3     6   
         /   \
        4     5 
        |   /   \
        1  1     7
           |   /   \
           7  3     10 
   """

    A = Node(0)
    Node(3, parent=A)
    C = Node(6, parent=A)
    D = Node(4, parent=C)
    E = Node(5, parent=C)
    Node(1, parent=D)
    G = Node(1, parent=E)
    H = Node(7, parent=E)
    Node(7, parent=G)
    Node(3, parent=H)
    Node(10, parent=H)
    theta = np.array(
        [
            [-1.41, 0.00, 0.00, 4.91, 1.03, 0.00, -1.91, -0.74, -1.35, 1.48],
            [-1.12, -2.26, 0.00, 0.82, 0.00, 0.00, 1.16, 0.00, -1.62, 0.00],
            [0.00, -0.86, -2.55, 1.58, 0.00, 0.00, 1.02, -2.70, 0.00, 0.68],
            [0.00, 0.00, 0.00, -3.69, 0.00, 0.00, -0.95, 1.42, 0.00, -1.01],
            [-3.08, -1.42, -3.14, 0.00, -3.95, 3.90, -1.46, -2.00, 0.00, 2.87],
            [-2.24, 0.00, 0.00, 0.00, 0.00, -2.38, -2.13, 1.50, 0.00, 1.35],
            [0.00, 0.00, 0.00, 0.00, 1.52, 0.00, -1.79, 0.00, 0.00, 0.00],
            [1.69, 0.76, 0.00, 1.29, 1.73, -0.82, -1.38, -4.65, 0.92, 0.00],
            [-1.22, 0.00, 0.00, 0.00, 0.65, -1.14, 0.00, 0.00, -3.25, 0.00],
            [0.97, 1.75, 0.00, -3.66, -1.28, 0.00, 1.66, 0.00, 0.00, -3.03],
        ]
    )
    sampling_rate = 1.0

    backend = OriginalTreeMHNBackend()
    log_value = backend.loglikelihood(TreeWrapper(A), theta, sampling_rate)

    assert np.allclose(log_value, -22.288420, atol=1e-5)
