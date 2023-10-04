from pmhn._trees._backend import OriginalTreeMHNBackend
from pmhn._trees._tree_utils import create_all_subtrees, bfs_compare
from anytree import Node, RenderTree
import csv
import numpy as np


def csv_to_numpy(file_path):
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        next(reader)
        data_list = list(reader)
    return np.array(data_list, dtype=float)


A = Node(0)
B = Node(1, parent=A)
C = Node(3, parent=A)
D = Node(3, parent=B)

mhn_file_path = "/home/laukeller/BSc Thesis/TreeMHN/Example/MHN_Matrix.csv"
mhn_array = csv_to_numpy(mhn_file_path)
print(mhn_array)
subtrees = create_all_subtrees(A)
node = bfs_compare(subtrees[0], subtrees[1])
print(RenderTree(node))
backend = OriginalTreeMHNBackend()

loglikelihood = backend.loglikelihood(A, mhn_array, 1.0)
print(loglikelihood)
