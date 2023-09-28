import pandas as pd
import matplotlib.pyplot as plt

r_trees_path = "/home/laukeller/BSc Thesis/TreeMHN/Example/trees_10000.csv"
python_trees_path = "/home/laukeller/BSc Thesis/pMHN/src/pmhn/_trees/trees_10000.csv"

r_trees = pd.read_csv(r_trees_path)
python_trees = pd.read_csv(python_trees_path)

r_tree_sizes = r_trees.groupby("Tree_ID").size()
python_tree_sizes = python_trees.groupby("Tree_ID").size()

bin_edges = [x - 0.5 for x in range(2, 12)] + [11.5]
plt.hist(r_tree_sizes, bins=bin_edges, alpha=0.5, edgecolor="k", label="R Trees")
plt.hist(
    python_tree_sizes, bins=bin_edges, alpha=0.5, edgecolor="k", label="Python Trees"
)
plt.xlabel("Tree Size")
plt.ylabel("Frequency")
plt.legend(loc="upper right")
plt.title("Tree Size Distribution")
plt.show()
