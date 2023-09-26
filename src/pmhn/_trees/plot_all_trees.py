import pandas as pd
import matplotlib.pyplot as plt

paths = {
    500: ('/home/laukeller/BSc Thesis/TreeMHN/Example/trees_500.csv', '/home/laukeller/BSc Thesis/pMHN/src/pmhn/_trees/trees_500.csv'),
    2000: ('/home/laukeller/BSc Thesis/TreeMHN/Example/trees_2000.csv', '/home/laukeller/BSc Thesis/pMHN/src/pmhn/_trees/trees_2000.csv'),
    5000: ('/home/laukeller/BSc Thesis/TreeMHN/Example/trees_5000.csv', '/home/laukeller/BSc Thesis/pMHN/src/pmhn/_trees/trees_5000.csv'),
    10000: ('/home/laukeller/BSc Thesis/TreeMHN/Example/trees_10000.csv', '/home/laukeller/BSc Thesis/pMHN/src/pmhn/_trees/trees_10000.csv'),
}

fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 grid of subplots
axs = axs.ravel()

bin_edges = [x - 0.5 for x in range(2, 12)] + [11.5]  # [1.5, 2.5, ..., 10.5, 11.5]

for i, (num_trees, (r_path, python_path)) in enumerate(paths.items()):
    r_trees = pd.read_csv(r_path)
    python_trees = pd.read_csv(python_path)
    
    r_tree_sizes = r_trees.groupby('Tree_ID').size()
    python_tree_sizes = python_trees.groupby('Tree_ID').size()
    
    axs[i].hist(r_tree_sizes, bins=bin_edges, alpha=0.5, label='R Trees')
    axs[i].hist(python_tree_sizes, bins=bin_edges, alpha=0.5, label='Python Trees')
    axs[i].set_xlabel('Tree Size')
    axs[i].set_ylabel('Frequency')
    axs[i].legend(loc='upper right')
    axs[i].set_title(f'Tree Size Distribution ({num_trees} trees)')

plt.tight_layout()  
plt.show()

