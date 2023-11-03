import pandas as pd
import pmhn._trees._io as io
from pmhn._trees._backend_code import TreeMHNBackendCode, TreeWrapperCode
import numpy as np
import time
from anytree import RenderTree

df = pd.read_csv("trees.csv")
theta = np.array(
    [
        [-1.41, 0.00, 0.00, -2.2],
        [1.12, -1.41, 0.00, -2.2],
        [0.00, 3.2, -1.8, 2.1],
        [3.00, -1.2, 2.1, -1.8],
    ]
)
sampling_rate = 1.0

# use modified io
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
trees = io.parse_forest(df, naming=naming)


backend = TreeMHNBackendCode()
theta_size = len(theta)
all_mut = set(range(1, theta_size + 1))
tree_logs = []
for idx, tree in trees.items():
    tree_logs.append(TreeWrapperCode(tree))

start_time = time.time()
for idx, tree_log in enumerate(tree_logs):
    print(f"Processing tree {idx+1} of {len(trees)}")
    print(RenderTree(trees[idx + 1]))
    log_value = backend.loglikelihood(tree_log, theta, sampling_rate, all_mut)
    print(f"log_value: {log_value}")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time elapsed: {elapsed_time} seconds")
