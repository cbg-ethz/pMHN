import pandas as pd
import pmhn._trees._io as io
from pmhn._trees._backend_code import TreeMHNBackendCode, TreeWrapperCode
import csv
import numpy as np
import time
from anytree import RenderTree


def csv_to_numpy(file_path):
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        next(reader)
        data_list = list(reader)
    return np.array(data_list, dtype=float)


# AML trees
df_AML = pd.read_csv("likelihood_R/trees_AML_R.csv")

# randomly generated 500 trees using a random theta
df_500 = pd.read_csv("likelihood_R/trees_500_R.csv")

# theta matrices
theta_AML = csv_to_numpy("likelihood_R/MHN_Matrix_AML.csv")
theta_500 = csv_to_numpy("likelihood_R/MHN_Matrix_500.csv")
# loglikelihoods in R
log_vec_R_AML = np.genfromtxt("likelihood_R/log_vec_R_AML.csv", delimiter=",")
log_vec_R_500 = np.genfromtxt("likelihood_R/log_vec_R_500.csv", delimiter=",")

# define sampling rate
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

# parse trees
trees_AML = io.parse_forest(df_AML, naming=naming)
trees_500 = io.parse_forest(df_500, naming=naming)

# calculate loglikelihoods
log_vec_py_AML = np.empty(len(trees_AML))
log_vec_py_500 = np.empty(len(trees_500))

start_time = time.time()
backend = TreeMHNBackendCode()
theta_AML_size = len(theta_AML)
all_mut_AML = set(range(1, theta_AML_size + 1))
for idx, tree in trees_AML.items():
    print(f"Processing tree {idx} of {len(trees_AML)}")
    print(RenderTree(tree))
    tree_log = TreeWrapperCode(tree)
    log_value = backend.loglikelihood(tree_log, theta_AML, sampling_rate, all_mut_AML)
    log_vec_py_AML[idx - 1] = log_value
    print(f"log_value: {log_value}")
theta_500_size = len(theta_500)
all_mut_500 = set(range(1, theta_500_size + 1))
for idx, tree in trees_500.items():
    print(f"Processing tree {idx} of {len(trees_500)}")
    print(RenderTree(tree))
    tree_log = TreeWrapperCode(tree)
    log_value = backend.loglikelihood(tree_log, theta_500, sampling_rate, all_mut_500)
    log_vec_py_500[idx - 1] = log_value
    print(f"log_value: {log_value}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time elapsed: {elapsed_time} seconds")
# write Python loglikelihoods to CSV
np.savetxt("likelihood_py/log_vec_py_AML.csv", log_vec_py_AML, delimiter=",")
np.savetxt("likelihood_py/log_vec_py_500.csv", log_vec_py_500, delimiter=",")


# check if the loglikelihood vectors are the same
if np.allclose(log_vec_py_AML, log_vec_R_AML, atol=1e-10):
    print("The loglikelihoods of the AML trees are the same in R and Python.")

if np.allclose(log_vec_py_500, log_vec_R_500, atol=1e-10):
    print(
        "The loglikelihoods of the 500 randomly generated"
        " trees are the same in R and Python."
    )
