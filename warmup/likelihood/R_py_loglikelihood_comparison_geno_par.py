import pandas as pd
import pmhn._trees._io as io
from pmhn._trees._backend_geno import OriginalTreeMHNBackend, LoglikelihoodSingleTree
import csv
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor


def csv_to_numpy(file_path):
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        data_list = list(reader)
    return np.array(data_list, dtype=float)


def calculate_loglikelihood(tree_data):
    idx, tree, theta, sampling_rate, all_mut = tree_data
    backend = OriginalTreeMHNBackend()
    tree_log = LoglikelihoodSingleTree(tree)
    log_value = backend.loglikelihood(tree_log, theta, sampling_rate, all_mut)
    return idx, log_value


def parallel_loglikelihood(trees, theta, sampling_rate, all_mut):
    args = [(idx, tree, theta, sampling_rate, all_mut) for idx, tree in trees.items()]
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(calculate_loglikelihood, args))
    results.sort()  # sort results by tree index
    log_values = np.array([log_value for idx, log_value in results])
    return log_values


def main():
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
    start_time = time.time()

    all_mut_AML = set(range(1, len(theta_AML) + 1))
    log_vec_py_AML = parallel_loglikelihood(
        trees_AML, theta_AML, sampling_rate, all_mut_AML
    )

    all_mut_500 = set(range(1, len(theta_500) + 1))
    log_vec_py_500 = parallel_loglikelihood(
        trees_500, theta_500, sampling_rate, all_mut_500
    )

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
            "The loglikelihoods of the 500 randomly"
            " generated trees are the same in R and Python."
        )


if __name__ == "__main__":
    main()
