import csv
import numpy as np
import pmhn._trees._simulate as _simulate
from anytree import RenderTree


def csv_to_numpy(file_path):
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        next(reader)
        data_list = list(reader)
    return np.array(data_list, dtype=float)


def write_trees_to_csv(trees, output_file_path):
    with open(output_file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Patient_ID", "Tree_ID", "Node_ID", "Mutation_ID", "Parent_ID"]
        )

        patient_id = 0
        for tree_dict in trees:
            patient_id += 1
            tree_id = patient_id
            node_id = 0
            node_id_dict = {}
            for node, _ in tree_dict.items():
                node_id += 1
                node_id_dict[node] = node_id
                mutation_id = node.name
                parent_id = node_id_dict[node.parent] if node.parent else node_id
                writer.writerow([patient_id, tree_id, node_id, mutation_id, parent_id])


if __name__ == "__main__":
    mhn_array = csv_to_numpy("MHN_Matrix.csv")
    print(mhn_array)

    rng = np.random.default_rng(111)
    theta = mhn_array
    mean_sampling_time = 0.8

    tree_counts = [200]

    min_tree_size = 2
    max_tree_size = 11
    for n_points in tree_counts:
        trees_file_path = f"trees_py_data/trees_{n_points}.csv"

        _, trees = _simulate.simulate_trees(
            rng,
            n_points,
            theta,
            mean_sampling_time,
            min_tree_size=None,
            max_tree_size=None,
        )
        for i, tree in enumerate(trees):
            print(f"tree {i}")
            for key, val in tree.items():
                print(RenderTree(key))
                break
        write_trees_to_csv(trees, trees_file_path)
