import csv
import numpy as np
from anytree import Node
import _simulate 

def csv_to_numpy(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        data_list = list(reader)
    return np.array(data_list, dtype=float)

def write_trees_to_csv(trees, output_file_path):
    with open(output_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Patient_ID", "Tree_ID", "Node_ID", "Mutation_ID", "Parent_ID"])
        
        patient_id = 0
        for tree_dict in trees:
            patient_id += 1
            tree_id = patient_id
            node_id = 0
            for node, _ in tree_dict.items():
                node_id += 1
                mutation_id = int(node.name)
                parent_id = int(node.parent.name) if node.parent else node_id
                writer.writerow([patient_id, tree_id, node_id, mutation_id, parent_id])

if __name__ == "__main__":
    mhn_file_path = "/home/laukeller/BSc Thesis/TreeMHN/Example/MHN_Matrix.csv"
    mhn_array = csv_to_numpy(mhn_file_path)
    print(mhn_array)
    
    rng = np.random.default_rng()
    theta = mhn_array 
    mean_sampling_time = 1.0  
    
    tree_counts = [500, 2000, 5000, 10000]
    
    for n_points in tree_counts:
        trees_file_path = f"/home/laukeller/BSc Thesis/pMHN/src/pmhn/_trees/trees_{n_points}.csv"
        
        _, trees = _simulate.simulate_trees(rng, n_points, theta, mean_sampling_time)
        write_trees_to_csv(trees, trees_file_path)

