import dataclasses

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import seaborn as sns
import csv
import pandas as pd
import pmhn
from  pmhn._trees._simulate import simulate_trees
from pmhn._ppl._treemhn import TreeMHNLoglikelihood
from pmhn._trees._backend_geno import OriginalTreeMHNBackend, LoglikelihoodSingleTree
from anytree import RenderTree, Node
matplotlib.use("agg")
    
# --- Working directory ---
workdir: "generated/treemhn-bayes"

N_CHAINS: int = 4


@dataclasses.dataclass
class Settings:
    n_mutations: int
    n_patients: int
    p_offdiag: float
    mean_sampling_time: float = 450.0 
    data_seed: int = 111
    prior_sampling_seed: int = 222
    tuning_samples: int = 10000
    mcmc_samples: int = 10000

    smc_particles: int = 1000


SCENARIOS = {
    #"small_treemhn_spike_and_slab_0.05_mcmc_normal": Settings(n_mutations=10, n_patients=200, p_offdiag=3/8**2),
    "large_treemhn_random_theta_mcmc_horse_shoe_450.0": Settings(n_mutations=10, n_patients=2000, p_offdiag=3/8**2),
}

rule all:
    input:
        theta_matrices = expand("{scenario}/theta.pdf", scenario=SCENARIOS.keys()),
        trees = expand("{scenario}/trees.pdf", scenario=SCENARIOS.keys()),
        mcmc_samples = expand("{scenario}/mcmc-samples.nc", scenario=SCENARIOS.keys()),
        theta_samples = expand("{scenario}/prior/theta_samples.pdf", scenario=SCENARIOS.keys()),
        posterior_theta_plots=expand("{scenario}/posterior_theta_from_mcmc.pdf", scenario=SCENARIOS.keys())
rule plot_theta_from_data:
    input:
        arrays="{scenario}/arrays.npz"
    output:
        theta = "{scenario}/theta.pdf"
    run:
        theta = np.load(input.arrays)["theta"]
        print(theta)
        fig, ax = plt.subplots()
        pmhn.plot_theta(theta, ax=ax)
        fig.tight_layout()
        fig.savefig(output.theta)

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
            node_id_dict= {}
            for node, _ in tree_dict.items():
                node_id += 1
                node_id_dict[node]=node_id
                mutation_id = node.name
                parent_id = node_id_dict[node.parent] if node.parent else node_id
                writer.writerow([patient_id, tree_id, node_id, mutation_id, parent_id])

rule plot_trees_from_data:
    input:
        arrays="{scenario}/arrays.npz"
    output:
        trees = "{scenario}/trees.pdf",
        trees_csv = "{scenario}/trees.csv"
    run:

        trees = np.load(input.arrays, allow_pickle=True)["trees_dict"]
        write_trees_to_csv(trees, output.trees_csv)
        trees_from_csv = pd.read_csv(output.trees_csv)

        tree_sizes = trees_from_csv.groupby("Tree_ID").size()
        plt.hist(tree_sizes, alpha=0.5, edgecolor="k", label="Trees")
        plt.xlabel("Tree Size")
        plt.ylabel("Frequency")
        plt.legend(loc="upper right")
        plt.title("Tree Size Distribution")
        plt.savefig(output.trees)



rule generate_data:
    output: 
      arrays="{scenario}/arrays.npz"
    run:
        settings = SCENARIOS[wildcards.scenario]
        
        rng = np.random.default_rng(settings.data_seed)

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
        theta = theta*5
        print(theta)
        sampling_times, trees_dict = simulate_trees(
            rng=rng,
            n_points=settings.n_patients,
            theta=theta,
            mean_sampling_time=settings.mean_sampling_time, min_tree_size = None,max_tree_size = None
        )
        trees = []
        print(len(trees_dict))
        for i, tree in enumerate(trees_dict):
            print(f"reached tree {i}")
            for key, val in tree.items():
                tree_log =LoglikelihoodSingleTree(key) 
                break 
        np.savez(output.arrays, trees_dict = trees_dict, trees=trees, theta=theta, sampling_times=sampling_times)

def prepare_full_model(trees, mean_sampling_time, n_mutations, all_mut) -> pm.Model:
    loglikelihood = TreeMHNLoglikelihood(data=trees, mean_sampling_time = mean_sampling_time, all_mut = all_mut, backend=OriginalTreeMHNBackend())
    model = pmhn.prior_regularized_horseshoe(n_mutations=n_mutations)

    with model:
        pm.Potential("loglikelihood", loglikelihood(model.theta))

    return model


rule sample_prior:
    output:
        prior_samples="{scenario}/prior/samples.nc"
    run:
        settings = SCENARIOS[wildcards.scenario]
        rng = np.random.default_rng(settings.prior_sampling_seed)
        n_samples: int = 300

        model = pmhn.prior_regularized_horseshoe(n_mutations=settings.n_mutations)
        with model:
            idata = pm.sample_prior_predictive(samples=n_samples, random_seed=rng)
        idata.to_netcdf(output.prior_samples)
        
rule plot_prior_predictive_theta:
    input: "{scenario}/prior/samples.nc"
    output: "{scenario}/prior/theta_samples.pdf"
    run:
        idata = az.from_netcdf(str(input))
        samples = idata.prior["theta"][0].values

        fig, _ = pmhn.plot_theta_samples(samples, width=6, height=4)
        fig.savefig(str(output))
rule smc_sample_one_chain:
    input:
        arrays="{scenario}/arrays.npz"
    output:
        smc_samples="{scenario}/smc-chains/{chain}.nc"
    run:
        

        chain = int(wildcards.chain)
        settings = SCENARIOS[wildcards.scenario]
        with np.load(input.arrays, allow_pickle=True) as data:
            tree_dict = data["trees"]
            theta = data["theta"]
        
        n_mutations = len(theta) 
        all_mut = set(i + 1 for i in range(n_mutations))
        model = prepare_full_model(trees, settings.mean_sampling_time, n_mutations, all_mut)
        with model:
            idata = pm.sample_smc(
                draws=settings.smc_particles,  
                chains=1,
                random_seed=chain,
            )        
        idata.to_netcdf(output.smc_samples)


rule smc_assemble_chains:
    input:
        chains=expand("{scenario}/smc-chains/{chain}.nc", chain=range(1, N_CHAINS + 1), allow_missing=True)
    output:
        all_samples="{scenario}/smc-samples.nc"
    run:
        assemble_chains(input.chains, output.all_samples)


def assemble_chains(chain_files, output_file) -> None:
    chains = [az.from_netcdf(chain_file) for chain_file in chain_files]
    all_samples = az.concat(chains, dim="chain")
    all_samples.to_netcdf(output_file)
rule mcmc_sample_one_chain:
    input:
        arrays="{scenario}/arrays.npz"
    output:
        chain="{scenario}/mcmc-chains/{chain}.nc"
    run:
        chain = int(wildcards.chain)
        settings = SCENARIOS[wildcards.scenario]
        print("reached !!!")
        with np.load(input.arrays, allow_pickle=True) as data:
            tree_dict = data["trees"]
            theta = data["theta"]
        trees = []
        print(len(tree_dict))
        for i, tree in enumerate(tree_dict):
            print(f"reached tree {i}")
            for key, val in tree.items():
                print(RenderTree(key))
                tree_log =LoglikelihoodSingleTree(key) 
                break
        n_mutations = len(theta) 
        all_mut = set(i + 1 for i in range(n_mutations))
        print("reached end")
        model = prepare_full_model(trees, settings.mean_sampling_time, n_mutations, all_mut)
        print("reached !!!!!!") 
        with model:
            idata = pm.sample(chains=4, random_seed=chain, tune=settings.tuning_samples, draws=settings.mcmc_samples
            )        
        idata.to_netcdf(output.chain) 

rule mcmc_assemble_chains:
    input:
        chains=expand("{scenario}/mcmc-chains/{chain}.nc", chain=range(1, N_CHAINS + 1), allow_missing=True)
    output:
        all_samples="{scenario}/mcmc-samples.nc"
    run:
        assemble_chains(input.chains, output.all_samples)
# for comparison with ground truth
rule plot_posterior_thetas_from_smc:
    input:
        smc_samples=expand("{scenario}/smc-samples.nc", scenario=SCENARIOS.keys())
    output:
        posterior_theta_plots=expand("{scenario}/posterior_theta_from_smc.pdf", scenario=SCENARIOS.keys())
    run:
        idata = az.from_netcdf(str(input))
        posterior_samples = idata.posterior["theta"][0].values

        fig, _ = pmhn.plot_theta_samples(posterior_samples, width=6, height=4)
        fig.savefig(str(output))


rule plot_posterior_thetas_from_mcmc:
    input:
        mcmc_samples=expand("{scenario}/mcmc-samples.nc", scenario=SCENARIOS.keys())
    output:
        posterior_theta_plots=expand("{scenario}/posterior_theta_from_mcmc.pdf", scenario=SCENARIOS.keys())
    run:
        idata = az.from_netcdf(str(input))
        posterior_samples = idata.posterior["theta"][0].values

        fig, _ = pmhn.plot_theta_samples(posterior_samples, width=6, height=4)
        fig.savefig(str(output))