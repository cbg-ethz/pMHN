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
from pmhn._trees._backend_code import TreeMHNBackendCode, TreeWrapperCode 
from anytree import RenderTree, Node
matplotlib.use("agg")
    
# --- Working directory ---
workdir: "generated/"

N_CHAINS: int = 1


@dataclasses.dataclass
class Settings:
    n_mutations: int
    n_patients: int
    p_offdiag: float
    mean_sampling_time: float = 1.0
    data_seed: int = 111
    prior_sampling_seed: int = 222
    tuning_samples: int = 100
    mcmc_samples: int = 100

    smc_particles: int = 24


SCENARIOS = {
    #"small_treemhn_spike_and_slab_0.05_mcmc_normal": Settings(n_mutations=10, n_patients=200, p_offdiag=3/8**2),
    "1000_patients_100_samples_4_mutations_1.0_jitter=0": Settings(n_mutations=4, n_patients=1000, p_offdiag=3/8**2),
}

rule all:
    input:
        theta_matrices = expand("{scenario}/theta.pdf", scenario=SCENARIOS.keys()),
        trees = expand("{scenario}/trees.pdf", scenario=SCENARIOS.keys()),
        mcmc_samples = expand("{scenario}/mcmc-samples.nc", scenario=SCENARIOS.keys()),
        theta_samples = expand("{scenario}/prior/theta_samples.pdf", scenario=SCENARIOS.keys()),
        posterior_theta_plots=expand("{scenario}/posterior_theta_from_mcmc.pdf", scenario=SCENARIOS.keys()),
        trace_plot= expand("{scenario}/posterior/trace_plot_theta.pdf", scenario=SCENARIOS.keys()),
        posterior_plot= expand("{scenario}/posterior/posterior_plot_theta.pdf", scenario=SCENARIOS.keys()),
        summary= expand("{scenario}/posterior/mcmc_summary_theta.txt", scenario=SCENARIOS.keys())
        
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
            [-1.41, 0.00, 0.00, -2.2],
            [1.12, -1.41, 0.00, -2.2],
            [0.00, 3.2, -1.8, 2.1],
           [3.00, -1.2, 2.1, -1.8]
        ]
    )   
        theta = theta*1.7
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
            for key, val in tree.items():
                print(RenderTree(key))
                tree_wrapper =TreeWrapperCode(key) 
                trees.append(tree_wrapper)
                break 

 
        np.savez(output.arrays, trees_dict = trees_dict, trees=trees, theta=theta, sampling_times=sampling_times)

def prepare_full_model(trees, mean_sampling_time, n_mutations, all_mut) -> pm.Model:
    loglikelihood = TreeMHNLoglikelihood(data=trees, mean_sampling_time = mean_sampling_time, all_mut = all_mut, backend=TreeMHNBackendCode())
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
        with np.load(input.arrays, allow_pickle=True) as data:
            trees = data["trees"]
            theta = data["theta"]
        n_mutations = len(theta) 
        all_mut = set(i + 1 for i in range(n_mutations))
        model = prepare_full_model(trees, settings.mean_sampling_time, n_mutations, all_mut)
        with model:
            idata = pm.sample(chains=1, random_seed=chain, tune=settings.tuning_samples, draws=settings.mcmc_samples
            )        
        idata.to_netcdf(output.chain) 

rule mcmc_assemble_chains:
    input:
        chains=expand("{scenario}/mcmc-chains/{chain}.nc", chain=range(1, N_CHAINS + 1), allow_missing=True)
    output:
        all_samples="{scenario}/mcmc-samples.nc"
    run:
        assemble_chains(input.chains, output.all_samples)



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

rule mcmc_plots_and_summary:
    input:
        samples=lambda wildcards: expand("{scenario}/mcmc-samples.nc", scenario=SCENARIOS.keys())
    output:
        trace_plot= expand("{scenario}/posterior/trace_plot_theta.pdf", scenario=SCENARIOS.keys()),
        posterior_plot= expand("{scenario}/posterior/posterior_plot_theta.pdf", scenario=SCENARIOS.keys()),
        summary= expand("{scenario}/posterior/mcmc_summary_theta.txt", scenario=SCENARIOS.keys())
    run:
        
        for scenario in SCENARIOS.keys():
            
            idata = az.from_netcdf(f"{scenario}/mcmc-samples.nc")
            
 
            az.plot_trace(idata, var_names=['theta'])
            plt.savefig(f"{scenario}/posterior/trace_plot_theta.pdf")
            plt.close()

            az.plot_posterior(idata, var_names=['theta'], kind='kde')
            plt.savefig(f"{scenario}/posterior/posterior_plot_theta.pdf")
            plt.close()


            
            summary = az.summary(idata, var_names=['theta'])
            summary.to_csv(f"{scenario}/posterior/mcmc_summary_theta.txt", sep='\t')
