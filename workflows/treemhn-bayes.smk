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
from anytree import RenderTree, Node, LevelOrderGroupIter
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
    tuning_samples: int = 24
    mcmc_samples: int = 24

    smc_particles: int = 24


SCENARIOS = {
    #"small_treemhn_spike_and_slab_0.05_mcmc_normal": Settings(n_mutations=10, n_patients=200, p_offdiag=3/8**2),
    "x2000_patients_100_samples_3_mutations_normal_10.0=None": Settings(n_mutations=3, n_patients=2000, p_offdiag=3/8**2), 
}

rule all:
    input:
        theta_matrices = expand("{scenario}/theta.pdf", scenario=SCENARIOS.keys()),
        theta_csv = expand("{scenario}/theta.csv", scenario=SCENARIOS.keys()),
        trees = expand("{scenario}/trees.pdf", scenario=SCENARIOS.keys()),
        sampling_times = expand("{scenario}/sampling_times.pdf", scenario=SCENARIOS.keys()),
        mcmc_samples = expand("{scenario}/mcmc-samples.nc", scenario=SCENARIOS.keys()),
        trees_prior = expand("{scenario}/prior/prior_trees.pdf", scenario=SCENARIOS.keys()),
        theta_samples = expand("{scenario}/prior/theta_samples.pdf", scenario=SCENARIOS.keys()),
        posterior_theta_plots=expand("{scenario}/posterior_theta_from_mcmc.pdf", scenario=SCENARIOS.keys()),
        trace_plot= expand("{scenario}/posterior/trace_plot_theta.pdf", scenario=SCENARIOS.keys()),
        histogram_plot=expand("{scenario}/posterior/all_histograms_theta.pdf", scenario=SCENARIOS.keys()),
        
        summary= expand("{scenario}/posterior/mcmc_summary_theta.txt", scenario=SCENARIOS.keys()),
        stats=expand("{scenario}/posterior/stats.pdf", scenario=SCENARIOS.keys()),
        posterior_plot= expand("{scenario}/posterior/posterior_plot_theta.pdf", scenario=SCENARIOS.keys()),
        mean_theta_posterior_csv = expand("{scenario}/mean_posterior_theta.csv", scenario=SCENARIOS.keys()),
        mean_theta_posterior_pdf = expand("{scenario}/mean_posterior_theta.pdf", scenario=SCENARIOS.keys())
rule generate_trees_from_mean_prior_samples:
    input: 
        prior_samples="{scenario}/prior/samples.nc"
    output: 
        trees_csv = "{scenario}/prior/prior_trees.csv",
        trees_prior = "{scenario}/prior/prior_trees.pdf"

    run:
        settings = SCENARIOS[wildcards.scenario]
        rng = np.random.default_rng(settings.data_seed)
        idata = az.from_netcdf(str(input))
        samples = idata.prior["theta"][0].values
        print(samples.shape) 
        mean_theta = np.mean(samples, axis=0)
        _, trees = simulate_trees(
            rng=rng,
            n_points=settings.n_patients,
            theta=mean_theta,
            mean_sampling_time=settings.mean_sampling_time, min_tree_size = None,max_tree_size = 25
        )
        write_trees_to_csv(trees, output.trees_csv)
        trees_from_csv = pd.read_csv(output.trees_csv)

        tree_sizes = trees_from_csv.groupby("Tree_ID").size()
        plt.hist(tree_sizes, alpha=0.5, edgecolor="k", label="Trees")
        plt.xlabel("Tree Size")
        plt.ylabel("Frequency")
        plt.legend(loc="upper right")
        plt.title("Tree Size Distribution")
        plt.savefig(output.trees_prior) 
rule plot_theta_from_data:
    input:
        arrays="{scenario}/arrays.npz"
    output:
        theta = "{scenario}/theta.pdf",
        theta_csv = "{scenario}/theta.csv"
    run:
        theta = np.load(input.arrays)["theta"]
        print(theta)
        fig, ax = plt.subplots()
        pmhn.plot_theta(theta, ax=ax)
        fig.tight_layout()
        fig.savefig(output.theta)
        pd.DataFrame(theta).to_csv(output.theta_csv, index=False)
rule plot_sampling_times_from_data:
    input:
        arrays="{scenario}/arrays.npz"
    output:
        sampling_times = "{scenario}/sampling_times.pdf",
    run:
        sampling_times = np.load(input.arrays)["sampling_times"]
        
        fig, ax = plt.subplots()
        ax.plot(range(len(sampling_times)), sampling_times, marker='o')
        
        ax.set_xlabel('Tree Index')
        ax.set_ylabel('Sampling Time')
        ax.set_title('Sampling Times for Each Tree')
        
        fig.tight_layout()
        
        fig.savefig(output.sampling_times)

def write_trees_to_csv(trees, output_file_path):
    with open(output_file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Patient_ID", "Tree_ID", "Node_ID", "Mutation_ID", "Parent_ID"]
        )

        patient_id = 0
        for tree in trees:
            patient_id += 1
            tree_id = patient_id
            node_id = 0
            node_id_dict= {}
            for level in LevelOrderGroupIter(tree):
                for node in level: 
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
        

        trees = np.load(input.arrays, allow_pickle=True)["trees"]
        write_trees_to_csv(trees, output.trees_csv)
        trees_from_csv = pd.read_csv(output.trees_csv)
        tree_sizes = trees_from_csv.groupby("Tree_ID").size().astype(int)

        plt.hist(tree_sizes, bins=range(tree_sizes.min(), tree_sizes.max() + 1), alpha=0.5, edgecolor="k", label="Trees")
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
[-0.282,0.0,0.0],
[-6.0,-0.45199999999999996,0.0],
[0.0,-6.86,-0.51]

        ]
    ) 

        np.fill_diagonal(theta, np.diag(theta)*1.25)
        print(theta) 
        sampling_times, trees = simulate_trees(
            rng=rng,
            n_points=settings.n_patients,
            theta=theta,
            mean_sampling_time=settings.mean_sampling_time, min_tree_size = None,max_tree_size = None
        )
        trees_wrapper = []
        for tree in trees:
            print(RenderTree(tree))
            tree_wrapper = TreeWrapperCode(tree)
            trees_wrapper.append(tree_wrapper)

 
        np.savez(output.arrays, trees = trees, trees_wrapper=trees_wrapper, theta=theta, sampling_times=sampling_times)

def prepare_full_model(trees, mean_sampling_time, n_mutations, all_mut) -> pm.Model:
    loglikelihood = TreeMHNLoglikelihood(data=trees, mean_sampling_time = mean_sampling_time, all_mut = all_mut, backend=TreeMHNBackendCode())
    model = pmhn.prior_normal(n_mutations=n_mutations, sigma=10.0)

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

        model = pmhn.prior_normal(n_mutations=settings.n_mutations, sigma=10.0)
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
    threads: 8 
    run:
        chain = int(wildcards.chain)
        settings = SCENARIOS[wildcards.scenario]
        with np.load(input.arrays, allow_pickle=True) as data:
            trees = data["trees_wrapper"]
            theta = data["theta"]
        n_mutations = len(theta) 
        all_mut = set(i + 1 for i in range(n_mutations))
        model = prepare_full_model(trees, settings.mean_sampling_time, n_mutations, all_mut)
        with model:
            idata = pm.sample(chains=1, random_seed=chain, tune=settings.tuning_samples, draws=settings.mcmc_samples
            )        
        print(idata.sample_stats)
        idata.to_netcdf(output.chain) 
rule sampler_stats:
    input:
        samples=lambda wildcards: expand("{scenario}/mcmc-samples.nc", scenario=SCENARIOS.keys())
    output:
        stats=expand("{scenario}/posterior/stats.pdf", scenario=SCENARIOS.keys())
    run:
        idata = az.from_netcdf(str(input))
        print(idata.sample_stats['accepted'].values)
        print(idata.sample_stats['accept'].values)
        az.plot_posterior(idata, group="sample_stats", var_names="accept", hdi_prob="hide", kind="hist")
        plt.savefig(str(output))
        plt.close()

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
        posterior_theta_plots="{scenario}/posterior_theta_from_mcmc.pdf",
        mean_theta_posterior_csv = "{scenario}/mean_posterior_theta.csv",
        mean_theta_posterior_pdf = "{scenario}/mean_posterior_theta.pdf"
    run:
        idata = az.from_netcdf(str(input))
        posterior_samples = idata.posterior["theta"][0].values
        theta_size = len(posterior_samples[0][0]) 
        mean_theta = np.zeros((theta_size, theta_size))
        for i in range(len(posterior_samples)):
            mean_theta += posterior_samples[i]
        mean_theta /= len(posterior_samples)
        
        pd.DataFrame(mean_theta).to_csv(output.mean_theta_posterior_csv, index=False)
        fig, _ = pmhn.plot_theta_samples(posterior_samples, width=6, height=4)
        fig.savefig(output.posterior_theta_plots)

        fig, ax = plt.subplots()
        pmhn.plot_theta(mean_theta, ax=ax)
        fig.tight_layout()
        fig.savefig(output.mean_theta_posterior_pdf)

rule mcmc_plots_and_summary:
    input:
        samples=lambda wildcards: expand("{scenario}/mcmc-samples.nc", scenario=SCENARIOS.keys()),
        arrays=lambda wildcards: expand("{scenario}/arrays.npz", scenario=SCENARIOS.keys())
    output:
        trace_plot=expand("{scenario}/posterior/trace_plot_theta.pdf", scenario=SCENARIOS.keys()),
        histogram_plot=expand("{scenario}/posterior/all_histograms_theta.pdf", scenario=SCENARIOS.keys()),
        posterior_plot= expand("{scenario}/posterior/posterior_plot_theta.pdf", scenario=SCENARIOS.keys()),
        summary=expand("{scenario}/posterior/mcmc_summary_theta.txt", scenario=SCENARIOS.keys())
    run:
        for scenario in SCENARIOS.keys():
            idata = az.from_netcdf(f"{scenario}/mcmc-samples.nc")
            theta_matrix = np.load(f"{scenario}/arrays.npz")['theta']

            
            az.plot_trace(idata, var_names=['theta'])
            plt.savefig(f"{scenario}/posterior/trace_plot_theta.pdf")
            plt.close()

            nrows, ncols = theta_matrix.shape
            figsize = (nrows, nrows * 2)  
            fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
            axs = axs.flatten()  

            for i in range(nrows):
                for j in range(ncols):
                    ax = axs[i * ncols + j]
                    theta_samples = idata.posterior['theta'].values[0, :, i, j]
                    true_theta = theta_matrix[i, j]
                    
                    ax.hist(theta_samples, bins=30, color='skyblue', edgecolor='black')
                    ax.axvline(true_theta, color='orange', linestyle='--', label=f'True θ[{i},{j}] = {true_theta:.2f}')
                    ax.legend()
                    
                    ax.set_title(f'Entry ({i},{j})')
                    ax.set_xlabel('θ values')
                    ax.set_ylabel('Frequency')

            plt.tight_layout()  
            plt.savefig(f"{scenario}/posterior/all_histograms_theta.pdf")
            plt.close()
            
            summary = az.summary(idata, var_names=['theta'])
            summary.to_csv(f"{scenario}/posterior/mcmc_summary_theta.txt", sep='\t')

            az.plot_posterior(idata, var_names=['theta'], kind='kde')
            plt.savefig(f"{scenario}/posterior/posterior_plot_theta.pdf")
            plt.close()




