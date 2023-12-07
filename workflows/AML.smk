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
from pmhn._trees._backend_new_v2 import TreeMHNBackendCode, TreeWrapperCode 
from anytree import RenderTree, Node, LevelOrderGroupIter
matplotlib.use("agg")
import pmhn._trees._io as io

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
    tuning_samples: int = 200 
    mcmc_samples: int = 200 

    smc_particles: int = 100 


SCENARIOS = {
    #"small_treemhn_spike_and_slab_0.05_mcmc_normal": Settings(n_mutations=10, n_patients=200, p_offdiag=3/8**2),
    "AML_normal_400_samples": Settings(n_mutations=31, n_patients=123, p_offdiag=3/8**2),
}

rule all:
    input:
        trees = expand("{scenario}/trees.pdf", scenario=SCENARIOS.keys()),
        mcmc_samples = expand("{scenario}/mcmc-samples.nc", scenario=SCENARIOS.keys()),
        theta_samples = expand("{scenario}/prior/theta_samples.pdf", scenario=SCENARIOS.keys()),
        posterior_theta_plots=expand("{scenario}/posterior_theta_from_mcmc.pdf", scenario=SCENARIOS.keys()),
        
        trace_plots_3=expand("{scenario}/posterior/trace_plot_theta_DNMT3A.pdf", scenario=SCENARIOS.keys(), mutation=["DNMT3A", "IDH2", "FLT3", "NRAS", "NPM1"]),
        trace_plots_8=expand("{scenario}/posterior/trace_plot_theta_IDH2.pdf", scenario=SCENARIOS.keys(), mutation=["DNMT3A", "IDH2", "FLT3", "NRAS", "NPM1"]),
        trace_plots_0=expand("{scenario}/posterior/trace_plot_theta_FLT3.pdf", scenario=SCENARIOS.keys(), mutation=["DNMT3A", "IDH2", "FLT3", "NRAS", "NPM1"]),
        trace_plots_5=expand("{scenario}/posterior/trace_plot_theta_NRAS.pdf", scenario=SCENARIOS.keys(), mutation=["DNMT3A", "IDH2", "FLT3", "NRAS", "NPM1"]),
        trace_plots_1=expand("{scenario}/posterior/trace_plot_theta_NPM1.pdf", scenario=SCENARIOS.keys(), mutation=["DNMT3A", "IDH2", "FLT3", "NRAS", "NPM1"]),
        histogram_plots_3=expand("{scenario}/posterior/histograms_theta_DNMT3A.pdf", scenario=SCENARIOS.keys()),
        histogram_plots_8=expand("{scenario}/posterior/histograms_theta_IDH2.pdf", scenario=SCENARIOS.keys()),
        histogram_plots_0=expand("{scenario}/posterior/histograms_theta_FLT3.pdf", scenario=SCENARIOS.keys()),
        histogram_plots_5=expand("{scenario}/posterior/histograms_theta_NRAS.pdf", scenario=SCENARIOS.keys()),
        histogram_plots_1=expand("{scenario}/posterior/histograms_theta_NPM1.pdf", scenario=SCENARIOS.keys()),
        summary= expand("{scenario}/posterior/mcmc_summary_theta.txt", scenario=SCENARIOS.keys()),
        stats=expand("{scenario}/posterior/stats.pdf", scenario=SCENARIOS.keys()),
        posterior_plot= expand("{scenario}/posterior/posterior_plot_theta.pdf", scenario=SCENARIOS.keys()),
        mean_theta_posterior_csv = expand("{scenario}/mean_posterior_theta.csv", scenario=SCENARIOS.keys()),
        mean_theta_posterior_pdf = expand("{scenario}/mean_posterior_theta.pdf", scenario=SCENARIOS.keys()),
        trees_prior = expand("{scenario}/prior/prior_trees.pdf", scenario=SCENARIOS.keys()),
        mean_theta_prior_pdf = expand("{scenario}/prior/mean_prior_theta.pdf", scenario=SCENARIOS.keys()),
        ess_csv=expand("{scenario}/posterior/ess_theta_diagonal.csv", scenario=SCENARIOS.keys()) 
rule plot_trees_from_data:
    output:
        trees = "{scenario}/trees.pdf",
    run:

        trees_from_csv= pd.read_csv("/cluster/home/laukeller/pMHN/workflows/trees_AML_R.csv")

        tree_sizes = trees_from_csv.groupby("Tree_ID").size()
        plt.hist(tree_sizes, alpha=0.5, edgecolor="k", label="Trees")
        plt.xlabel("Tree Size")
        plt.ylabel("Frequency")
        plt.legend(loc="upper right")
        plt.title("Tree Size Distribution")
        plt.savefig(output.trees)
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

rule generate_trees_from_mean_prior_samples:
    input: 
        prior_samples="{scenario}/prior/samples.nc"
    output: 
        trees_csv = "{scenario}/prior/prior_trees.csv", 
        trees_prior = "{scenario}/prior/prior_trees.pdf",
        mean_theta_prior_pdf = "{scenario}/prior/mean_prior_theta.pdf",
        mean_theta_prior_csv = "{scenario}/prior/mean_prior_theta.csv"
    run:
        settings = SCENARIOS[wildcards.scenario]
        rng = np.random.default_rng(settings.data_seed)
        idata = az.from_netcdf(str(input))
        prior_samples = idata.prior["theta"][0].values
        theta_size = len(prior_samples[0][0]) 
        mean_theta = np.zeros((theta_size, theta_size))
        for i in range(len(prior_samples)):
            mean_theta += prior_samples[i]
        mean_theta /= len(prior_samples) 
        pd.DataFrame(mean_theta).to_csv(output.mean_theta_prior_csv, index=False)
        

        fig, ax = plt.subplots()
        pmhn.plot_theta(mean_theta, ax=ax)
        fig.tight_layout()
        fig.savefig(output.mean_theta_prior_pdf)
        plt.close()
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

rule generate_data_AML:
    output: 
      arrays="{scenario}/arrays.npz"
    
    run:
        settings = SCENARIOS[wildcards.scenario]
        df_AML = pd.read_csv("/cluster/home/laukeller/pMHN/workflows/trees_AML_R.csv")
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
        trees_AML = io.parse_forest(df_AML, naming=naming)

        trees_wrapper = []
        for tree in trees_AML.values():
            print(RenderTree(tree))
            tree_wrapper = TreeWrapperCode(tree)
            trees_wrapper.append(tree_wrapper)

 
        np.savez(output.arrays, trees_wrapper=trees_wrapper)

def prepare_full_model(trees, mean_sampling_time, n_mutations, all_mut) -> pm.Model:
    loglikelihood = TreeMHNLoglikelihood(data=trees, mean_sampling_time = mean_sampling_time, all_mut = all_mut, backend=TreeMHNBackendCode())
    model = pmhn.prior_normal(n_mutations=n_mutations)

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

        model = pmhn.prior_normal(n_mutations=settings.n_mutations)
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
        n_mutations = 31 
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
        posterior_plot= expand("{scenario}/posterior/posterior_plot_theta.pdf", scenario=SCENARIOS.keys()),
        summary=expand("{scenario}/posterior/mcmc_summary_theta.txt", scenario=SCENARIOS.keys())
    run:
        for scenario in SCENARIOS.keys():
            idata = az.from_netcdf(f"{scenario}/mcmc-samples.nc")
            

            
            
 
            
            summary = az.summary(idata, var_names=['theta'])
            summary.to_csv(f"{scenario}/posterior/mcmc_summary_theta.txt", sep='\t')

            az.plot_posterior(idata, var_names=['theta'], kind='kde')
            plt.savefig(f"{scenario}/posterior/posterior_plot_theta.pdf")
            plt.close()

rule mcmc_histogram_plots:
    input:
        samples=lambda wildcards: expand("{scenario}/mcmc-samples.nc", scenario=SCENARIOS.keys()),
        arrays=lambda wildcards: expand("{scenario}/arrays.npz", scenario=SCENARIOS.keys())
    output:
        histogram_plots=expand("{scenario}/posterior/histograms_theta_{mutation}.pdf", scenario=SCENARIOS.keys(), mutation=["DNMT3A", "IDH2", "FLT3", "NRAS", "NPM1"]),
        ess_csv=expand("{scenario}/posterior/ess_theta_diagonal.csv", scenario=SCENARIOS.keys()),
        trace_plots=expand("{scenario}/posterior/trace_plot_theta_{mutation}.pdf", scenario=SCENARIOS.keys(), mutation=["DNMT3A", "IDH2", "FLT3", "NRAS", "NPM1"]),
    run:
        for scenario in SCENARIOS.keys():
            idata = az.from_netcdf(f"{scenario}/mcmc-samples.nc")

            mutation_mapping = {3: "DNMT3A", 8: "IDH2", 0: "FLT3", 5: "NRAS", 1: "NPM1"}
            ncols = 7
            nrows = (31 + ncols - 1) // ncols
            for row, mutation in mutation_mapping.items():
                figsize = (ncols * 3, nrows * 3)  # Adjust as needed
                fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
                axs = axs.flatten()  # Flatten the array of axes

                theta_values = idata.posterior['theta'].values
                
                for i in range(31):  # Assuming you have 31 columns for the mutations
                    trace = theta_values[0, :, row, i]  # Select the chain and theta matrix entry
                    ax = axs[i]
                    
                    ax.plot(trace, color='royalblue')
                    ax.set_title(f'{mutation} [{row},{i}]')
                    ax.set_xlabel('Sample')
                    ax.set_ylabel('Value')

                for i in range(31, nrows * ncols):
                    axs[i].axis('off')

                plt.tight_layout()
                plt.savefig(f"{scenario}/posterior/trace_plot_theta_{mutation}.pdf")
                plt.close()

            for row, mutation in mutation_mapping.items():
                figsize = (ncols * 4, nrows * 4)
                fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

                for i in range(31):
                    r, c = divmod(i, ncols)
                    ax = axs[r, c]
                    theta_samples = idata.posterior['theta'].values[0, :, row, i]
                    
                    ax.hist(theta_samples, bins=30, color='skyblue', edgecolor='black')
                    ax.legend()

                    ax.set_title(f'{mutation} Entry ({row},{i})')
                    ax.set_xlabel('θ values')
                    ax.set_ylabel('Frequency')

                for i in range(31, nrows * ncols):
                    r, c = divmod(i, ncols)
                    axs[r, c].axis('off')

                plt.subplots_adjust(wspace=0.4, hspace=0.6)
                plt.savefig(f"{scenario}/posterior/histograms_theta_{mutation}.pdf")
                plt.close()

 
            ess_theta = az.ess(idata, var_names=['theta'])

            ess_data = {'theta_{}_{}'.format(i, i): ess_theta.sel(theta_dim_0=i, theta_dim_1=i).theta.values
                        for i in range(31)}

            ess_df = pd.DataFrame([ess_data])

            ess_csv_file = f"{scenario}/posterior/ess_theta_diagonal.csv"
            ess_df.to_csv(ess_csv_file, index=False) 
