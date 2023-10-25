import dataclasses

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import seaborn as sns

import pmhn
pmhn.control_no_mutation_warning(silence=True)
matplotlib.use("agg")

# --- Working directory ---
workdir: "generated/mhn-bayes"

N_CHAINS: int = 4


@dataclasses.dataclass
class Settings:
    n_mutations: int
    n_patients: int
    p_offdiag: float
    mean_sampling_time: float = 1.0
    data_seed: int = 111
    prior_sampling_seed: int = 222
    tuning_samples: int = 1000
    mcmc_samples: int = 1000

    smc_particles: int = 1000


SCENARIOS = {
    "small": Settings(n_mutations=8, n_patients=200, p_offdiag=3/8**2),
    # "large": Settings(n_mutations=8, n_patients=2_000, p_offdiag=3/8**2),
}

rule all:
    input:
        theta_matrices = expand("{scenario}/theta.pdf", scenario=SCENARIOS.keys()),
        genotypes = expand("{scenario}/genotypes.pdf", scenario=SCENARIOS.keys()),
        # mcmc_samples = expand("{scenario}/mcmc-samples.nc", scenario=SCENARIOS.keys())
        smc_samples = expand("{scenario}/smc-samples.nc", scenario=SCENARIOS.keys())

rule plot_theta_from_data:
    input:
        arrays="{scenario}/arrays.npz"
    output:
        theta = "{scenario}/theta.pdf"
    run:
        theta = np.load(input.arrays)["theta"]
        fig, ax = plt.subplots()
        pmhn.plot_theta(theta, ax=ax)
        fig.tight_layout()
        fig.savefig(output.theta)

rule plot_genotypes_from_data:
    input:
        arrays="{scenario}/arrays.npz"
    output:
        genotypes = "{scenario}/genotypes.pdf"
    run:
        genotypes = np.load(input.arrays)["genotypes"]
        n_patients, n_mutations = genotypes.shape

        # We will transpose the matrix so that the patients are on the x-axis,
        # as usually we have more patients than mutations.
        characteristic_length = 0.02
        height = max(2, min(characteristic_length * n_mutations, 5))
        width = max(3, min(characteristic_length * n_patients, 10))

        fig, axs = plt.subplots(2, 1, figsize=(width, 2 * height))

        pmhn.plot_genotypes(genotypes, ax=axs[0], patients_label="Patients (sampled)", sort=False)
        pmhn.plot_genotypes(genotypes, ax=axs[1], patients_label="Patients (sorted)", sort=True)

        fig.tight_layout()
        fig.savefig(output.genotypes)


rule generate_data:
    output: 
      arrays="{scenario}/arrays.npz"
    run:
        settings = SCENARIOS[wildcards.scenario]
        
        rng = np.random.default_rng(settings.data_seed)

        theta = pmhn.sample_spike_and_slab(
            rng,
            n_mutations=settings.n_mutations,
            p_offdiag=settings.p_offdiag,
        )

        sampling_times, genotypes = pmhn.simulate_dataset(
            rng=rng,
            theta=theta,
            n_points=settings.n_patients,
            mean_sampling_time=settings.mean_sampling_time,
        )

        np.savez(output.arrays, genotypes=genotypes, theta=theta, sampling_times=sampling_times)


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


rule plot_prior_predictives:
    input:
        thetas = expand("{scenario}/prior/theta_samples.pdf", scenario=SCENARIOS.keys()),
        offdiagonal_histograms = expand("{scenario}/prior/offdiagonal_histograms.pdf", scenario=SCENARIOS.keys()),
        offdiagonal_sparsity = expand("{scenario}/prior/offdiagonal_sparsity.pdf", scenario=SCENARIOS.keys()),
        genotypes = expand("{scenario}/prior/genotype_samples.pdf", scenario=SCENARIOS.keys())


rule plot_prior_predictive_theta:
    input: "{scenario}/prior/samples.nc"
    output: "{scenario}/prior/theta_samples.pdf"
    run:
        idata = az.from_netcdf(str(input))
        samples = idata.prior["theta"][0].values

        fig, _ = pmhn.plot_theta_samples(samples, width=6, height=4)
        fig.savefig(str(output))


rule plot_prior_predictive_offdiagonal_histograms:
    input: "{scenario}/prior/samples.nc"
    output: "{scenario}/prior/offdiagonal_histograms.pdf"
    run:
        idata = az.from_netcdf(str(input))
        thetas = idata.prior["theta"][0].values

        fig, ax = plt.subplots()
        pmhn.plot_offdiagonal_histograms(thetas, ax=ax)
        fig.tight_layout()
        fig.savefig(str(output))      

rule plot_prior_predictive_offdiagonal_sparsity:
    input: "{scenario}/prior/samples.nc"
    output: "{scenario}/prior/offdiagonal_sparsity.pdf"
    run:
        idata = az.from_netcdf(str(input))
        thetas = idata.prior["theta"][0].values

        fig, ax = plt.subplots()
        pmhn.plot_offdiagonal_sparsity(thetas, ax=ax)
        fig.tight_layout()
        fig.savefig(str(output))      

rule plot_prior_predictive_genotypes:
    input: "{scenario}/prior/samples.nc"
    output: "{scenario}/prior/genotype_samples.pdf"
    run:
        settings = SCENARIOS[wildcards.scenario]
        idata = az.from_netcdf(str(input))
        thetas = idata.prior["theta"][0].values

        n_samples: int = 30
        n_patients = settings.n_patients
        n_mutations = settings.n_mutations

        rng = np.random.default_rng(settings.prior_sampling_seed)

        genotype_matrices = np.zeros((n_samples, n_patients, n_mutations), dtype=int)
        for i in range(n_samples):
            _, genotypes = pmhn.simulate_dataset(rng, n_points=n_patients, theta=thetas[i], mean_sampling_time=settings.mean_sampling_time)
            genotype_matrices[i, ...] = genotypes

        fig, _ = pmhn.plot_genotype_samples(genotype_matrices)
        fig.savefig(str(output))


def prepare_full_model(genotypes) -> pm.Model:
    loglikelihood = pmhn.MHNLoglikelihood(data=genotypes, backend=pmhn.MHNCythonBackend())
    model = pmhn.prior_regularized_horseshoe(n_mutations=genotypes.shape[1])

    with model:
        pm.Potential("loglikelihood", loglikelihood(model.theta))

    return model


rule mcmc_sample_one_chain:
    input:
        arrays="{scenario}/arrays.npz"
    output:
        chain="{scenario}/mcmc-chains/{chain}.nc"
    run:
        chain = int(wildcards.chain)
        settings = SCENARIOS[wildcards.scenario]
        genotypes = np.load(input.arrays)["genotypes"]

        model = prepare_full_model(genotypes)
        with model:
            idata = pm.sample(chains=1, random_seed=chain, tune=settings.tuning_samples, draws=settings.mcmc_samples)
        
        idata.to_netcdf(output.chain)

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

def assemble_chains(chain_files, output_file) -> None:
    chains = [az.from_netcdf(chain_file) for chain_file in chain_files]
    all_samples = az.concat(chains, dim="chain")
    all_samples.to_netcdf(output_file)


rule mcmc_assemble_chains:
    input:
        chains=expand("{scenario}/mcmc-chains/{chain}.nc", chain=range(1, N_CHAINS + 1), allow_missing=True)
    output:
        all_samples="{scenario}/mcmc-samples.nc"
    run:
        assemble_chains(input.chains, output.all_samples)

rule smc_sample_one_chain:
    input:
        arrays="{scenario}/arrays.npz"
    output:
        smc_samples="{scenario}/smc-chains/{chain}.nc"
    run:
        chain = int(wildcards.chain)
        settings = SCENARIOS[wildcards.scenario]
        genotypes = np.load(input.arrays)["genotypes"]
    
        model = prepare_full_model(genotypes)
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
