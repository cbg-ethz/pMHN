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

N_CHAINS: int = 2


@dataclasses.dataclass
class Settings:
    n_mutations: int
    n_patients: int
    p_offdiag: float
    mean_sampling_time: float = 1.0
    data_seed: int = 111
    prior_sampling_seed: int = 222


SCENARIOS = {
    "small": Settings(n_mutations=8, n_patients=100, p_offdiag=3/8**2),
    # "large": Settings(n_mutations=25, n_patients=300, p_offdiag=10/25**2),
}

rule all:
    input:
        theta_matrices = expand("{scenario}/theta.pdf", scenario=SCENARIOS.keys()),
        genotypes = expand("{scenario}/genotypes.pdf", scenario=SCENARIOS.keys()),
        samples = expand("{scenario}/mcmc-samples.nc", scenario=SCENARIOS.keys())

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

        model = pmhn.construct_regularized_horseshoe(n_mutations=settings.n_mutations)
        with model:
            idata = pm.sample_prior_predictive(samples=n_samples, random_seed=rng)

        idata.to_netcdf(output.prior_samples)

rule plot_prior_predictives:
    input:
        thetas = expand("{scenario}/prior/theta_samples.pdf", scenario=SCENARIOS.keys()),
        offdiagonal_histograms = expand("{scenario}/prior/offdiagonal_histograms.pdf", scenario=SCENARIOS.keys()),
        offdiagonal_sparsity = expand("{scenario}/prior/offdiagonal_sparsity.pdf", scenario=SCENARIOS.keys())

rule plot_prior_predictive_theta:
    input: "{scenario}/prior/samples.nc"
    output: "{scenario}/prior/theta_samples.pdf"
    run:
        idata = az.from_netcdf(str(input))
        samples = idata.prior["theta"][0].values

        fig, axs = plt.subplots(4, 6, figsize=(12, 8))

        for i, ax in enumerate(axs.ravel()):
            plot_cbar = True  # (i == len(axs.ravel()) - 1)
            pmhn.plot_theta(samples[i], ax=ax, vmin=samples.min(), vmax=samples.max(), no_labels=True, cbar=plot_cbar)

        fig.tight_layout()
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


rule generate_samples_for_one_chain:
    input:
        arrays="{scenario}/arrays.npz"
    output:
        chain="{scenario}/mcmc-chains/{chain}.nc"
    run:
        chain = int(wildcards.chain)
        settings = SCENARIOS[wildcards.scenario]
        genotypes = np.load(input.arrays)["genotypes"]

        loglikelihood = pmhn.MHNLoglikelihood(
            data=genotypes,
            backend=pmhn.MHNCythonBackend(),
        )

        model = pmhn.construct_regularized_horseshoe(n_mutations=genotypes.shape[1])

        with model:
            pm.Potential("loglikelihood", loglikelihood(model.theta))
            idata = pm.sample(chains=1, random_seed=chain, tune=2, draws=2)
        
        idata.to_netcdf(output.chain)


rule generate_samples_all:
    input:
        chains=expand("{scenario}/mcmc-chains/{chain}.nc", chain=range(1, N_CHAINS + 1), allow_missing=True)
    output:
        all_samples="{scenario}/mcmc-samples.nc"
    run:
        chains = [az.from_netcdf(chain_file) for chain_file in input.chains]
        all_samples = az.concat(chains, dim="chain")
        all_samples.to_netcdf(output.all_samples)
