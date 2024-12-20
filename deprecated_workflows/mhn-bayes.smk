import dataclasses

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("agg")

import numpy as np

import pmhn
from pmhn import mhn

import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


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
