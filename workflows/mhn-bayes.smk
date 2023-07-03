import dataclasses
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


@dataclasses.dataclass
class Settings:
    n_mutations: int
    n_patients: int
    p_offdiag: float
    mean_sampling_time: float = 1.0
    data_seed: int = 111


SCENARIOS = {
    "small": Settings(n_mutations=8, n_patients=200, p_offdiag=3/8**2),
    "large": Settings(n_mutations=25, n_patients=300, p_offdiag=10/25**2),
}

rule all:
    input:
        theta_matrices = expand("{scenario}/theta.pdf", scenario=SCENARIOS.keys()),
        genotypes = expand("{scenario}/genotypes.pdf", scenario=SCENARIOS.keys())

rule plot_theta_from_data:
    input:
        arrays="{scenario}/arrays.npz"
    output:
        theta = "{scenario}/theta.pdf"
    run:
        theta = np.load(input.arrays)["theta"]
        fig, ax = plt.subplots()
        sns.heatmap(theta, ax=ax, square=True, cmap="coolwarm", center=0)
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

        index_sorted = sorted(np.arange(n_patients), key=lambda i: -np.sum(genotypes[i, :]))

        fig, axs = plt.subplots(2, 1, figsize=(width, 2 * height))
        
        sns.heatmap(genotypes.T, ax=axs[0], cmap="Greys", square=True, vmin=0, vmax=1, cbar=False)
        sns.heatmap(genotypes[index_sorted, :].T, ax=axs[1], cmap="Greys", square=True, vmin=0, vmax=1, cbar=False)
        

        axs[0].set_xlabel("Patients (as in the data set)")
        axs[1].set_xlabel("Patients (sorted by number of mutations)")

        for ax in axs:
            ax.set_xticks([], [])
            ax.set_yticks([], [])
            ax.set_ylabel("Genes")
        
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

# loglikelihood = pmhn.MHNLoglikelihood(
#     data=mutations,
#     backend=pmhn.MHNCythonBackend(),
# )

# with pm.Model() as model:  # type: ignore
#     theta_var = pm.Cauchy("theta", alpha=0.0, beta=0.1, shape=theta.shape)
#     pm.Potential("loglikelihood", loglikelihood(theta_var))


# t0 = time.time()
# n_tune = 200
# n_samples = 200
# n_chains = 4

# print("Sampling...")


# with model:
#     idata = pm.sample(chains=n_chains, random_seed=rng, tune=n_tune, draws=n_samples)

# idata.to_netcdf("idata.nc")

# print(idata)


# t1 = time.time()

# print(f"Sampling {n_chains * (n_tune + n_samples)} took {t1 - t0:.2f} seconds")
