from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pymc as pm
import pydantic

import pmhn

# --- Working directory ---
workdir: "generated/mhn-lasso"

matplotlib.use("agg")


N_MAP_RUNS: int = 4
N_PATIENTS: list[int] = [20, 100, 500]
MAX_OPTIMIZATON_STEPS: int = 200
REGULARISATION: list[float] = [0.1, 0.5, 1.0, 3.0, 20.0]

class ThetaSettings(pydantic.BaseModel):
    n_mutations: int
    true_effect: float
    probability_nonzero: pydantic.confloat(gt=0.0, lt=1.0)
    seed: int

    sampling_type: Literal["bernoulli", "spike_and_slab"]


def _generate_theta_bernoulli(settings: ThetaSettings) -> np.ndarray:
    rng = np.random.default_rng(settings.seed)
    theta = settings.true_effect * rng.binomial(1, settings.probability_nonzero, size=(settings.n_mutations, settings.n_mutations))

    np.fill_diagonal(theta, settings.true_effect)

    return theta

def _generate_theta_spike_and_slab(settings: ThetaSettings) -> np.ndarray:
    rng = np.random.default_rng(settings.seed)
    mask = rng.binomial(1, settings.probability_nonzero, size=(settings.n_mutations, settings.n_mutations))
    np.fill_diagonal(mask, 1)
    
    normal = settings.true_effect * rng.normal(size=(settings.n_mutations, settings.n_mutations))

    return mask * normal

def generate_theta(settings: ThetaSettings) -> np.ndarray:
    if settings.sampling_type == "bernoulli":
        return _generate_theta_bernoulli(settings)
    elif settings.sampling_type == "spike_and_slab":
        return _generate_theta_spike_and_slab(settings)
    else:
        raise ValueError(f"Unknown sampling type {settings.sampling_type}")

THETAS = {
    "spike-and-slab-10": ThetaSettings(n_mutations=10, true_effect=2.0, probability_nonzero=8/10**2, seed=0, sampling_type="spike_and_slab"),
}

rule all:
    input: expand("{theta_spec}/{n_patients}/result_visualisation-{regularisation}.pdf", theta_spec=THETAS, n_patients=N_PATIENTS, regularisation=REGULARISATION)

rule generate_theta:
    output:
      matrix="{theta_spec}/theta.txt",
      visualisation="{theta_spec}/theta.pdf"
    run:
        theta = generate_theta(THETAS[wildcards.theta_spec])
        # Save matrix
        np.savetxt(str(output.matrix), theta)
        # Plot visualisation
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(theta, ax=ax, square=True, cbar=False, cmap="coolwarm", center=0.0)
        fig.tight_layout()
        fig.savefig(str(output.visualisation))


rule generate_data:
    input:
        matrix="{theta_spec}/theta.txt"
    output:
        data="{theta_spec}/{n_patients}/patient_data.txt"
    run:
        n_patients = int(wildcards.n_patients)
        theta = np.loadtxt(input.matrix)

        rng = np.random.default_rng(12)
        _, data = pmhn.simulate_dataset(
            rng=rng,
            n_points=n_patients,
            theta=theta,
            mean_sampling_time=1.0,
        )
        np.savetxt(str(output.data), data)


rule plot_data_summary:
    input:
        data = "{theta_spec}/{n_patients}/patient_data.txt"
    output:
        histogram = "{theta_spec}/{n_patients}/data_visualisation.pdf"
    run:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        data = np.loadtxt(input.data)

        # Plot mutation frequencies
        ax = axs[0]
        mutation_frequencies = data.mean(axis=0)
        x_axis = np.arange(len(mutation_frequencies))
        ax.scatter(x_axis, mutation_frequencies, s=5)
        ax.set_xlabel("Gene")
        ax.set_ylabel("Mutation frequency")
        ax.set_title(f"Mutation frequencies, $n={wildcards.n_patients}$")
        ax.set_ylim(0, 1)

        # Plot number of mutations histogram
        ax = axs[1]
        n_mutations = data.sum(axis=1)
        ax.hist(n_mutations, bins=data.shape[1], density=False)
        ax.set_title("Number of mutations per patient")

        # Save figure
        fig.tight_layout()
        fig.savefig(str(output.histogram))


rule fit_map:
    input: "{theta_spec}/{n_patients}/patient_data.txt"
    output: "{theta_spec}/{n_patients}/map_estimate-{regularisation}-{map_seed}.txt"
    run:
        regularisation = float(wildcards.regularisation)
        
        # Read the data
        data = np.asarray(np.loadtxt(str(input)), dtype=np.int32)
        n_genes = data.shape[1]
        
        # Apply bootstrap
        rng = np.random.default_rng(int(wildcards.map_seed))
        data = data[rng.choice(data.shape[0], size=data.shape[0], replace=True), :]

        # Fit the model MAP
        loglikelihood = pmhn.MHNLoglikelihood(data=data, backend=pmhn.MHNCythonBackend())
        with pm.Model() as model:
            theta_var = pm.Laplace("theta", mu=0.0, b=regularisation, shape=(n_genes, n_genes))
            pm.Potential("loglikelihood", loglikelihood(theta_var))

        with model:
            solution = pm.find_MAP(maxeval=MAX_OPTIMIZATON_STEPS, seed=int(wildcards.map_seed))
            found_map = solution["theta"]

        np.savetxt(str(output), found_map)


rule visualise_maps:
    input: 
        ground_truth = "{theta_spec}/theta.txt",
        maps = expand("{theta_spec}/{n_patients}/map_estimate-{regularisation}-{map_seed}.txt", map_seed=range(N_MAP_RUNS), allow_missing=True)
    output:
        thetas = "{theta_spec}/{n_patients}/result_visualisation-{regularisation}.pdf"
    run:
        theta = np.loadtxt(str(input.ground_truth))
        maps = [np.loadtxt(str(map)) for map in input.maps]
        regularisation = float(wildcards.regularisation)

        fig, axs = plt.subplots(1, len(maps) + 1, figsize=(5*(len(maps)+1), 5))
        
        scale_min = min(np.min([np.min(map) for map in maps]), np.min(theta))
        scale_max = max(np.max([np.max(map) for map in maps]), np.max(theta))

        ax = axs[0]
        ax.set_title("Ground truth")
        sns.heatmap(theta, ax=ax, center=0, cmap="coolwarm", square=True, vmin=scale_min, vmax=scale_max, cbar=True)

        for i, map in enumerate(maps):
            ax = axs[i+1]
            ax.set_title(f"MAP estimate {i+1}")
            
            sns.heatmap(map, ax=ax, center=0, cmap="coolwarm", square=True, vmin=scale_min, vmax=scale_max, cbar=True)

        fig.tight_layout()
        fig.savefig(str(output))