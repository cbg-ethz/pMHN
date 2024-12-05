# In this workflow we will demonstrate the utility of
# personalised MHN framework. We will have two subpopulations
# of patients (smoking/nonsmoking).
# Additionally, we will model patient age.
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns

import pmhn

matplotlib.use("agg")

# --- Working directory ---
workdir: "generated/pmhn-demo"

MAX_OPTIMIZATON_STEPS: int = 200

N_MUTATIONS: list[int] = [10]
N_PATIENTS: list[int] = [200]
REGULARISATION: list[float] = [0.1, 1.0, 10.0]
REGULATISATION_2: list[float] = [0.1, 1.0, 10.0]

rule all:
    input:
        vanilla_mhn = expand("{n_mutations}-{n_patients}/mhn-theta-{regularisation}.txt", n_mutations=N_MUTATIONS, n_patients=N_PATIENTS, regularisation=REGULARISATION),
        personalised_mhn = expand("{n_mutations}-{n_patients}/pmhn-map-{regularisation}-{regg}.npz", n_mutations=N_MUTATIONS, n_patients=N_PATIENTS, regularisation=REGULARISATION, regg=REGULATISATION_2)

rule generate_data:
    output:
        arrays = "{n_mutations}-{n_patients}/arrays.npz",
        base_theta = "{n_mutations}-{n_patients}/base_theta.pdf",
        design_matrix = "{n_mutations}-{n_patients}/design_matrix.pdf",
        effect_sizes = "{n_mutations}-{n_patients}/effect_sizes.pdf"
    run:
        n_mutations = int(wildcards.n_mutations)
        n_patients = int(wildcards.n_patients)

        p_smoking = 0.7

        rng = np.random.default_rng(32)
    
        mask = rng.binomial(1, 5 / (n_mutations**2), size=(n_mutations, n_mutations))
        np.fill_diagonal(mask, 1)
    
        # Base theta (for nonsmokers at age 0)
        theta_base = 0.5 * rng.normal(size=mask.shape) * mask

        # Smoking effects
        smoking_effect = 0.5 * np.abs(rng.normal(size=n_mutations))

        # Age effects
        age_effect = 0.25 * np.abs(rng.normal(size=n_mutations))

        # Design matrix
        # Centered age
        age = rng.normal(size=n_patients)
        # Smoking status
        smoking = rng.binomial(1, p=p_smoking, size=n_patients)
        design_matrix = np.hstack([age[:, None], smoking[:, None]])

        # Generate individual thetas
        thetas = np.zeros((n_patients, n_mutations, n_mutations))
        for i in range(n_patients):
            theta = theta_base + smoking[i] * smoking_effect + age[i] * age_effect
            thetas[i] = theta

        _, data = pmhn.simulate_dataset(rng=rng, n_points=n_patients, theta=thetas, mean_sampling_time=1.0)

        np.savez(
            str(output.arrays),
            thetas=thetas,
            design_matrix=design_matrix,
            theta_base=theta_base,
            smoking_effect=smoking_effect,
            age_effect=age_effect,
            data=data,
        )

        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(theta_base, ax=ax, center=0, cbar=True, square=True, cmap="coolwarm")
        ax.set_title("Base theta")
        ax.set_xlabel("Genes")
        ax.set_ylabel("Genes")
        fig.tight_layout()
        fig.savefig(str(output.base_theta))

        fig, ax = plt.subplots(figsize=(2, 5))
        sns.heatmap(design_matrix, ax=ax, cbar=True, square=False, cmap="coolwarm")
        ax.set_title("Design matrix")
        ax.set_xticks([0.5, 1.5], ["Age", "Smoking"])
        ax.set_ylabel("Patients")
        fig.tight_layout()
        fig.savefig(str(output.design_matrix))

        fig, ax = plt.subplots(figsize=(7, 2))
        effect_sizes = np.hstack([age_effect[:, None], smoking_effect[:, None]]).T
        sns.heatmap(effect_sizes, ax=ax, cbar=True, square=False, cmap="coolwarm", vmin=-0.001, center=0, vmax=2)
        ax.set_title("Effect sizes")
        ax.set_xlabel("Genes")
        ax.set_yticks([0.5, 1.5], ["Age", "Smoking"])
        fig.tight_layout()
        fig.savefig(str(output.effect_sizes))


rule fit_map_mhn:
    input: "{n_mutations}-{n_patients}/arrays.npz"
    output:
      map_array = "{n_mutations}-{n_patients}/mhn-theta-{regularisation}.txt",
      map_picture = "{n_mutations}-{n_patients}/mhn-theta-{regularisation}.pdf"
    run:
        regularisation = float(wildcards.regularisation)
        arrays = np.load(str(input))
        
        data = arrays["data"]
        n_patients, n_genes = data.shape

        # Fit the model MAP
        loglikelihood = pmhn.MHNLoglikelihood(data=data, backend=pmhn.MHNCythonBackend())
        with pm.Model() as model:
            theta_var = pm.Laplace("theta", mu=0.0, b=regularisation, shape=(n_genes, n_genes))
            pm.Potential("loglikelihood", loglikelihood(theta_var))

        with model:
            solution = pm.find_MAP(maxeval=MAX_OPTIMIZATON_STEPS, seed=42)
            found_map = solution["theta"]

        np.savetxt(str(output.map_array), found_map)

        fig, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(found_map, ax=ax, center=0, cbar=True, square=True, cmap="coolwarm")
        fig.tight_layout()
        fig.savefig(str(output.map_picture))


rule fit_map_personalised:
    input: "{n_mutations}-{n_patients}/arrays.npz"
    output:
        map_arrays = "{n_mutations}-{n_patients}/pmhn-map-{regularisation}-{regg}.npz",
        map_base_theta = "{n_mutations}-{n_patients}/pmhn-theta-base-{regularisation}-{regg}.pdf",
        map_effects = "{n_mutations}-{n_patients}/pmhn-effects-{regularisation}-{regg}.pdf",
    run:
        regg = float(wildcards.regg)
        regularisation = float(wildcards.regularisation)
        arrays = np.load(str(input))
        
        data = arrays["data"]
        n_patients, n_genes = data.shape
        design_matrix = arrays["design_matrix"]

        n_covs = design_matrix.shape[1]

        # Fit the model MAP
        loglikelihood = pmhn.PersonalisedMHNLoglikelihood(data=data, n_jobs=1)
        
        with pm.Model() as model:
            theta_var = pm.Laplace("theta", mu=0.0, b=regularisation, shape=(n_genes, n_genes))
            effects_var = pm.Laplace("effects", mu=0.0, b=regg, shape=(n_covs, n_genes))
            
            mask = pt.eye(n_genes)

            # compute sum_k covariates[n, k] * effect_size[k, j]
            cov_effect_product = pt.tensordot(design_matrix, effects_var, axes=[[1], [0]])  # shape: (n, j)
            # reshape cov_effect_product from (n, j) to (n, 1, j) to allow broadcasting
            cov_effect_product = cov_effect_product[:, None, :]

            # put it all together
            thetas_all = theta_var + mask * cov_effect_product  # broadcasting happens here

            pm.Potential("loglikelihood", loglikelihood(thetas_all))

        with model:
            solution = pm.find_MAP(maxeval=MAX_OPTIMIZATON_STEPS, seed=42)
            found_theta = solution["theta"]
            found_effects = solution["effects"]
        
        np.savez(
            str(output.map_arrays),
            theta=found_theta,
            effects=found_effects,
        )

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title("Base theta")
        sns.heatmap(found_theta, ax=ax, center=0, cbar=True, square=True, cmap="coolwarm")
        fig.tight_layout()
        fig.savefig(str(output.map_base_theta))

        fig, ax = plt.subplots(figsize=(5, 2))
        sns.heatmap(found_effects, ax=ax, cbar=True, square=False, cmap="coolwarm", center=0, vmax=2, vmin=-2)
        ax.set_title("Effect sizes")
        fig.tight_layout() 
        fig.savefig(str(output.map_effects))
