import time

import numpy as np
import pymc as pm

import pmhn

pmhn.control_no_mutation_warning(silence=True)

n_mutations: int = 9
n_patients: int = 200

rng = np.random.default_rng(12)

theta_raw = rng.normal(size=(n_mutations, n_mutations))
mask = rng.binomial(1, 0.1, size=theta_raw.shape)
theta = theta_raw * mask

_, mutations = pmhn.simulate_dataset(
    rng=rng,
    n_points=n_patients,
    theta=theta,
    mean_sampling_time=1.0,
)


loglikelihood = pmhn.MHNLoglikelihood(
    data=mutations,
    backend=pmhn.MHNCythonBackend(),
)

with pm.Model() as model:  # type: ignore
    theta_var = pm.Cauchy("theta", alpha=0.0, beta=0.1, shape=theta.shape)
    pm.Potential("loglikelihood", loglikelihood(theta_var))


t0 = time.time()
n_tune = 200
n_samples = 200
n_chains = 4

print("Sampling...")


with model:
    idata = pm.sample(chains=n_chains, random_seed=rng, tune=n_tune, draws=n_samples)

idata.to_netcdf("idata.nc")

print(idata)


t1 = time.time()

print(f"Sampling {n_chains * (n_tune + n_samples)} took {t1 - t0:.2f} seconds")
