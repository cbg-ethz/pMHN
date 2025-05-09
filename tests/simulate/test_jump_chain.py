import numpy as np
import pytest
from pmhn._simulate._jump_chain import generate_start_state, mutate, simulate_dataset


@pytest.mark.parametrize("n", [3, 5])
def test_mutate(n: int) -> None:
    state = generate_start_state(n)
    mutated = mutate(state, 0)

    assert type(mutated) == type(state)
    assert len(mutated) == len(state)
    assert sum(mutated) == 1
    assert mutated[0] == 1


@pytest.mark.parametrize("mutations", [3, 5])
@pytest.mark.parametrize("patients", [10, 20])
def test_genotypes(mutations: int, patients: int):
    rng = np.random.default_rng(42)
    theta = rng.uniform(-1, 1, size=(mutations, mutations))
    times, genotypes = simulate_dataset(
        rng, n_points=patients, theta=theta, mean_sampling_time=1.0
    )
    assert times.shape == (patients,)
    assert genotypes.shape == (patients, mutations)
