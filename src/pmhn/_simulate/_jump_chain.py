"""The jump Markov chain simulation from an underlying Markov process."""
from typing import Optional

import numpy as np


# States are binary vectors representing genotypes
# For example: (0, 1, 0, ...)
State = tuple[int, ...]


def generate_start_state(n: int) -> State:
    """The state with all 0s."""
    return tuple(0 for _ in range(n))


def generate_end_state(n: int) -> State:
    """The state with all 1s."""
    return tuple(1 for _ in range(n))


def mutate(state: State, i: int) -> State:
    """Adds mutation at `i`th position.

    Args:
        state: the state to mutate.
        i: the position to mutate.

    Returns:
        The mutated state, shape the same as `state`.

    Raises:
        ValueError, if gene is already mutated.

    Note:
        This is a pure function, i.e., it does not change the input state,
        but returns a mutated version.
    """
    if state[i] != 0:
        raise ValueError(f"Gene {i} in state {state} can't be mutated.")
    new_state = list(state)
    new_state[i] = 1
    return tuple(new_state)


def jump_to_the_next_state(rng, state: State, theta: np.ndarray) -> tuple[float, State]:
    """Jumps to the next state.

    Args:
        rng: the random number generator.
        state: the current state, should not be the end state
        theta: the log-MHN matrix, theta[i, j] describes the additive log-hazard
          of mutation `j` on the appearance of mutation `i`.

    Returns:
        jump time: elapsed time needed to make the jump.
        mutated state: the state after the jump.

    Raises:
        ValueError, if the state is the end state.
    """
    n_mutations = len(state)
    if state == generate_end_state(n_mutations):
        raise ValueError("Can't jump from the end state.")

    # The transition rates to the state with `i`th mutation turned on
    q = np.zeros(n_mutations, dtype=float)

    for i in range(n_mutations):
        # If the mutation is active, we can't add it
        if state[i] == 1:
            q[i] = 0.0
        # The mutation is inactive!
        # Let's calculate Q = exp(theta[i, i] + sum_j theta[i, j] y_j)
        else:
            log_q = theta[i, i] + np.sum(theta[i, :] * np.asarray(state))
            q[i] = np.exp(log_q)

    leaving_rate = np.sum(q)
    jump_time = rng.exponential(scale=1 / leaving_rate)

    transition_probs = q / leaving_rate

    new_mutation = rng.choice(np.arange(n_mutations), p=transition_probs)

    return jump_time, mutate(state, new_mutation)


def simulate_trajectory(
    rng,
    theta: np.ndarray,
    max_time: float,
    start_state: Optional[State] = None,
) -> list[tuple[float, State]]:
    """Simulates a trajectory of the jump Markov chain.

    Args:
        rng: the random number generator.
        theta: the log-MHN matrix, theta[i, j]
          describes the additive log-hazard of mutation `j`
          onto appearance of mutation `i`
        max_time: the maximum time to simulate.
        start_state: the initial state of the chain.
          By default, it's the state with all 0s.

    Returns:
        A list of (time, state) pairs, where `time` is the time of the jump and `state`
        is the state to which the jump appeared
        We initialize the list with (0, start_state), i.e., the initial state at time 0.
    """
    if start_state is None:
        start_state = generate_start_state(theta.shape[0])

    # We start at the (0, 0, ..., 0) state
    current_state = start_state
    current_time = 0.0

    end = generate_end_state(theta.shape[0])

    history = [(current_time, current_state)]

    while True:
        # Absorbing state, we don't do anything else
        if current_state == end:
            return history

        # There is at least one state which is possible to reach
        delta_t, new_state = jump_to_the_next_state(
            rng,
            state=current_state,
            theta=theta,
        )
        # If the mutation happens after we take the biopsy,
        # we just ignore it
        new_time = current_time + delta_t
        if new_time >= max_time:
            return history
        else:
            history.append((new_time, new_state))

            current_time = new_time
            current_state = new_state
