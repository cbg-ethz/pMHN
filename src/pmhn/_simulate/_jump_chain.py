"""The jump Markov chain simulation from an underlying Markov process."""

# States are binary vectors representing genotypes
# For example: (0, 1, 0, ...)
State = tuple[int, ...]


def start_state(n: int) -> State:
    """The state with all 0s."""
    return tuple(0 for _ in range(n))


def end_state(n: int) -> State:
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
