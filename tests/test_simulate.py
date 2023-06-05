import pytest

from pmhn._simulate._jump_chain import start_state, mutate


@pytest.mark.parametrize("n", [3, 5])
def test_mutate(n: int) -> None:
    state = start_state(n)
    mutated = mutate(state, 0)

    assert type(mutated) == type(state)
    assert len(mutated) == len(state)
    assert sum(mutated) == 1
    assert mutated[0] == 1
