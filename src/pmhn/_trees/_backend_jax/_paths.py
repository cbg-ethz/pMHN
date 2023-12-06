from typing import NamedTuple

from jaxtyping import Array, Int


class _Paths(NamedTuple):
    """Represents a collection of paths.

    Attrs:
        start: indices representing
          the starting nodes of transitions
        end: indices representing the ending
          nodes of the transitions.
          Use `None` if `start` should only be used.
        path: path representing the rate
          used in the transition

    Note:
        All paths should have the same length.
        We use padding for this.
    """

    start: Int[Array, " K"]
    end: Int[Array, " K"] | None
    path: Int[Array, "K n_events"]

    def size(self) -> int:
        return self.start.shape[0]

    def path_length(self) -> int:
        return self.path.shape[1]
