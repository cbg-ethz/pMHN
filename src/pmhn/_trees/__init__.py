"""This subpackage contains the implementation of TreeMHN.

Luo, X.G., Kuipers, J. & Beerenwinkel, N.
"Joint inference of exclusivity patterns
and recurrent trajectories from tumor mutation trees."
Nat Commun 14, 3676 (2023).
https://doi.org/10.1038/s41467-023-39400-w
"""

# TODO(Pawel): test if this implementation works.


from pmhn._trees._backend import OriginalTreeMHNBackend
from pmhn._trees._interfaces import Tree
from pmhn._trees._simulate import simulate_trees

__all__ = [
    "simulate_trees",
    "Tree",
    "OriginalTreeMHNBackend",
]
