"""Wrappers around MHN likelihood compatible with
probabilistic programming languages (PPLs)."""
from pmhn._ppl._singlemhn import MHNLoglikelihood
from pmhn._ppl._multiplemhn import PersonalisedMHNLoglikelihood

__all__ = ["MHNLoglikelihood", "PersonalisedMHNLoglikelihood"]
