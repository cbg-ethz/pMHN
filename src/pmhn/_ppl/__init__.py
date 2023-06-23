"""Wrappers around MHN likelihood compatible with
probabilistic programming languages (PPLs)."""
from pmhn._ppl._singlemhn import MHNLoglikelihood

__all__ = ["MHNLoglikelihood"]
