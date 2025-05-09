import numpy as np
import pmhn._trees as trees
import pytest


def get_all_backends() -> list:
    return [trees.OriginalTreeMHNBackend()]


@pytest.mark.skip("TODO(Pawel): part of https://github.com/cbg-ethz/pMHN/issues/15")
def test_loglikelihood_example_1() -> None:
    """Tests if the loglikelihood is calculated properly
    on a simple example:

        root -> 0 -> 1
    """
    raise NotImplementedError


@pytest.mark.skip("TODO(Pawel): part of https://github.com/cbg-ethz/pMHN/issues/15")
def test_loglikelihood_example_2() -> None:
    """Tests if the loglikelihood is calculated properly
    on a simple example:

      0 <- root -> 1 -> 0
    """
    raise NotImplementedError


@pytest.mark.skip(
    """TODO(Pawel): part of https://github.com/cbg-ethz/pMHN/issues/15
    I'd suggest to sample a tree from TreeMHN,
    evaluate its loglikelihood against some simple matrix
    (e.g., the entries can be 0, +-0.5, and +-1)
    and then manually encode both the matrix, the tree and the loglikelihood here.
    Remember that TreeMHN package uses 'large Theta' 
    and we use 'small theta' convention.
    """
)
def test_loglikelihood_larger_tree_1() -> None:
    """This test compares the loglikelihood of a larger tree (with 5 mutations)
    against the value provided by the TreeMHN package in R:
    https://github.com/cbg-ethz/TreeMHN"""
    raise NotImplementedError


@pytest.mark.skip("TODO(Pawel): part of https://github.com/cbg-ethz/pMHN/issues/15")
@pytest.mark.parametrize("backend", get_all_backends())
@pytest.mark.parametrize("n_mutations", (2, 5))
def test_loglikelihood_empty_tree(backend, n_mutations: int) -> None:
    """Tests if the loglikelihood of a tree without mutations is well-defined."""
    root = trees.Tree("root")
    rng = np.random.default_rng(0)

    theta = rng.normal(size=(n_mutations, n_mutations))

    loglike = backend.loglikelihood(root, theta)
    assert isinstance(loglike, float)


@pytest.mark.skip("TODO(Pawel): part of https://github.com/cbg-ethz/pMHN/issues/18")
def test_gradient_example_1() -> None:
    """Tests if the gradient loglikelihood is calculated properly
    on a simple example:

        root -> 0 -> 1
    """
    raise NotImplementedError


@pytest.mark.skip("TODO(Pawel): part of https://github.com/cbg-ethz/pMHN/issues/18")
def test_gradient_example_2() -> None:
    """Tests if the loglikelihood is calculated properly
    on a simple example:

      0 <- root -> 1 -> 0
    """
    raise NotImplementedError
