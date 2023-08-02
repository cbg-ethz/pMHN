import pytest

from pmhn._trees._interfaces import Tree
from pmhn._trees._validate import validate_tree


def valid_trees() -> list[Tree]:
    """A list of valid trees."""
    raise NotImplementedError


@pytest.mark.skip(
    """TODO(Pawel): this test is skipped 
    until the implementation of the validation is ready.
    Part of https://github.com/cbg-ethz/pMHN/issues/17 
    """
)
@pytest.mark.skip(
    "This test is skipped until the implementation of the validation is ready"
)
@pytest.mark.parametrize("tree", valid_trees())
def test_valid_tree_is_valid(tree: Tree) -> None:
    """Checks if the valid trees are valid."""
    assert validate_tree(tree)


def trees_identical_siblings() -> list[Tree]:
    """A list of trees with identical siblings."""
    raise NotImplementedError


@pytest.mark.skip(
    """TODO(Pawel): this test is skipped 
    until the implementation of the validation is ready.
    Part of https://github.com/cbg-ethz/pMHN/issues/17"""
)
@pytest.mark.parametrize("tree", trees_identical_siblings())
def test_invalid_tree_identical_siblings(tree: Tree) -> None:
    """Checks if the trees with identical siblings are invalid."""
    assert not validate_tree(tree)


def trees_repeated_mutations_in_lineage() -> list[Tree]:
    """A list of trees with repeated mutations in a lineage."""
    raise NotImplementedError


@pytest.mark.skip(
    """TODO(Pawel): this test is skipped 
    until the implementation of the validation is ready
    Part of https://github.com/cbg-ethz/pMHN/issues/17"""
)
@pytest.mark.parametrize("tree", trees_repeated_mutations_in_lineage())
def test_invalid_tree_repeated_mutations_in_lineage(tree: Tree) -> None:
    assert not validate_tree(tree)
