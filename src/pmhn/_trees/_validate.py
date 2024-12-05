"""Tree validation utilities."""

from pmhn._trees._interfaces import Tree


def _validate_repeated_mutations_in_lineage(tree: Tree) -> bool:
    """Validates that there are no repeated mutations in a lineage:

        A -> X -> C -> D -> X -> ...

    Args:
        tree: the tree to validate.

    Returns:
        True if the tree is valid, False otherwise.
    """
    # TODO(Pawel): part of https://github.com/cbg-ethz/pMHN/issues/17
    raise NotImplementedError


def _validate_identical_siblings(tree: Tree) -> bool:
    """Validates if there are no identical siblings:

        X <- A -> X -> B

    Args:
        tree: the tree to validate.

    Returns:
        True if the tree is valid, False otherwise.
    """
    # TODO(Pawel): part of https://github.com/cbg-ethz/pMHN/issues/17
    raise NotImplementedError


def validate_tree(tree: Tree) -> bool:
    """Validates a tree.

    Args:
        tree: the tree to validate.

    Returns:
        True if the tree is valid, False otherwise.
    """
    return _validate_identical_siblings(
        tree
    ) and _validate_repeated_mutations_in_lineage(tree)
