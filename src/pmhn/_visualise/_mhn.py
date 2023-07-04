from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pmhn._theta import decompose_matrix

DEFAULT_COLORMAP: str = "bwr_r"


def plot_theta(
    theta: np.ndarray,
    *,
    ax: plt.Axes,
    gene_names: Optional[Sequence[str]] = None,
    cmap: str = DEFAULT_COLORMAP,
    cbar: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    no_labels: bool = False,
) -> None:
    if gene_names is None:
        gene_names = [f"Gene {i}" for i in range(1, 1 + theta.shape[0])]

    sns.heatmap(
        theta,
        ax=ax,
        cmap=cmap,
        square=True,
        center=0,
        cbar=cbar,
        xticklabels=gene_names,  # type: ignore
        yticklabels=gene_names,  # type: ignore
        vmin=vmin,
        vmax=vmax,
    )

    if no_labels:
        ax.set_xticks([], [])  # type: ignore
        ax.set_yticks([], [])  # type: ignore


def _calculate_offdiagonal_sparsity(theta: np.ndarray, threshold: float = 0.1) -> float:
    _, offdiag = decompose_matrix(theta)

    raveled = offdiag.ravel()

    return float(np.sum(np.abs(raveled) < threshold) / len(raveled))


def plot_offdiagonal_sparsity(
    thetas: np.ndarray,
    *,
    ax: plt.Axes,
    thresholds: Sequence[float] = (0.01, 0.1, 0.2),
    true_theta: Optional[np.ndarray] = None,
    true_theta_color: str = "orangered",
    true_theta_label: str = "Data",
    xlabel: str = "Off-diagonal sparsity",
    ylabel: str = "Count",
) -> None:
    """Plots histogram representing the sparsity of the off-diagonal part of theta.

    Args:
        thetas: Array of theta matrices, shape (n_samples, n_mutations, n_mutations)
        ax: axis to plot on
        thresholds: sparsity threshold
          (distinguishing between "existing" and "non-existing" interactions)
    """
    for threshold in thresholds:
        sparsities = [
            _calculate_offdiagonal_sparsity(theta, threshold=threshold)
            for theta in thetas
        ]
        sns.histplot(sparsities, alpha=0.3, label=f"< {threshold}", kde=True)

    if true_theta is not None:
        true_sparsity = _calculate_offdiagonal_sparsity(true_theta)
        ax.axvline(
            true_sparsity,  # type: ignore
            color=true_theta_color,
            label=true_theta_label,
        )

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.legend()


def plot_offdiagonal_histograms(
    thetas: np.ndarray,
    *,
    ax: plt.Axes,
    theta_true: Optional[np.ndarray] = None,
    alpha: float = 0.1,
) -> None:
    for theta in thetas:
        _, offdiag = decompose_matrix(theta)
        ax.hist(
            offdiag.ravel(), histtype="step", alpha=alpha, color="k", linestyle="solid"
        )

    if theta_true is not None:
        _, offdiag = decompose_matrix(theta_true)
        ax.hist(offdiag.ravel(), histtype="step", alpha=1, color="r", linestyle="solid")
