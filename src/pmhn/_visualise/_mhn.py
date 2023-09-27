from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pmhn._theta import decompose_matrix

DEFAULT_COLORMAP: str = "bwr_r"


def plot_theta(
    theta: np.ndarray,
    *,
    ax: plt.Axes,  # type: ignore
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
    ax: plt.Axes,  # type: ignore
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
    ax: plt.Axes,  # type: ignore
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


def plot_theta_samples(
    theta_samples: np.ndarray,
    *,
    width: int = 4,
    height: int = 3,
    theta_true: Optional[np.ndarray] = None,
) -> tuple[plt.Figure, np.ndarray]:  # type: ignore
    """Plot samples from theta."""
    if len(theta_samples) < width * height:
        raise NotImplementedError("We need more samples to plot")

    fig, axs = plt.subplots(height, width, figsize=(width * 3, height * 3))

    n_samples_to_plot = width * height
    vmin = np.min(theta_samples[:n_samples_to_plot])
    vmax = np.max(theta_samples[:n_samples_to_plot])

    if theta_true is not None:
        n_samples_to_plot -= 1
        vmin = min(vmin, np.min(theta_true))
        vmax = min(vmax, np.max(theta_true))

    for i in range(n_samples_to_plot):
        plot_theta(
            theta_samples[i],
            ax=axs.ravel()[i],
            no_labels=True,
            cbar=True,
            vmin=vmin,
            vmax=vmax,
        )

    if theta_true is not None:
        ax = axs.ravel()[-1]
        plot_theta(theta_true, ax=ax, no_labels=True, cbar=True, vmin=vmin, vmax=vmax)
        ax.set_title("True matrix")

    fig.tight_layout()  # type: ignore
    return fig, axs  # type: ignore
