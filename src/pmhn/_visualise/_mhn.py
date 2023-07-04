from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DEFAULT_COLORMAP: str = "bwr_r"


def plot_theta(
    theta: np.ndarray,
    *,
    ax: plt.Axes,
    gene_names: Optional[Sequence[str]] = None,
    cmap: str = DEFAULT_COLORMAP,
    cbar: bool = True,
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
    )
