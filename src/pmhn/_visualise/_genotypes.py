import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_genotypes(
    genotypes: np.ndarray,
    *,
    ax: plt.Axes,
    patients_on_x_axis: bool = True,
    patients_label: str = "Patients",
    genes_label: str = "Genes",
    sort: bool = True,
) -> None:
    if not patients_on_x_axis:
        raise NotImplementedError("Only patients on x axis is supported")

    n_patients, n_genes = genotypes.shape
    if sort:
        index = sorted(np.arange(n_patients), key=lambda i: -np.sum(genotypes[i, :]))
    else:
        index = np.arange(n_patients)

    sns.heatmap(
        genotypes[index, :].T,
        ax=ax,
        cmap="Greys",
        square=True,
        vmin=0,
        vmax=1,
        cbar=False,
    )

    # Silent pyright false positive
    ax.set_xticks([], [])  # type: ignore
    ax.set_yticks([], [])  # type: ignore
    ax.set_ylabel(genes_label)
    ax.set_xlabel(patients_label)
