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

    n_patients, _ = genotypes.shape
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


def plot_genotype_samples(
    genotype_samples: np.ndarray,
) -> tuple[plt.Figure, np.ndarray]:
    fig, axs = plt.subplots(
        len(genotype_samples), 1, figsize=(15, 2 * len(genotype_samples))
    )

    for i, (genotypes, ax) in enumerate(zip(genotype_samples, axs.ravel())):
        plot_genotypes(genotypes, ax=ax, patients_on_x_axis=True, sort=True)

    fig.tight_layout()  # type: ignore
    return fig, axs  # type: ignore
