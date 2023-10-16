import pandas as pd
import matplotlib.pyplot as plt

paths = {
    500: (
        "/home/laukeller/BSc Thesis/TreeMHN/Example/trees_500.csv",
        "/home/laukeller/BSc Thesis/pMHN/warmup/trees_500.csv",
    ),
    5000: (
        "/home/laukeller/BSc Thesis/TreeMHN/Example/trees_5000.csv",
        "/home/laukeller/BSc Thesis/pMHN/warmup/trees_5000.csv",
    ),
    10000: (
        "/home/laukeller/BSc Thesis/TreeMHN/Example/trees_10000.csv",
        "/home/laukeller/BSc Thesis/pMHN/warmup/trees_10000.csv",
    ),
    50000: (
        "/home/laukeller/BSc Thesis/TreeMHN/Example/trees_50000.csv",
        "/home/laukeller/BSc Thesis/pMHN/warmup/trees_50000.csv",
    ),
}

fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 grid of subplots
axs = axs.ravel()

for i, (num_trees, (r_path, python_path)) in enumerate(paths.items()):
    r_trees = pd.read_csv(r_path)
    python_trees = pd.read_csv(python_path)

    r_mutation_frequencies = r_trees["Mutation_ID"].value_counts().sort_index()
    python_mutation_frequencies = (
        python_trees["Mutation_ID"].value_counts().sort_index()
    )

    bar_width = 0.35

    r_mutation_frequencies.plot(
        kind="bar",
        width=bar_width,
        position=0,
        align="center",
        color="b",
        alpha=0.5,
        label="R Trees",
        ax=axs[i],
    )
    python_mutation_frequencies.plot(
        kind="bar",
        width=bar_width,
        position=1,
        align="center",
        color="r",
        alpha=0.5,
        label="Python Trees",
        ax=axs[i],
    )

    axs[i].set_xlabel("Mutation ID")
    axs[i].set_ylabel("Frequency")
    axs[i].legend(loc="upper right")
    axs[i].set_title(f"Mutation Frequency Distribution ({num_trees} trees)")

plt.tight_layout()
plt.show()
