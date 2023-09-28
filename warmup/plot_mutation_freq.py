import pandas as pd
import matplotlib.pyplot as plt

r_trees_path ='/home/laukeller/BSc Thesis/TreeMHN/Example/trees_10000.csv'
python_trees_path ='/home/laukeller/BSc Thesis/pMHN/src/pmhn/_trees/trees_10000.csv'

r_trees = pd.read_csv(r_trees_path)
python_trees = pd.read_csv(python_trees_path)

r_mutation_frequencies = r_trees['Mutation_ID'].value_counts().sort_index()
python_mutation_frequencies = python_trees['Mutation_ID'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.35

r_mutation_frequencies.plot(kind='bar', width=bar_width, position=0, align='center', color='b', alpha=0.5, label='R Trees', ax=ax)
python_mutation_frequencies.plot(kind='bar', width=bar_width, position=1, align='center', color='r', alpha=0.5, label='Python Trees', ax=ax)

ax.set_xlabel('Mutation ID')
ax.set_ylabel('Frequency')
ax.set_title('Mutation Frequencies Comparison')
ax.legend()

plt.tight_layout()
plt.show()

