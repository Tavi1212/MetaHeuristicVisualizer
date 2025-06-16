import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Incarca fisierul PSO solutii
df = pd.read_csv("../input/PSO_sphere_solutions.txt", sep="\t")

# Extrage coloanele relevante
fitness_vals = df["Fitness"]

# Curbe PSO convergență pentru fiecare run
# Presupunem 50 PSO runuri, fiecare cu 100 PSO epoci => 5000 PSO linii
runs = df["Run"].unique()
nr_epoci = df["Run"].value_counts().min()  # presupunem 100 epoci per run

plt.figure(figsize=(10, 6))
for run in runs:
    fitness_run = df[df["Run"] == run]["Fitness"].values
    plt.plot(range(1, len(fitness_run) + 1), fitness_run, alpha=0.4)

plt.title("Curbele PSO convergență pentru 50 PSO rulări PSO")
plt.xlabel("Epoca")
plt.ylabel("Fitness")
plt.tight_layout()
plt.savefig("output/PSO_sphere_convergence.png")


# Încarcă fișierul PSO soluții
df = pd.read_csv("../input/PSO_sphere_solutions.txt", sep="\t")

# Calculează distanțele euclidiene între soluții consecutive per run
runs = df["Run"].unique()
exploration_values = []
max_epochs = df["Run"].value_counts().min() - 1  # dacă ai 100 linii pe run => 99 distanțe

for run in runs:
    solutii = df[df["Run"] == run]["Solution"].apply(eval).tolist()
    distante = [
        np.linalg.norm(np.array(solutii[i+1]) - np.array(solutii[i]))
        for i in range(len(solutii) - 1)
    ]
    # Taie la lungimea minimă comună
    exploration_values.append(distante[:max_epochs])

# Convertește într-un array 2D uniform
exploration_array = np.array(exploration_values)  # shape: (50, num_epoci)

# Media pe fiecare epocă
exploration_mean = np.mean(exploration_array, axis=0)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(exploration_mean) + 1), exploration_mean)
plt.title("Exploration vs Epoch (mean eucliPSOan distance)")
plt.xlabel("Epoch")
plt.ylabel("Mean distance between consecutive solutions")
plt.tight_layout()
plt.savefig("output/PSO_sphere_exploration.png")
plt.close()
