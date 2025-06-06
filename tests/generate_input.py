import numpy as np
import pandas as pd
from mealpy import FloatVar, DE

def objective_function(solution):
    return np.sum(solution ** 2)

problem_dict = {
    "bounds": FloatVar(lb=(-10.,) * 3, ub=(10.,) * 3, name="delta"),
    "minmax": "min",
    "obj_func": objective_function
}

def generate_run(run_number):
    model = DE.OriginalDE(epoch=100, pop_size=21, wf=0.7, cr=0.9, strategy=0)
    model.solve(problem_dict)
    pop = model.pop
    data = []

    for i in range(len(pop) - 1):
        fitness1 = pop[i].target.fitness
        sol1 = pop[i].solution
        fitness2 = pop[i + 1].target.fitness
        sol2 = pop[i + 1].solution
        data.append([run_number, fitness1, sol1, fitness2, sol2])

    df = pd.DataFrame(data, columns=["Run", "Fitness1", "Solution1", "Fitness2", "Solution2"])
    return df

# Generează rulări multiple și agregă rezultatele
all_runs = []
for i in range(1, 31):
    df = generate_run(i)
    all_runs.append(df)

# Concatenează totul
final_df = pd.concat(all_runs, ignore_index=True)

# === Normalizare vectori ===

# 1. Extrage toate valorile din toate soluțiile
all_values = []
for row in final_df.itertuples(index=False):
    all_values.extend(row.Solution1)
    all_values.extend(row.Solution2)

# 2. Găsește minimul și maximul global
min_val = min(all_values)
max_val = max(all_values)

# 3. Funcție de normalizare pentru vector
def normalize_vector(vec):
    return [
        100 * (x - min_val) / (max_val - min_val) if max_val != min_val else 100.0
        for x in vec
    ]

# 4. Aplică normalizarea pe vectori
final_df["Solution1"] = final_df["Solution1"].apply(normalize_vector)
final_df["Solution2"] = final_df["Solution2"].apply(normalize_vector)

# === (Opțional) Normalizare fitness ===
# Dezactivează dacă nu vrei
# min_fitness = min(final_df[["Fitness1", "Fitness2"]].values.flatten())
# max_fitness = max(final_df[["Fitness1", "Fitness2"]].values.flatten())
# final_df["Fitness1"] = 100 * (final_df["Fitness1"] - min_fitness) / (max_fitness - min_fitness)
# final_df["Fitness2"] = 100 * (final_df["Fitness2"] - min_fitness) / (max_fitness - min_fitness)

# Scrie în fișier
with open("../input/stn_input1.txt", "w") as f:
    f.write("Run\tFitness1\tSolution1\tFitness2\tSolution2\n")
    for _, row in final_df.iterrows():
        run = int(row["Run"])
        fitness1 = f"{row['Fitness1']:.6f}"
        fitness2 = f"{row['Fitness2']:.6f}"
        solution1 = "[" + ",".join(f"{x:.6f}" for x in row["Solution1"]) + "]"
        solution2 = "[" + ",".join(f"{x:.6f}" for x in row["Solution2"]) + "]"
        f.write(f"{run}\t{fitness1}\t{solution1}\t{fitness2}\t{solution2}\n")
