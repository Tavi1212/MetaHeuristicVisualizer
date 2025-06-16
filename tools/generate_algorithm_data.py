import numpy as np
import pandas as pd
from mealpy import FloatVar
from mealpy.swarm_based import PSO
from mealpy.evolutionary_based import DE

def rastrigin(solution):
    A = 10
    n = len(solution)
    return A * n + sum([(x ** 2 - A * np.cos(2 * np.pi * x)) for x in solution])

def sphere(solution):
    return np.sum(np.square(solution))

rastasign_dict = {
    "bounds": FloatVar(lb=(-5.12,) * 3, ub=(5.12,) * 3, name="delta"),
    "minmax": "min",
    "obj_func": rastrigin
}

sphere_dict = {
    "bounds": FloatVar(lb=(-100.0,) * 3, ub=(100.0,) * 3, name="delta"),
    "minmax": "min",
    "obj_func": sphere
}

def generate_run(run_number):
    model1 = PSO.OriginalPSO(epoch=200, pop_size=50, c1=2.05, c2=2.5, w=0.4, constraint_handling="clip")
    model2= DE.OriginalDE(epoch=200, pop_size=50, wf=0.7, cr=0.9, strategy=0, constraint_handling="clip")

    model2.solve(sphere_dict)

    best_solutions = model2.history.list_global_best
    data = []
    for i in range(len(best_solutions)-1):
        fitness1 = best_solutions[i].target.fitness
        sol1 = best_solutions[i].solution
        fitness2 = best_solutions[i + 1].target.fitness
        sol2 = best_solutions[i + 1].solution
        data.append([run_number, fitness1, sol1, fitness2, sol2])
    return pd.DataFrame(data, columns=["Run", "Fitness1", "Solution1", "Fitness2", "Solution2"])

# === Run + concat ===
all_runs = [generate_run(i) for i in range(1, 51)]
final_df = pd.concat(all_runs, ignore_index=True)

# === Write file ===
with open("../input/DE_sphere.txt", "w") as f:
    f.write("Run\tFitness1\tSolution1\tFitness2\tSolution2\n")
    for _, row in final_df.iterrows():
        fitness1 = f"{row['Fitness1']:.16e}"
        fitness2 = f"{row['Fitness2']:.16e}"
        solution1 = "[" + ",".join(str(x) for x in row["Solution1"]) + "]"
        solution2 = "[" + ",".join(str(x) for x in row["Solution2"]) + "]"
        f.write(f"{int(row['Run'])}\t{fitness1}\t{solution1}\t{fitness2}\t{solution2}\n")

# === Write raw solutions with fitness and run to a separate file ===
with open("../input/DE_sphere_solutions.txt", "w") as f:

    f.write("Run\tFitness\tSolution\n")
    for _, row in final_df.iterrows():
        f.write(f"{int(row['Run'])}\t{row['Fitness1']:.16e}\t[{','.join(str(x) for x in row['Solution1'])}]\n")
    # Add the last solution from Fitness2
    last_row = final_df.iloc[-1]
    f.write(
        f"{int(last_row['Run'])}\t{last_row['Fitness2']:.16e}\t[{','.join(str(x) for x in last_row['Solution2'])}]\n")