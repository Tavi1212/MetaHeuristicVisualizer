import pandas as pd

def analizeaza_fisier(path):
    df = pd.read_csv(path, sep="\t")
    fitness = df["Fitness"]

    fitness_mediu = fitness.mean()
    deviatia_std = fitness.std()

    print(f"Fitness mediu: {fitness_mediu:.2f}")
    print(f"Deviație standard: {deviatia_std:.2f}")

    optimal_fitness = 0.0
    epsilon = 1e-6  # toleranță pentru comparație numerică
    nr_total_rulari = df["Run"].nunique()


# Exemplu de utilizare
analizeaza_fisier("../input/DE_rastrigin_solutions.txt")


