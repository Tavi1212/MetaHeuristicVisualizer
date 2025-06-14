import pandas as pd
import networkx as nx


#Takes an input file and returns a graph containing the unique nodes and their attributes:
#Type:    the node can either indicate the start, end, or intermediary solution of a run
#Fitness: the fitness associated to the solution
#Count:   counts how many times the solution was found in all runs
def create_stn(infile, nr_of_runs, best_solution, objective_type):
    df = pd.read_table(infile, sep=r"\s+")
    df["Run"] = pd.to_numeric(df["Run"], errors="coerce")

    if "Run" not in df.columns:
        df["Run"] = 0

    if nr_of_runs != -1 and nr_of_runs > 0:
        df = df[df["Run"] <= nr_of_runs]

    #Identify where runs start and end
    run_changes = df["Run"].diff() != 0
    start_ids = df.loc[run_changes, "Solution1"].tolist()
    end_ids = df["Solution2"].shift(1).loc[run_changes].tolist()
    end_ids.pop(0)
    end_ids.append(df["Solution2"].iloc[-1])

    #Count node occurrences
    all_nodes = pd.concat([df["Solution1"], df["Solution2"]])
    node_counts = all_nodes.value_counts()

    #Collect fitness values for each solution
    fitness_1 = df[["Solution1", "Fitness1"]].rename(columns={"Solution1": "node", "Fitness1": "fitness"})
    fitness_2 = df[["Solution2", "Fitness2"]].rename(columns={"Solution2": "node", "Fitness2": "fitness"})
    fitness_all = pd.concat([fitness_1, fitness_2])
    fitness_avg = fitness_all.groupby("node").mean()["fitness"].to_dict()

    #Create the list of edges
    edges = df[["Solution1", "Solution2"]].copy()
    edges = edges.rename(columns={"Solution1": "Start", "Solution2": "End"})
    edges["weight"] = edges.groupby(["Start", "End"]).transform("size")
    edges = edges.drop_duplicates(subset=["Start", "End"])
    edges = edges[edges["Start"] != edges["End"]].reset_index(drop=True)

    #Build the graph
    G = nx.DiGraph()
    G.add_edges_from(edges[["Start", "End"]].itertuples(index=False, name=None))

    #Set edge weights
    weight_map = dict(zip(edges[["Start", "End"]].itertuples(index=False, name=None), edges["weight"]))
    nx.set_edge_attributes(G, weight_map, name="weight")

    #Set the node attributes: type, fitness, count
    for node in G.nodes:
        if node in start_ids:
            G.nodes[node]["type"] = "start"
            G.nodes[node]["count"] = node_counts.get(node, 1)
        elif node in end_ids:
            G.nodes[node]["type"] = "end"
            G.nodes[node]["count"] = node_counts.get(node, 1)
        else:
            G.nodes[node]["type"] = "intermediate"
            G.nodes[node]["count"] = node_counts.get(node, 1) / 2

        G.nodes[node]["fitness"] = fitness_avg.get(node, None)

    if best_solution:
        if best_solution in G.nodes:
            G.nodes[best_solution]["type"] = "best"
        else:
            G.add_node(best_solution)
            G.nodes[best_solution]["type"] = "best"
            G.nodes[best_solution]["count"] = 1
            G.nodes[best_solution]["fitness"] = None

    if objective_type == "minimization":
        best_node = min(
            ((n, d["fitness"]) for n, d in G.nodes(data=True) if d.get("fitness") is not None),
            key=lambda x: x[1]
        )[0]
    else:
        best_node = max(
            ((n, d["fitness"]) for n, d in G.nodes(data=True) if d.get("fitness") is not None),
            key=lambda x: x[1]
        )[0]
    G.nodes[best_node]["type"] = "best"

    return G