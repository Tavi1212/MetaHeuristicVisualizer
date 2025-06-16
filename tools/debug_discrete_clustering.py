from typing import final
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
from scripts.structures import ConfigData, AdvancedOptions
from scripts.partition import discrete_clustering
from scripts.create import create_stn
from matplotlib import colormaps
import webbrowser
import itertools
import random


def visualize_clusters(final_clusters, G, output_path="clusters.html"):
    from pyvis.network import Network
    import matplotlib.pyplot as plt
    import random
    import networkx as nx

    net = Network(height="800px", width="100%", notebook=False)

    # Normalize cluster sizes to [20, 80]
    raw_sizes = [len(c) for c in final_clusters]
    min_raw, max_raw = min(raw_sizes), max(raw_sizes)

    def normalize_size(raw):
        if max_raw == min_raw:
            return 50
        return 20 + (raw - min_raw) / (max_raw - min_raw) * 60  # maps to [20, 80]

    total_clusters = len(final_clusters)
    total_nodes = len(G.nodes)

    # Global fitness range
    fitness_dict = nx.get_node_attributes(G, "fitness")
    global_min_fitness = min(fitness_dict.values())
    global_max_fitness = max(fitness_dict.values())
    global_range = global_max_fitness - global_min_fitness or 1e-9

    cluster_nodes = []
    for i, cluster in enumerate(final_clusters):
        size = normalize_size(len(cluster))

        cmap = plt.colormaps['tab20']
        color = cmap(i / total_clusters)
        hex_color = '#%02x%02x%02x' % tuple(int(255 * x) for x in color[:3])

        fitness_values = [G.nodes[n]["fitness"] for n in cluster if "fitness" in G.nodes[n]]
        min_fitness = min(fitness_values)
        max_fitness = max(fitness_values)
        volume = (max_fitness - min_fitness) / global_range * 100
        percentage = (len(cluster) / total_nodes) * 100

        label = f"Cluster {i + 1}"
        title = (
            f"Cluster {i + 1}\n"
            f"Size: {len(cluster)} ({percentage:.2f}%)\n"
            f"Min Fitness: {min_fitness:.4g}\n"
            f"Max Fitness: {max_fitness:.4g}\n"
            f"Volume: {volume:.2f}%"
        )

        net.add_node(
            i,
            label=label,
            size=size,
            color=hex_color,
            title=title
        )
        cluster_nodes.append((i, set(cluster)))

    # Draw edges if any connection between cluster members
    for i, (id_a, nodes_a) in enumerate(cluster_nodes):
        for j, (id_b, nodes_b) in enumerate(cluster_nodes):
            if i >= j:
                continue
            if any(G.has_edge(u, v) or G.has_edge(v, u) for u in nodes_a for v in nodes_b):
                net.add_edge(id_a, id_b)

    net.show(output_path, notebook=False)

input_path = "../input/BRKGA_pmed7.txt"
output_path = "graph_testing.html"

config = ConfigData(
    problemType="discrete",
    objectiveType="minimization",
    partitionStrategy="clustering",
    dPartitioning=0,
    dCSize=50,
    dVSize=25,
    dDistance="hamming",
    cMinBound=0,
    cMaxBound=0,
    cDimension=0,
    cHypercube=0,
    cClusterSize=50,
    cVolumeSize=25,
    cDistance="euclidean"
)

advanced = AdvancedOptions(
    best_solution="",
    nr_of_runs=-1,
    vertex_size=20,
    arrow_size=1,
    tree_layout=False
)

G = create_stn(input_path)

final_clusters = discrete_clustering(G, config)
print(final_clusters)
visualize_clusters(final_clusters, G)



