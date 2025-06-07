from typing import final

from pyvis.network import Network
import matplotlib.pyplot as plt
from scripts.structures import ConfigData, AdvancedSettings
from scripts.clustering import agglomerative_clustering_with_volume_constraints
from scripts.create import create_stn
from matplotlib import colormaps
import webbrowser
import itertools
import random


def visualize_clusters(final_clusters, output_path="clusters.html", min_cluster_size=50):
    net = Network(height="800px", width="100%", notebook=False)

    # Normalize cluster sizes to [20, 80]
    raw_sizes = [len(c) for c in final_clusters]
    min_raw, max_raw = min(raw_sizes), max(raw_sizes)

    def normalize_size(raw):
        if max_raw == min_raw:
            return 50
        return 20 + (raw - min_raw) / (max_raw - min_raw) * 60  # maps to [20, 80]

    # Add one node per cluster
    cluster_nodes = []
    for i, cluster in enumerate(final_clusters):
        size = normalize_size(len(cluster))
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        label = f"Cluster {i + 1}\nSize: {len(cluster)}"
        net.add_node(i, label=label, size=size, color=color)
        cluster_nodes.append((i, set(cluster)))  # (cluster_id, node set)

    # Optional: add edges between clusters if original graph is provided
    if G:
        for i, (id_a, nodes_a) in enumerate(cluster_nodes):
            for j, (id_b, nodes_b) in enumerate(cluster_nodes):
                if i >= j:
                    continue
                if any(G.has_edge(u, v) for u in nodes_a for v in nodes_b):
                    net.add_edge(id_a, id_b)

    net.show(output_path, notebook=False)

input_path = "../input/stn_input2.txt"
output_path = "graph_testing.html"

config = ConfigData(
    problemType="continuous",
    objectiveType="minimization",
    partitionStrategy="clustering",
    dPartitioning=0,
    dCSize=0,
    dVSize=0,
    dDistance="",
    cMinBound=0,
    cMaxBound=0,
    cDimension=0,
    cHypercube=0,
    cClusterSize=50,
    cVolumeSize=25,
    cDistance="euclidean"
)

advanced = AdvancedSettings(
    best_solution="",
    nr_of_runs=-1,
    vertex_size=20,
    arrow_size=1,
    tree_layout=False
)

G = create_stn(input_path)

final_clusters = agglomerative_clustering_with_volume_constraints(G, config)
print(final_clusters)
visualize_clusters(final_clusters)



