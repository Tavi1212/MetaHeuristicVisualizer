
import numpy as np
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scripts.structures import ConfigData

from scripts.utils import (
    parse_continuous_solution,
    parse_discrete_solutions,
    get_valid_distance_fn,
    compute_distance
)

def agglomerative_clustering_with_volume_constraints(G, config : ConfigData) -> list[list[str]]:
    sample_node = next(iter(G.nodes), None)
    if not sample_node:
        print("❌ Empty graph — no nodes.")
        return []

    try:
        parse_continuous_solution(sample_node)
    except ValueError:
        print("❌ Could not parse any node vector.")
        return []

    distance_type = config.cDistance
    try:
        distance_fn = get_valid_distance_fn(distance_type, "real")
    except ValueError:
        print(f"❌ Invalid distance type: {distance_type}")
        return []

    cluster_size_percent = config.cClusterSize
    volume_size_percent = config.cVolumeSize

    node_vectors = {}
    node_fitness = {}
    for node in G.nodes:
        try:
            vec = parse_continuous_solution(node)
            node_vectors[node] = vec
            node_fitness[node] = G.nodes[node]["fitness"]
        except (ValueError, KeyError):
            print(f"⚠️ Skipping node: {node}")
            continue

    if len(node_vectors) < 2:
        print("❌ Not enough valid nodes to cluster.")
        return []

    # Normalize fitness values to [0, 1]
    if not node_fitness:
        print("❌ No valid fitness values.")
        return []

    global_min_fitness = min(node_fitness.values())
    global_max_fitness = max(node_fitness.values())
    global_fitness_range = global_max_fitness - global_min_fitness

    if global_fitness_range == 0:
        print("⚠️ All nodes have identical fitness — using 0.0 for normalization.")
        for node in node_fitness:
            node_fitness[node] = 0.0
    else:
        for node in node_fitness:
            node_fitness[node] = (
                node_fitness[node] - global_min_fitness
            ) / global_fitness_range

    ordered_nodes = list(node_vectors.keys())
    vectors = [node_vectors[n] for n in ordered_nodes]

    def condensed_distance_matrix(vectors, fn):
        n = len(vectors)
        dist_list = []
        for i in range(n):
            for j in range(i + 1, n):
                d = compute_distance(vectors[i], vectors[j], fn)
                dist_list.append(d)
        return np.array(dist_list)

    condensed = condensed_distance_matrix(vectors, distance_fn)
    Z = linkage(condensed, method="average")
    threshold = float(Z[-1, 2]) + 1
    cluster_labels = fcluster(Z, t=threshold, criterion="distance")

    clusters_by_label = defaultdict(list)
    for node, label in zip(ordered_nodes, cluster_labels):
        clusters_by_label[label].append(node)

    print(f"\n🧩 Raw clusters formed: {len(clusters_by_label)}")

    total_nodes = len(node_vectors)
    max_cluster_size = (cluster_size_percent / 100) * total_nodes
    max_cluster_volume = volume_size_percent / 100  # Percent of normalized [0–1] range

    print(f"📏 Max allowed cluster size: {max_cluster_size}")
    print(f"📏 Max allowed cluster volume (normalized): {max_cluster_volume:.4f}")

    final_clusters = []

    for label, cluster_nodes in clusters_by_label.items():
        if not cluster_nodes:
            continue

        cluster_fitnesses = [node_fitness[n] for n in cluster_nodes]
        cluster_volume = max(cluster_fitnesses) - min(cluster_fitnesses)

        print(f"🔍 Cluster {label}: size={len(cluster_nodes)}, volume={cluster_volume:.6f}")

        if cluster_volume > max_cluster_volume:
            # Bin by fitness range segments
            bin_width = max_cluster_volume
            bins = defaultdict(list)
            for node in cluster_nodes:
                fitness = node_fitness[node]
                bin_index = int(fitness / bin_width)
                bins[bin_index].append(node)

            for bin_nodes in bins.values():
                # Apply size limit within each bin
                for i in range(0, len(bin_nodes), int(max_cluster_size)):
                    final_clusters.append(bin_nodes[i:i + int(max_cluster_size)])
        else:
            # Volume is OK, still enforce max size
            for i in range(0, len(cluster_nodes), int(max_cluster_size)):
                final_clusters.append(cluster_nodes[i:i + int(max_cluster_size)])

    print(f"\n✅ Final accepted clusters: {len(final_clusters)}")
    return final_clusters



def agglomerative_clustering_discrete(G, config : ConfigData) -> list[list[str]]:
    sample_node = next(iter(G.nodes), None)
    if not sample_node:
        print("❌ Empty graph — no nodes.")
        return []

    try:
        parse_discrete_solutions(sample_node)
    except ValueError:
        print("❌ Could not parse any node as discrete vector.")
        return []

    distance_type = config.dDistance
    try:
        distance_fn = get_valid_distance_fn(distance_type, "discrete")
    except ValueError:
        print(f"❌ Invalid distance type for discrete space: {distance_type}")
        return []

    cluster_size_percent = config.dCSize
    volume_size_percent = config.dVSize

    node_vectors = {}
    node_fitness = {}
    for node in G.nodes:
        try:
            vec = parse_discrete_solutions(node)[0]  # flatten
            node_vectors[node] = vec
            node_fitness[node] = G.nodes[node]["fitness"]
        except (ValueError, KeyError):
            print(f"⚠️ Skipping node: {node}")
            continue

    if len(node_vectors) < 2:
        print("❌ Not enough valid nodes to cluster.")
        return []

    # Normalize fitness to [0, 1]
    global_min_fitness = min(node_fitness.values())
    global_max_fitness = max(node_fitness.values())
    global_fitness_range = global_max_fitness - global_min_fitness

    if global_fitness_range == 0:
        print("⚠️ All nodes have identical fitness — assigning 0.0 to all.")
        for node in node_fitness:
            node_fitness[node] = 0.0
    else:
        for node in node_fitness:
            node_fitness[node] = (
                node_fitness[node] - global_min_fitness
            ) / global_fitness_range

    ordered_nodes = list(node_vectors.keys())
    vectors = [node_vectors[n] for n in ordered_nodes]

    def condensed_distance_matrix(vectors, fn):
        n = len(vectors)
        dist_list = []
        for i in range(n):
            for j in range(i + 1, n):
                d = compute_distance(vectors[i], vectors[j], fn)
                dist_list.append(d)
        return np.array(dist_list)

    condensed = condensed_distance_matrix(vectors, distance_fn)
    Z = linkage(condensed, method="average")
    threshold = float(Z[-1, 2]) + 1
    cluster_labels = fcluster(Z, t=threshold, criterion="distance")

    clusters_by_label = defaultdict(list)
    for node, label in zip(ordered_nodes, cluster_labels):
        clusters_by_label[label].append(node)

    print(f"\n🧩 Raw clusters formed: {len(clusters_by_label)}")

    total_nodes = len(node_vectors)
    max_cluster_size = (cluster_size_percent / 100) * total_nodes
    max_cluster_volume = volume_size_percent / 100  # normalized fitness

    print(f"📏 Max allowed cluster size: {max_cluster_size}")
    print(f"📏 Max allowed cluster volume (normalized): {max_cluster_volume:.4f}")

    final_clusters = []

    for label, cluster_nodes in clusters_by_label.items():
        if not cluster_nodes:
            continue

        cluster_fitnesses = [node_fitness[n] for n in cluster_nodes]
        cluster_volume = max(cluster_fitnesses) - min(cluster_fitnesses)

        print(f"🔍 Cluster {label}: size={len(cluster_nodes)}, volume={cluster_volume:.6f}")

        if cluster_volume > max_cluster_volume:
            bin_width = max_cluster_volume
            bins = defaultdict(list)
            for node in cluster_nodes:
                fitness = node_fitness[node]
                bin_index = int(fitness / bin_width)
                bins[bin_index].append(node)

            for bin_nodes in bins.values():
                for i in range(0, len(bin_nodes), int(max_cluster_size)):
                    final_clusters.append(bin_nodes[i:i + int(max_cluster_size)])
        else:
            for i in range(0, len(cluster_nodes), int(max_cluster_size)):
                final_clusters.append(cluster_nodes[i:i + int(max_cluster_size)])

    print(f"\n✅ Final accepted clusters: {len(final_clusters)}")
    return final_clusters


