import math
import numpy as np
from itertools import combinations
import networkx as nx
from scripts.structures import ConfigData
import scripts.utils as utils
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform


def discrete_shannon_entropy(G, nr):
    start_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "start"]
    end_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "end"]
    if not start_nodes or not end_nodes:
        raise ValueError("Start and end nodes are required.")

    start, end = start_nodes[0], end_nodes[0]

    try:
        critical_path = set(nx.shortest_path(G, source=start, target=end))
    except nx.NetworkXNoPath:
        critical_path = set()

    entropies = []
    for node_id, _ in G.nodes(data=True):
        if node_id in critical_path:
            continue
        entropy = utils.shannon_entropy_string(node_id)
        entropies.append((node_id, entropy))

    node_count = len(entropies)
    del_count = int((node_count * nr) / 100)

    entropies.sort(key=lambda x: x[1])
    nodes_to_remove = [node_id for node_id, _ in entropies[:del_count]]

    for node_id in nodes_to_remove:
        if node_id in G and G.nodes[node_id].get("type") == "intermediate":
            utils.remove_node_between_two(G, node_id)


def cont_standard_partitioning(G, n_bins=5, min_bound=0.0, max_bound=1.0):
    if n_bins < 0:
        n_bins = 10 ** abs(n_bins)

    cube_size = (max_bound - min_bound) / n_bins
    cube_map = {}
    supernode_counts = defaultdict(int)
    supernode_types = defaultdict(set)  # Track original types

    for node in list(G.nodes):
        vectors = utils.parse_vectors_string(node)
        flat = vectors[0]

        try:
            bin_indices = tuple(
                min(int((float(x) - min_bound) / cube_size), n_bins - 1)
                for x in flat
            )
        except ValueError as e:
            print(f"[WARNING] Skipping node {node} due to invalid coordinate: {e}")
            continue

        cube_id = str(bin_indices)
        cube_map[node] = cube_id
        supernode_counts[cube_id] += G.nodes[node].get("count", 1)
        supernode_types[cube_id].add(G.nodes[node].get("type", "intermediate"))

    H = nx.DiGraph()

    for cube_id in supernode_counts:
        # Determine supernode type
        types = supernode_types[cube_id]
        if "start" in types:
            node_type = "start"
        elif "end" in types:
            node_type = "end"
        else:
            node_type = "intermediate"

        H.add_node(
            cube_id,
            count=supernode_counts[cube_id],
            type=node_type,
            shape="circle",  # optional override
            color="gray"
        )

    for u, v in G.edges:
        if u not in cube_map or v not in cube_map:
            continue
        cu, cv = cube_map[u], cube_map[v]
        if cu == cv:
            continue
        if H.has_edge(cu, cv):
            H[cu][cv]["weight"] += G[u][v].get("weight", 1)
        else:
            H.add_edge(cu, cv, weight=G[u][v].get("weight", 1), color=G[u][v].get("color", "black"))

    return H


def normalize_node_ids(G):
    max_length = max(len(node) for node in G.nodes)
    mapping = {node: node.rjust(max_length, '0') for node in G.nodes}
    return nx.relabel_nodes(G, mapping)

def get_distance_fn(name):
    if name == "hamming":
        return utils.hamming_distance
    elif name == "euclidean":
        return utils.euclidean_distance
    elif name == "manhattan":
        return utils.manhattan_distance
    else:
        raise ValueError(f"Unknown distance metric: {name}")


def trim_cluster_by_fitness(cluster_nodes, node_vectors, G, max_volume_pct, domain_size=2):
    entries = list(zip(cluster_nodes, node_vectors))
    # Sort by fitness, lowest first
    entries.sort(key=lambda x: G.nodes[x[0]].get("fitness", float("inf")))

    while True:
        volume = estimate_volume_percent([v for _, v in entries], domain_size)
        if volume <= max_volume_pct or len(entries) <= 1:
            break
        entries.pop(0)  # Remove the worst (lowest fitness)

    return {node for node, _ in entries}  # Keep only surviving node IDs


def estimate_volume_percent(cluster_vectors, domain_size=2):
    dimensions = list(zip(*cluster_vectors))  # transpose
    per_dim_variation = [len(set(col)) for col in dimensions]
    cluster_volume = 1
    for v in per_dim_variation:
        cluster_volume *= v
    total_volume = domain_size ** len(dimensions)
    return (cluster_volume / total_volume) * 100


def apply_partitioning_from_config(G, config):
    if config.problemType == "discrete":
        if config.partitionStrategy == "shannon":
            discrete_shannon_entropy(G, nr=config.dPartitioning)
            return G
        elif config.partitionStrategy == "clustering":
            G = constrained_cluster(G, config)
            return G

    elif config.problemType == "continuous":
        if config.partitionStrategy == "partitioning":
            # Ensure n_bins is always positive and non-zero
            if config.cHypercube == 0:
                n_bins = 10  # default fallback
            elif config.cHypercube < 0:
                n_bins = 10 ** abs(config.cHypercube)
            else:
                n_bins = config.cHypercube

            return cont_standard_partitioning(
                G,
                n_bins=n_bins,
                min_bound=config.dMinBound,
                max_bound=config.cMaxBound
            )

        elif config.partitionStrategy == "clustering":
            return constrained_cluster(G, config)

    raise ValueError(f"Unsupported configuration: {config.problemType}, {config.partitionStrategy}")


def constrained_cluster(G, config: ConfigData):
    max_cluster_size = config.dCSize
    max_volume       = config.dVSize
    distance_metric  = config.dDistance

    intermediates = [n for n in G.nodes if G.nodes[n].get("type") == "intermediate"]
    if not intermediates:
        return G

    node_vectors = {n: utils.parse_vectors_string(n, mode='auto')[0] for n in intermediates}
    nodes = list(node_vectors.keys())
    vectors = np.array([node_vectors[n] for n in nodes])

    dist_fn = get_distance_fn(distance_metric)
    dist_matrix = squareform(pdist(vectors, lambda u, v: dist_fn(u, v)))

    clusters = {i: [i] for i in range(len(nodes))}
    active = set(clusters.keys())
    idx_map = {i: nodes[i] for i in range(len(nodes))}

    def max_internal_distance(indices):
        return max(dist_matrix[i][j] for i, j in combinations(indices, 2)) if len(indices) > 1 else 0

    while True:
        merge_candidates = []
        active_list = list(active)

        for i in range(len(active_list)):
            for j in range(i + 1, len(active_list)):
                ci, cj = active_list[i], active_list[j]
                merged = clusters[ci] + clusters[cj]
                if len(merged) > max_cluster_size:
                    continue
                if max_internal_distance(merged) > max_volume:
                    continue
                merge_candidates.append((ci, cj))

        if not merge_candidates:
            break

        ci, cj = merge_candidates[0]
        clusters[ci] += clusters[cj]
        del clusters[cj]
        active.remove(cj)

    # Assign cluster labels to nodes
    for cluster_id, members in enumerate(clusters.values()):
        for idx in members:
            node = idx_map[idx]
            G.nodes[node]["cluster"] = cluster_id

    return G

def collapse_to_supernodes(G, color_attr=None):
    H = nx.DiGraph()
    cluster_map = defaultdict(list)

    # Step 1: Collect intermediate nodes into clusters
    for node in G.nodes:
        if G.nodes[node].get("type") != "intermediate":
            continue  # Skip start/end nodes
        cid = G.nodes[node].get("cluster")
        if cid is not None:
            cluster_map[cid].append(node)

    # Step 2: Add start/end nodes as-is
    for node in G.nodes:
        node_type = G.nodes[node].get("type")
        if node_type in ["start", "end"]:
            color = "#f5e642" if node_type == "start" else "#f54242"
            H.add_node(node, label=node, type=node_type, size=20, color=color)

    # Step 3: Add supernodes for each cluster
    for cid, members in cluster_map.items():
        label = f"cluster_{cid}"

        fitnesses = [G.nodes[n].get("fitness", float("inf")) for n in members]
        entropies = [G.nodes[n].get("entropy", 0.0) for n in members if "entropy" in G.nodes[n]]

        avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0
        min_fitness = min(fitnesses) if fitnesses else 0
        max_fitness = max(fitnesses) if fitnesses else 0
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0

        node_size = math.log2(len(members) + 1) * 10

        if color_attr == "fitness":
            g = int(255 - min(avg_fitness, 1.0) * 255)
            node_color = f"rgba(100, {g}, 100, 0.8)"
        elif color_attr == "entropy":
            r = int(min(avg_entropy, 1.0) * 255)
            node_color = f"rgba({r}, 100, 100, 0.8)"
        else:
            node_color = "gray"

        H.add_node(label,
                   label=label,
                   size=node_size,
                   count=len(members),
                   avg_fitness=avg_fitness,
                   fitness_range=(min_fitness, max_fitness),
                   avg_entropy=avg_entropy,
                   color=node_color)

    # Step 4: Redirect edges
    for u, v in G.edges:
        cu = G.nodes[u].get("cluster")
        cv = G.nodes[v].get("cluster")
        type_u = G.nodes[u].get("type")
        type_v = G.nodes[v].get("type")

        if type_u in ["start", "end"] and type_v == "intermediate":
            target_cluster = f"cluster_{cv}"
            H.add_edge(u, target_cluster, weight=1)
        elif type_v in ["start", "end"] and type_u == "intermediate":
            source_cluster = f"cluster_{cu}"
            H.add_edge(source_cluster, v, weight=1)
        elif type_u == "intermediate" and type_v == "intermediate" and cu != cv:
            u_cluster = f"cluster_{cu}"
            v_cluster = f"cluster_{cv}"
            if H.has_edge(u_cluster, v_cluster):
                H[u_cluster][v_cluster]["weight"] += 1
            else:
                H.add_edge(u_cluster, v_cluster, weight=1)

    return H