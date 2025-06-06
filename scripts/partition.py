import math
import numpy as np
from scripts import utils
from itertools import combinations
import networkx as nx
from scripts.structures import ConfigData
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform


def discrete_shannon_entropy(G, config):

    sample_node = next(iter(G.nodes))
    vec = utils.sol_to_vector(sample_node)
    solution_type = utils.detect_solution_type(vec)
    if not utils.is_entropy_applicable(solution_type):
        print("Warning: Shannon entropy does not make sense for the problem type")
        return
    print("Shannon Entropy Partitioning")

    nr = config.dPartitioning

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
        vec_raw = utils.sol_to_vector(node_id)
        solution_type = utils.detect_solution_type(vec_raw)
        if solution_type == "binary":
            vec = [int(x) for x in vec_raw]
        else:
            vec = vec_raw
        if not utils.is_entropy_applicable(solution_type):
            continue

        entropy = utils.shannon_entropy_vector(vec)
        entropies.append((node_id, entropy))

    node_count = len(entropies)
    del_count = int((node_count * nr) / 100)

    entropies.sort(key=lambda x: x[1])
    nodes_to_remove = [node_id for node_id, _ in entropies[:del_count]]

    print(f"Percentage of nodes to be deleted: {nr}")
    print(f"Total nodes before: {len(G.nodes)}")
    print(f"Total edges before: {len(G.edges)}")

    for node_id in nodes_to_remove:
        if node_id in G and G.nodes[node_id].get("type") == "intermediate":
            utils.remove_node_between_two(G, node_id)

    print(f"Total nodes after: {len(G.nodes)}")
    print(f"Total edges after: {len(G.edges)}")

    return G


def cont_standard_partitioning(G, hypercube_factor=-2, min_bound=0.0, max_bound=1.0):
    min_nodes_per_bin = 5
    edge_weight_threshold = 1
    keep_top_k_edges = 3

    if hypercube_factor == 0:
        return G
    if hypercube_factor > 0:
        cube_size = hypercube_factor
    else:
        cube_size = (max_bound - min_bound) / (10 ** abs(int(hypercube_factor)))

    node_vectors = {}
    node_meta = {}
    bin_map = {}
    bin_contents = defaultdict(list)
    supernode_counts = defaultdict(int)
    supernode_fitness = defaultdict(list)
    supernode_iterations = defaultdict(list)
    supernode_types = defaultdict(set)

    for node in G.nodes:
        vec = utils.parse_vectors_string(node, mode='auto')[0]
        if len(vec) > 5:
            continue

        # Compute bin index
        bin_indices = tuple(
            min(int((x - min_bound) / cube_size), int((max_bound - min_bound) / cube_size) - 1)
            for x in vec
        )
        cube_id = str(bin_indices)
        bin_map[node] = cube_id
        bin_contents[cube_id].append(node)

        # Collect metadata
        node_vectors[node] = vec
        meta = G.nodes[node]
        node_meta[node] = meta
        supernode_counts[cube_id] += meta.get("count", 1)
        supernode_fitness[cube_id].append(meta.get("fitness", float("inf")))
        supernode_iterations[cube_id].append(meta.get("iteration", 0))
        supernode_types[cube_id].add(meta.get("type", "intermediate"))

    # Filter bins with too few nodes
    filtered_bins = {b for b, nodes in bin_contents.items() if len(nodes) >= min_nodes_per_bin}

    # Create supernode graph
    H = nx.DiGraph()
    for cube_id in filtered_bins:
        types = supernode_types[cube_id]
        node_type = "start" if "start" in types else "end" if "end" in types else "intermediate"
        fitness_vals = supernode_fitness[cube_id]
        avg_fitness = sum(fitness_vals) / len(fitness_vals) if fitness_vals else float("inf")
        min_fitness = min(fitness_vals) if fitness_vals else float("inf")
        avg_iter = sum(supernode_iterations[cube_id]) / len(supernode_iterations[cube_id])

        H.add_node(cube_id,
                   count=supernode_counts[cube_id],
                   avg_fitness=avg_fitness,
                   min_fitness=min_fitness,
                   avg_iteration=avg_iter,
                   type=node_type)

    # Edge aggregation
    edge_map = defaultdict(lambda: defaultdict(list))
    for u, v in G.edges:
        if u not in bin_map or v not in bin_map:
            continue
        cu, cv = bin_map[u], bin_map[v]
        if cu == cv:
            continue
        if cu not in filtered_bins or cv not in filtered_bins:
            continue

        u_meta, v_meta = node_meta[u], node_meta[v]
        if u_meta.get("run_id") != v_meta.get("run_id"):
            continue
        if v_meta.get("iteration", 0) <= u_meta.get("iteration", 0):
            continue

        edge_map[cu][cv].append({
            "weight": G[u][v].get("weight", 1),
            "iter_gap": v_meta.get("iteration", 0) - u_meta.get("iteration", 0),
            "fitness_diff": v_meta.get("fitness", float("inf")) - u_meta.get("fitness", float("inf"))
        })

    # Finalize edges with aggregation and filtering
    for cu in edge_map:
        sorted_cv = sorted(edge_map[cu].items(), key=lambda item: len(item[1]), reverse=True)
        for i, (cv, transitions) in enumerate(sorted_cv[:keep_top_k_edges]):
            weight = sum(t["weight"] for t in transitions)
            if weight < edge_weight_threshold:
                continue
            avg_iter_gap = sum(t["iter_gap"] for t in transitions) / len(transitions)
            avg_fit_diff = sum(t["fitness_diff"] for t in transitions) / len(transitions)
            H.add_edge(cu, cv,
                       weight=weight,
                       avg_iteration_gap=avg_iter_gap,
                       avg_fitness_delta=avg_fit_diff)

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
            return discrete_shannon_entropy(G, config)
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
                hypercube_factor = n_bins,
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