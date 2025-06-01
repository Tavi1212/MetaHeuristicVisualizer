import math
import networkx as nx
from scripts.utils import normalize
import scripts.utils as utils
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform


def discrete_shannon_entropy(G, nr):
    entropies = []
    for node_id, _ in G.nodes(data=True):
        entropy = utils.shannon_entropy_string(node_id)
        entropies.append((node_id, entropy))

    node_count = len(entropies)
    del_count = int((node_count * nr) / 100)

    entropies.sort(key=lambda x: x[1])
    nodes_to_remove = [node_id for node_id, _ in entropies[:del_count]]

    for node_id in nodes_to_remove:
        if node_id in G:
            utils.remove_node_between_two(G, node_id)


def cont_standard_partitioning(G, n_bins=5, min_bound=0.0, max_bound=1.0):

    if n_bins < 0:
        n_bins = 10 ** abs(n_bins)

    cube_size = (max_bound - min_bound) / n_bins
    cube_map = {}
    supernode_counts = defaultdict(int)

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

    H = nx.DiGraph()

    for node, cube in cube_map.items():
        if cube not in H:
            H.add_node(
                cube,
                count=supernode_counts[cube],
                shape=G.nodes[node].get("shape", "circle"),
                color=G.nodes[node].get("color", "gray")
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

def estimate_volume_percent(cluster_vectors, domain_size=2):
    dimensions = list(zip(*cluster_vectors))  # transpose
    per_dim_variation = [len(set(col)) for col in dimensions]
    cluster_volume = 1
    for v in per_dim_variation:
        cluster_volume *= v
    total_volume = domain_size ** len(dimensions)
    return (cluster_volume / total_volume) * 100

def apply_agglomerative_clustering(G, max_cluster_percentage=50, max_volume_percentage=50, distance_metric="hamming"):
    nodes = list(G.nodes)
    node_count = len(nodes)

    if len(set(map(len, nodes))) != 1:
        raise ValueError("All node strings must be the same length")

    node_vectors = [[int(c) for c in n] for n in nodes]
    dist_fn = get_distance_fn(distance_metric)
    dist_matrix = squareform(pdist(node_vectors, lambda u, v: dist_fn(u, v)))

    n_clusters = max(1, int((node_count * max_cluster_percentage) / 100))

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(dist_matrix)

    for node, label in zip(nodes, labels):
        G.nodes[node]['cluster'] = label

    # Volume constraint check (discrete only)
    violations = []
    for cluster_id in set(labels):
        cluster_data = [node_vectors[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
        volume_pct = estimate_volume_percent(cluster_data)
        if volume_pct > max_volume_percentage:
            violations.append((cluster_id, volume_pct))
    if violations:
        print("Clusters exceeding volume limit:")
        for cid, vol in violations:
            print(f" - Cluster {cid}: volume = {vol:.2f}%")

    return G

def collapse_to_supernodes(G):
    cluster_ids = set(nx.get_node_attributes(G, 'cluster').values())
    H = nx.DiGraph()

    for c in cluster_ids:
        H.add_node(c, label=f"Cluster {c}", size=sum(G.nodes[n].get('count', 1) for n in G.nodes if G.nodes[n]['cluster'] == c))

    for u, v in G.edges:
        cu = G.nodes[u]['cluster']
        cv = G.nodes[v]['cluster']
        if cu != cv:
            if H.has_edge(cu, cv):
                H[cu][cv]['weight'] += 1
            else:
                H.add_edge(cu, cv, weight=1)

    return H

def apply_general_clustering(G, max_cluster_percentage=50, distance_metric="euclidean", is_discrete=True):
    from utils import parse_vectors_string, hamming_distance, euclidean_distance, manhattan_distance

    nodes = list(G.nodes)
    node_vectors = []

    for node in nodes:
        parsed = parse_vectors_string(node)
        flat = parsed[0]
        # Discrete = categorical characters; Continuous = numeric
        if is_discrete:
            node_vectors.append([int(c) if isinstance(c, str) and c.isdigit() else c for c in flat])
        else:
            node_vectors.append([float(x) for x in flat])

    dist_fn_map = {
        "hamming": hamming_distance,
        "euclidean": euclidean_distance,
        "manhattan": manhattan_distance
    }

    distance_metric = distance_metric.lower()
    if distance_metric not in dist_fn_map:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")
    if not is_discrete and distance_metric == "hamming":
        raise ValueError("Hamming distance is only valid for discrete problems.")

    dist_fn = dist_fn_map[distance_metric]
    dist_matrix = squareform(pdist(node_vectors, lambda u, v: dist_fn(u, v)))
    n_clusters = max(1, int((len(nodes) * max_cluster_percentage) / 100))

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(dist_matrix)

    for node, label in zip(nodes, labels):
        G.nodes[node]['cluster'] = label

    return collapse_to_supernodes(G)

def apply_partitioning_from_config(G, config):
    if config.problemType == "discrete":
        if config.partitionStrategy == "shannon":
            discrete_shannon_entropy(G, nr=config.dPartitioning)
            return G

        elif config.partitionStrategy == "clustering":
            return apply_general_clustering(
                G,
                max_cluster_percentage=config.dCSize,
                distance_metric=config.dDistance,
                is_discrete=True
            )

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
            return apply_general_clustering(
                G,
                max_cluster_percentage=config.cClusterSize,
                distance_metric=config.cDistance,
                is_discrete=False
            )

    raise ValueError(f"Unsupported configuration: {config.problemType}, {config.partitionStrategy}")
