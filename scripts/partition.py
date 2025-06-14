import numpy as np
import networkx as nx
from scripts import utils
from scripts.structures import ConfigData
from scripts.utils import parse_vectors_string
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster



# ------------------ Partitioning methods ------------------

# Applies Shannon Entropy pruning on a graph
# Removes low-entropy intermediate nodes based on a user-defined percentage
def shannon_entropy_partitioning(G, config):
    # Detect the type of solution (binary, categorical, permutation)
    solution_type = utils.detect_solution_type_on_sample(G)

    if not utils.is_entropy_applicable(solution_type):
        print("Warning: Shannon entropy does not make sense for the problem type")
        return

    print("Shannon Entropy Partitioning")

    # Percentage of nodes to remove
    nr = config.dPartitioning

    # Identify start and end nodes
    start_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "start"]
    end_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "end"]
    if not start_nodes or not end_nodes:
        raise ValueError("Start and end nodes are required.")

    start, end = start_nodes[0], end_nodes[0]

    # Try to find the shortest path between start and end
    try:
        critical_path = set(nx.shortest_path(G, source=start, target=end))
    except nx.NetworkXNoPath:
        critical_path = set()

    entropies = []

    # Compute entropy for each node not on the critical path
    for node_id, _ in G.nodes(data=True):
        if node_id in critical_path:
            continue

        vec_raw = utils.sol_to_vector(node_id)
        solution_type = utils.detect_solution_type(vec_raw)

        # Convert binary strings to integer list
        if solution_type == "binary":
            vec = [int(x) for x in vec_raw]
        else:
            vec = vec_raw
        if not utils.is_entropy_applicable(solution_type):
            continue

        # Compute Shannon entropy of the vector
        entropy = utils.shannon_entropy_vector(vec)
        entropies.append((node_id, entropy))

    # Sort nodes by entropy and compute how many to delete
    node_count = len(entropies)
    del_count = int((node_count * nr) / 100)
    entropies.sort(key=lambda x: x[1])
    nodes_to_remove = [node_id for node_id, _ in entropies[:del_count]]

    print(f"Percentage of nodes to be deleted: {nr}")
    print(f"Total nodes before: {len(G.nodes)}")
    print(f"Total edges before: {len(G.edges)}")

    # Remove selected nodes that are marked as intermediate
    for node_id in nodes_to_remove:
        if node_id in G and G.nodes[node_id].get("type") == "intermediate":
            utils.remove_node_between_two(G, node_id)

    print(f"Total nodes after: {len(G.nodes)}")
    print(f"Total edges after: {len(G.edges)}")

    return G


# Performs agglomerative clustering for continuous-valued solutions
# Clusters nodes based on vector similarity and range-based volume constraints
def continuous_clustering(G, config: ConfigData) -> list[list[str]]:
    # Check for at least one node
    sample_node = next(iter(G.nodes), None)
    if not sample_node:
        print("Empty graph — no nodes.")
        return []

    # Verify parsing is possible
    try:
        utils.parse_continuous_solution(sample_node)
    except ValueError:
        print("Could not parse any node vector.")
        return []

    # Load distance function from config
    distance_type = config.cDistance
    try:
        distance_fn = utils.get_valid_distance_fn(distance_type, "real")
    except ValueError:
        print(f"Invalid distance type: {distance_type}")
        return []

    # Load clustering constraints
    cluster_size_percent = config.cClusterSize
    volume_size_percent = config.cVolumeSize

    # Parse all valid vectors and fitness values
    node_vectors = {}
    node_fitness = {}
    for node in G.nodes:
        try:
            vec = utils.parse_continuous_solution(node)
            node_vectors[node] = vec
            node_fitness[node] = G.nodes[node]["fitness"]
        except (ValueError, KeyError):
            print(f"Skipping node: {node}")
            continue

    if len(node_vectors) < 2:
        print("Not enough valid nodes to cluster.")
        return []

    ordered_nodes = list(node_vectors.keys())
    vectors = [node_vectors[n] for n in ordered_nodes]

    # Compute distance matrix for clustering
    def condensed_distance_matrix(vectors, fn):
        n = len(vectors)
        dist_list = []
        for i in range(n):
            for j in range(i + 1, n):
                d = utils.compute_distance(vectors[i], vectors[j], fn)
                dist_list.append(d)
        return np.array(dist_list)

    condensed = condensed_distance_matrix(vectors, distance_fn)
    Z = linkage(condensed, method="average")
    threshold = float(Z[-1, 2]) + 1
    cluster_labels = fcluster(Z, t=threshold, criterion="distance")

    # Assign nodes to raw clusters
    clusters_by_label = defaultdict(list)
    for node, label in zip(ordered_nodes, cluster_labels):
        clusters_by_label[label].append(node)

    print(f"\nRaw clusters formed: {len(clusters_by_label)}")

    total_nodes = len(node_vectors)
    max_cluster_size = (cluster_size_percent / 100) * total_nodes

    # Determine domain bounds from all vectors
    dimension = len(vectors[0])
    min_coords = [min(vec[i] for vec in vectors) for i in range(dimension)]
    max_coords = [max(vec[i] for vec in vectors) for i in range(dimension)]
    full_volume = np.prod([max_coords[i] - min_coords[i] for i in range(dimension)])

    # Compute max allowed volume
    max_cluster_volume = (volume_size_percent / 100) * full_volume
    print(f"Max allowed cluster size: {max_cluster_size}")
    print(f"Max allowed cluster volume (solution space): {max_cluster_volume:.4f}")

    final_clusters = []

    for label, cluster_nodes in clusters_by_label.items():
        if not cluster_nodes:
            continue

        cluster_vecs = [node_vectors[n] for n in cluster_nodes]
        per_dim_min = [min(vec[i] for vec in cluster_vecs) for i in range(dimension)]
        per_dim_max = [max(vec[i] for vec in cluster_vecs) for i in range(dimension)]
        cluster_volume = np.prod([per_dim_max[i] - per_dim_min[i] for i in range(dimension)])

        print(f"Cluster {label}: size={len(cluster_nodes)}, volume={cluster_volume:.6f}")

        if cluster_volume > max_cluster_volume:
            # Re-bin cluster by coordinate-based fitness surrogate
            # (this fallback keeps structure similar — could use PCA or entropy here too)
            bin_width = max_cluster_volume ** (1 / dimension)
            bins = defaultdict(list)
            for node in cluster_nodes:
                vec = node_vectors[node]
                bin_index = tuple(int((vec[i] - min_coords[i]) / bin_width) for i in range(dimension))
                bins[bin_index].append(node)

            for bin_nodes in bins.values():
                for i in range(0, len(bin_nodes), int(max_cluster_size)):
                    final_clusters.append(bin_nodes[i:i + int(max_cluster_size)])
        else:
            for i in range(0, len(cluster_nodes), int(max_cluster_size)):
                final_clusters.append(cluster_nodes[i:i + int(max_cluster_size)])

    print(f"\nFinal accepted clusters: {len(final_clusters)}")
    return final_clusters



# Performs agglomerative clustering for discrete-valued solutions
# Clusters nodes based on vector similarity and enforces cluster size and volume constraints
def discrete_clustering(G, config : ConfigData) -> list[list[str]]:
    # Try to get a sample node to check if parsing is possible
    sample_node = next(iter(G.nodes), None)
    if not sample_node:
        print("Empty graph — no nodes.")
        return []

    # Make sure the node can be parsed as a discrete solution
    try:
        utils.parse_discrete_solutions(sample_node)
    except ValueError:
        print("Could not parse any node as discrete vector.")
        return []

    # Get the distance function selected by the user
    distance_type = config.dDistance
    try:
        distance_fn = utils.get_valid_distance_fn(distance_type, "discrete")
    except ValueError:
        print(f"Invalid distance type for discrete space: {distance_type}")
        return []

    # Get cluster size and volume thresholds from config
    cluster_size_percent = config.dCSize
    volume_size_percent = config.dVSize

    # Build a dictionary of parsed solution vectors and fitness values
    node_vectors = {}
    node_fitness = {}
    for node in G.nodes:
        try:
            vec = utils.parse_discrete_solutions(node)[0]  # flatten
            node_vectors[node] = vec
            node_fitness[node] = G.nodes[node]["fitness"]
        except (ValueError, KeyError):
            print(f"Skipping node: {node}")
            continue

    # If not enough valid nodes are found, stop
    if len(node_vectors) < 2:
        print("Not enough valid nodes to cluster.")
        return []

    # Normalize fitness values to [0, 1] so they can be used in volume checks
    global_min_fitness = min(node_fitness.values())
    global_max_fitness = max(node_fitness.values())
    global_fitness_range = global_max_fitness - global_min_fitness

    if global_fitness_range == 0:
        print("All nodes have identical fitness — assigning 0.0 to all.")
        for node in node_fitness:
            node_fitness[node] = 0.0
    else:
        for node in node_fitness:
            node_fitness[node] = (
                node_fitness[node] - global_min_fitness
            ) / global_fitness_range

    # Get node order and associated vectors
    ordered_nodes = list(node_vectors.keys())
    vectors = [node_vectors[n] for n in ordered_nodes]

    # Compute a condensed pairwise distance matrix between solution vectors
    def condensed_distance_matrix(vectors, fn):
        n = len(vectors)
        dist_list = []
        for i in range(n):
            for j in range(i + 1, n):
                d = utils.compute_distance(vectors[i], vectors[j], fn)
                dist_list.append(d)
        return np.array(dist_list)

    # Run agglomerative clustering using the selected distance function
    condensed = condensed_distance_matrix(vectors, distance_fn)
    Z = linkage(condensed, method="average")
    threshold = float(Z[-1, 2]) + 1
    cluster_labels = fcluster(Z, t=threshold, criterion="distance")

    # Group nodes by cluster label
    clusters_by_label = defaultdict(list)
    for node, label in zip(ordered_nodes, cluster_labels):
        clusters_by_label[label].append(node)

    print(f"\nRaw clusters formed: {len(clusters_by_label)}")

    # Compute size and volume limits in absolute terms
    total_nodes = len(node_vectors)
    max_cluster_size = (cluster_size_percent / 100) * total_nodes
    max_cluster_volume = volume_size_percent / 100  # fitness is normalized

    print(f"Max allowed cluster size: {max_cluster_size}")
    print(f"Max allowed cluster volume (normalized): {max_cluster_volume:.4f}")

    final_clusters = []

    # Refine clusters to respect volume and size constraints
    for label, cluster_nodes in clusters_by_label.items():
        if not cluster_nodes:
            continue

        cluster_fitnesses = [node_fitness[n] for n in cluster_nodes]
        cluster_volume = max(cluster_fitnesses) - min(cluster_fitnesses)

        print(f"Cluster {label}: size={len(cluster_nodes)}, volume={cluster_volume:.6f}")

        if cluster_volume > max_cluster_volume:
            # Split the cluster by fitness bins
            bin_width = max_cluster_volume
            bins = defaultdict(list)
            for node in cluster_nodes:
                fitness = node_fitness[node]
                bin_index = int(fitness / bin_width)
                bins[bin_index].append(node)

            # Then apply size constraints to each bin
            for bin_nodes in bins.values():
                for i in range(0, len(bin_nodes), int(max_cluster_size)):
                    final_clusters.append(bin_nodes[i:i + int(max_cluster_size)])
        else:
            # Only enforce size constraint
            for i in range(0, len(cluster_nodes), int(max_cluster_size)):
                final_clusters.append(cluster_nodes[i:i + int(max_cluster_size)])

    print(f"\nFinal accepted clusters: {len(final_clusters)}")
    return final_clusters



# Performs agglomerative clustering for discrete-valued solutions
# Clusters nodes based on vector similarity and normalized fitness
def standard_partitioning(G: nx.DiGraph, config: ConfigData) -> nx.DiGraph:
    pf = config.cHypercube
    configured_dim = config.cDimension
    lower_bound = config.cMinBound
    upper_bound = config.cMaxBound

    hypercube_length = 10 ** pf

    solution_to_cube = {}
    cube_types = defaultdict(lambda: {"start": 0, "end": 0, "intermediate": 0})
    cube_members = defaultdict(list)
    cube_fitnesses = defaultdict(list)


    for node in G.nodes:
        parsed = parse_vectors_string(node)
        if not parsed:
            continue

        vec = np.array(parsed[0])
        actual_dim = len(vec)
        used_dim = min(actual_dim, configured_dim)

        # Truncate the vector to used dimension
        vec = vec[:used_dim]

        if np.any(vec < lower_bound) or np.any(vec >= upper_bound):
            continue
        # Assign to hypercube
        cube_id = tuple(int(np.floor((vec[i] - lower_bound) / hypercube_length)) for i in range(used_dim))
        solution_to_cube[node] = cube_id
        cube_members[cube_id].append(node)

        print(f"Node: {parsed}  Cube: {cube_id}")

        if "fitness" in G.nodes[node]:
            cube_fitnesses[cube_id].append(G.nodes[node]["fitness"])

        # Record types
        node_type = G.nodes[node].get("type", "intermediate")
        if node_type in cube_types[cube_id]:
            cube_types[cube_id][node_type] += 1

    # Step 2: Build new graph
    H = nx.DiGraph()

    # Add hypercube nodes
    for cube_id, type_counts in cube_types.items():
        str_cube_id = str(cube_id)  # Convert once
        H.add_node(str_cube_id)
        H.nodes[str_cube_id]["original_id"] = cube_id

        # Determine the node type of the hypercube
        if type_counts["end"] > 0:
            H.nodes[str_cube_id]["type"] = "end"
        elif type_counts["start"] > 0:
            H.nodes[str_cube_id]["type"] = "start"
        else:
            H.nodes[str_cube_id]["type"] = "intermediate"

        # Determine the count of nodes in each hypercube
        H.nodes[str_cube_id]["count"] = len(cube_members[cube_id])

        fitnesses = cube_fitnesses.get(cube_id, [])
        if fitnesses:
            if config.objectiveType == "minimization":
                best_fitness = min(fitnesses)
            else:
                best_fitness = max(fitnesses)
            H.nodes[str_cube_id]["fitness"] = best_fitness

    #Add edges between hypercubes
    for u, v in G.edges:
        if u not in solution_to_cube or v not in solution_to_cube:
            continue

        cube_u = str(solution_to_cube[u])
        cube_v = str(solution_to_cube[v])

        if cube_u == cube_v:
            continue  # Skip self-transitions inside same hypercube

        if H.has_edge(cube_u, cube_v):
            H[cube_u][cube_v]["weight"] += 1
        else:
            H.add_edge(cube_u, cube_v, weight=1)

    print(f"lower bound: {lower_bound}\nupper_bound: {upper_bound}\npf: {pf}\nlength: {hypercube_length}")
    return H
