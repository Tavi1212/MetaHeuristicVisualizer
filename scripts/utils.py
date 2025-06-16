import ast
import math
import colorsys
import networkx as nx
from collections import Counter
from flatbuffers.flexbuffers import String

# Normalize a value using scaled bounds
def normalize(value, min_val, max_val, scaled_min=10, scaled_max=50):
    if min_val == max_val:
        return (scaled_min + scaled_max) / 2
    return scaled_min + (value - min_val) / (max_val - min_val) * (scaled_max - scaled_min)

# Adjust color brightness
def adjust_color_lightness(hex_color, lightness_percent):
    lightness_percent = max(30, min(90, lightness_percent))

    # Convert hex to RGB
    hex_color = hex_color.lstrip("#")
    r, g, b = [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]

    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = lightness_percent / 100.0
    r, g, b = colorsys.hls_to_rgb(h, l, s)

    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

# Compute shannon entropy for a binary vector
def shannon_entropy_vector(vec):
    if not isinstance(vec, (list, tuple)):
        vec = list(vec)
    if not vec:
        return 0.0
    counts = Counter(vec)
    total = len(vec)
    probabilities = [count / total for count in counts.values()]
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

# Remove a node from a graph and tie the parents
def remove_node_between_two(G, node):
    if node not in G:
        return
    preds = list(G.predecessors(node))
    succs = list(G.successors(node))

    for u in preds:
        for v in succs:
            if u != v:  # avoid self-loops
                weight = (G[u][node].get("weight", 1) + G[node][v].get("weight", 1)) / 2
                G.add_edge(u, v, weight=weight)

    G.remove_node(node)

# Parse a string representation of a vector into a nested list of values
def parse_vectors_string(s, mode='auto'):
    def hex_to_int32(h):
        val = int(h, 16)
        return val - (1 << 32) if val >= (1 << 31) else val

    if mode == 'hex':
        if len(s) % 8 != 0:
            raise ValueError("Hex string must be divisible by 8")
        chunks = [s[i:i+8] for i in range(0, len(s), 8)]
        return [[hex_to_int32(chunk) for chunk in chunks]]

    elif mode == 'float':
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (int, float)):
                return [[parsed]]
            elif isinstance(parsed, list) and all(isinstance(x, (int, float, str)) for x in parsed):
                return [parsed]
            elif isinstance(parsed, list):
                return parsed
            else:
                raise ValueError("Unsupported float structure")
        except Exception as e:
            raise ValueError(f"Could not parse float: {e}")

    elif mode == 'binary':
        if not all(c in '01' for c in s):
            raise ValueError("Non-binary character found in binary mode")
        return [[int(c) for c in s]]

    elif mode == 'categorical':
        return [[c for c in s]]

    elif mode == 'auto':
        # Try hex mode
        if all(c in '0123456789abcdefABCDEF' for c in s) and len(s) % 8 == 0:
            try:
                return parse_vectors_string(s, mode='hex')
            except Exception:
                pass
        try:
            return parse_vectors_string(s, mode='float')
        except Exception:
            pass
        # Try binary
        if all(c in '01' for c in s):
            return parse_vectors_string(s, mode='binary')
        # Fallback: treat as categorical string
        return parse_vectors_string(s, mode='categorical')

    else:
        raise ValueError(f"Unknown parse mode: {mode}")

# Parse discrete solution based on it's type
def parse_discrete_solutions(s, mode='auto'):
    if mode == 'binary':
        if not all(c in '01' for c in s):
            raise ValueError("Non-binary character found in binary mode")
        return [[int(c) for c in s]]

    elif mode == 'categorical':
        return [[c for c in s]]

    elif mode == 'permutation':
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [parsed]
            else:
                raise ValueError("Permutation must be a list")
        except Exception as e:
            raise ValueError(f"Invalid permutation format: {e}")

    elif mode == 'auto':
        # Try binary first (fast and avoids int misinterpretation)
        if all(c in '01' for c in s):
            return [[int(c) for c in s]]

        # Try permutation
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [parsed]
        except Exception:
            pass

        # Fallback to categorical
        return [[c for c in s]]

    else:
        raise ValueError(f"Unknown parse mode: {mode}")

# Parse a string as a real-valued solution vector
def parse_continuous_solution(s):
    try:
        parsed = ast.literal_eval(s)

        # Single number
        if isinstance(parsed, (int, float)):
            return [float(parsed)]

        if isinstance(parsed, list) and all(isinstance(x, (int, float)) for x in parsed):
            return [float(x) for x in parsed]

        raise ValueError("Invalid continuous solution structure")

    except Exception:
        raise ValueError(f"Failed to parse as continuous solution: {s}")

# Calculate the hamming distance between two binary solutions
def hamming_distance(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Solutions need to be the same length")
    return sum(a != b for a, b in zip(v1, v2))

# Calculate euclidian distance between two solutions
def euclidean_distance(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of equal length")
    try:
        return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(v1, v2)))
    except ValueError as e:
        raise TypeError("Euclidean distance requires numeric vectors") from e

# Calculate manhattan distance between two solutions
def manhattan_distance(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of equal length")
    try:
        return sum(abs(float(a) - float(b)) for a, b in zip(v1, v2))
    except (ValueError, TypeError):
        raise TypeError("Manhattan distance requires numeric inputs")

# Get the distance automatically based on extra parameters
def get_valid_distance_fn(distance_type: str, solution_type: str):
    if distance_type == "hamming":
        return hamming_distance
    elif distance_type == "euclidean":
        if solution_type not in ("binary", "real", "integer"):
            raise ValueError(f"Euclidean distance is invalid for {solution_type}")
        return euclidean_distance
    elif distance_type == "manhattan":
        if solution_type not in ("binary", "real", "integer"):
            raise ValueError(f"Manhattan distance is invalid for {solution_type}")
        return manhattan_distance
    else:
        raise ValueError(f"Unsupported distance type: {distance_type}")

# Calculates the distance given as attribute between two solutions
def compute_distance(sol1, sol2, distance_fn):
    if len(sol1) != len(sol2):
        raise ValueError("Vectors must be of equal length")
    return distance_fn(sol1, sol2)

def sol_to_vector(sol: str):
    try:
        parsed = ast.literal_eval(sol)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, str):
            return list(parsed)  # e.g. "'ABCD'" → ['A','B','C','D']
        else:
            raise ValueError  # force fallback
    except (ValueError, SyntaxError):
        return list(sol)  # fallback: treat raw string as character vector

def is_binary_vector(vec):
    if not isinstance(vec, list):
        return False
    return all(str(x) in ('0', '1') for x in vec)

def is_permutation_vector(vec):
    if not isinstance(vec, list):
        return False
    if len(vec) != len(set(vec)):
        return False
    if not all(isinstance(x, (int, str)) for x in vec):
        return False
    return True

def detect_solution_type(vec):
    if is_binary_vector(vec):
        return "binary"
    elif is_permutation_vector(vec):
        return "permutation"
    else:
        return "categorical"

def detect_solution_type_on_sample(G, sample_size=10):
    intermediates = [n for n in G.nodes if G.nodes[n].get("type") == "intermediate"]
    sampled = intermediates[:sample_size]

    permutation_count = 0

    for node in sampled:
        vec = sol_to_vector(node)
        if is_binary_vector(vec):
            return "binary"
        if is_permutation_vector(vec):
            permutation_count += 1

    if permutation_count == sample_size:
        return "permutation"
    return "categorical"


def are_permutations_of_same_set(vectors):
    if not vectors:
        return True  # empty list is trivially valid

    reference_set = set(vectors[0])
    reference_length = len(vectors[0])

    for vec in vectors[1:]:
        if len(vec) != reference_length:
            return False  # not same length
        if set(vec) != reference_set:
            return False  # not same elements
        if len(set(vec)) != len(vec):
            return False  # duplicate in current vec

    return True

def is_distance_applicable(distance, encoding):
    if distance == "hamming":
        return True
    elif distance in ("euclidean", "manhattan"):
        return encoding == "binary"
    return False

def is_entropy_applicable(solution_type: str):
    solution_type = solution_type.lower()
    return solution_type in ("binary", "categorical")

def merge_graphs_with_count(graphs):
    merged = nx.DiGraph()
    for G in graphs:
        for node, data in G.nodes(data=True):
            if node in merged:
                merged.nodes[node]["count"] += data.get("count", 1)
            else:
                merged.add_node(node, **data)
                if "count" not in merged.nodes[node]:
                    merged.nodes[node]["count"] = data.get("count", 1)
        for u, v, attr in G.edges(data=True):
            merged.add_edge(u, v, **attr)
    return merged


def assign_cluster_levels(G, clusters):
    cluster_graph = nx.DiGraph()
    cluster_nodes = [(i, set(cluster)) for i, cluster in enumerate(clusters)]

    for i, (id_a, nodes_a) in enumerate(cluster_nodes):
        for j, (id_b, nodes_b) in enumerate(cluster_nodes):
            if i != j and any(G.has_edge(u, v) for u in nodes_a for v in nodes_b):
                cluster_graph.add_edge(i, j)

    # Initialize all levels to 0
    levels = {node: 0 for node in cluster_graph.nodes}

    # BFS-style manual traversal to assign levels
    visited = set()
    queue = [n for n in cluster_graph.nodes if cluster_graph.in_degree(n) == 0]

    while queue:
        node = queue.pop(0)
        visited.add(node)
        for succ in cluster_graph.successors(node):
            levels[succ] = max(levels[succ], levels[node] + 1)
            if succ not in visited:
                queue.append(succ)

    # Any unreachable nodes (e.g., in cycles) will stay at level 0
    return levels