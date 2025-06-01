import ast
import math
import colorsys
from collections import Counter

def normalize(value, min_val, max_val, scaled_min=10, scaled_max=50):
    if min_val == max_val:
        return (scaled_min + scaled_max) / 2
    return scaled_min + (value - min_val) / (max_val - min_val) * (scaled_max - scaled_min)


def adjust_color_lightness(hex_color, lightness_percent):
    lightness_percent = max(30, min(90, lightness_percent))  # Clamp for visibility

    # Convert hex to RGB
    hex_color = hex_color.lstrip("#")
    r, g, b = [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]

    # RGB → HLS, update lightness, then HLS → RGB
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = lightness_percent / 100.0
    r, g, b = colorsys.hls_to_rgb(h, l, s)

    # RGB → hex
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

def shannon_entropy_string(s):
    counts = Counter(s)
    total = len(s)
    probabilities = [count / total for count in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return entropy

def remove_node_between_two(G, v):
    preds = list(G.predecessors(v))
    succs = list(G.successors(v))
    if len(preds) == 1 and len(succs) == 1:
        G.add_edge(preds[0], succs[0])
    G.remove_node(v)


def parse_vectors_string(s):
    # Special case: treat digit-only strings as character vectors
    if s.isdigit():
        return [[int(c) for c in s]]

    try:
        parsed = ast.literal_eval(s)

        # Scalar like 5.21
        if isinstance(parsed, (int, float)):
            return [[parsed]]

        # Flat list [1,2,3]
        if isinstance(parsed, list) and all(isinstance(x, (int, float, str)) for x in parsed):
            return [parsed]

        # List of vectors [[...], [...]]
        elif isinstance(parsed, list):
            return parsed

    except (SyntaxError, ValueError):
        return [[int(c) if c.isdigit() else c for c in s]]

    raise ValueError("Unsupported format for vector string.")

def hamming_distance(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Solutions need to be the same length")
    return sum(a != b for a, b in zip(v1, v2))

def euclidean_distance(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of equal length")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

def manhattan_distance(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of equal length")
    return sum(abs(a - b) for a, b in zip(v1, v2))

def compute_distance(sol1, sol2, distance_fn):
    if len(sol1) != len(sol2):
        raise ValueError("Solutions must contain the same number of subvectors")
    return sum(distance_fn(v1, v2) for v1, v2 in zip(sol1, sol2))