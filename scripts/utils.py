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

def remove_node_between_two(G, node):
    if node not in G:
        return
    preds = list(G.predecessors(node))
    succs = list(G.successors(node))

    # Add edges from each pred to each succ with weight = avg or 1
    for u in preds:
        for v in succs:
            if u != v:  # avoid self-loops
                weight = (G[u][node].get("weight", 1) + G[node][v].get("weight", 1)) / 2
                G.add_edge(u, v, weight=weight)

    G.remove_node(node)


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
        # Try float
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