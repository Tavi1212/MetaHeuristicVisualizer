from tensorboard import notebook

from scripts.create import create_stn
from scripts.utils import parse_vectors_string
from scripts.visualize import visualize_stn
from scripts.structures import ConfigData, AdvancedSettings
import matplotlib.pyplot as plt
from collections import defaultdict
from pyvis import network
from pyvis.network import Network
import networkx as nx
import numpy as np
import webbrowser
import os



config = ConfigData(cHypercube=0, cMinBound=50, cMaxBound=60)
advanced = AdvancedSettings()
G = create_stn("../input/stn_input1.txt")

def standard_partitioning(G: nx.DiGraph, config: ConfigData) -> nx.DiGraph:
    pf = config.cHypercube
    dim = config.cDimension
    lower_bound = config.cMinBound
    upper_bound = config.cMaxBound

    hypercube_length = 10 ** pf

    solution_to_cube = {}
    cube_types = defaultdict(lambda: {"start": 0, "end": 0, "intermediate": 0})
    cube_members = defaultdict(list)


    for node in G.nodes:
        parsed = parse_vectors_string(node)
        if not parsed:
            continue

        vec = np.array(parsed[0])

        if dim is None:
            dim = len(vec)

        if np.any(vec < lower_bound) or np.any(vec >= upper_bound):
            continue

        # Assign to hypercube
        cube_id = tuple(int((vec[i] - lower_bound) // hypercube_length) for i in range(dim))
        solution_to_cube[node] = cube_id
        cube_members[cube_id].append(node)

        print(f"Node: {parsed}  Cube: {cube_id}")

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

    return H

H = standard_partitioning(G, config)
visualize_stn([H], AdvancedSettings(tree_layout=False, vertex_size=15), config)

webbrowser.open("file://" + os.path.realpath("graph.html"))


