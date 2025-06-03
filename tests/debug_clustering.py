# debug_partitioning.py
import os
import webbrowser
from networkx.classes import nodes
from scripts.create import create_stn
from scripts.partition import apply_partitioning_from_config, constrained_cluster, collapse_to_supernodes
from scripts.visualize import tag_graph_origin, visualize_stn
from scripts.structures import ConfigData

config = ConfigData(
        problemType="discrete",
        objectiveType="minimization",
        partitionStrategy="clustering",

        dPartitioning=0,

        dCSize=50,
        dVSize=80,
        dDistance="hamming",

        dMinBound=0,
        cMaxBound=0,
        cDimension=0,

        cHypercube=0,
        cClusterSize=0,
        cVolumeSize=0,
        cDistance=""
)

files_data = [
    {"path": "../input/BRKGA.txt", "name": "brkga", "color": "#13aec3"},
]

graphs = []
for algo in files_data:
    G = create_stn(algo["path"])
    G = constrained_cluster(G, config)

    collapsed = collapse_to_supernodes(G, color_attr="fitness")  # or "entropy" or None
    tag_graph_origin(collapsed, algo["color"])
    graphs.append(collapsed)


legend_entries = {algo["color"]: algo["name"] for algo in files_data}

visualize_stn(graphs, output_file="graph_debug.html", minmax=config.objectiveType, legend_entries=legend_entries)

webbrowser.open_new_tab("graph_debug.html")
