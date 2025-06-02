# debug_partitioning.py
import os
import webbrowser
from scripts.create import create_stn
from scripts.partition import apply_partitioning_from_config
from scripts.visualize import tag_graph_origin, visualize_stn
from scripts.structures import ConfigData

config = ConfigData(
        problemType="discrete",
        objectiveType="minimization",
        partitionStrategy="clustering",

        dPartitioning=0,

        dCSize=5,
        dVSize=20,
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
    G = apply_partitioning_from_config(G, config)
    tag_graph_origin(G, algo["color"])
    graphs.append(G)

legend_entries = {algo["color"]: algo["name"] for algo in files_data}

visualize_stn(graphs, output_file="graph_debug.html", minmax=config.objectiveType, legend_entries=legend_entries)

webbrowser.open_new_tab("graph_debug.html")
