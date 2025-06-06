import webbrowser
from scripts.create import create_stn
from scripts.visualize import tag_graph_origin, visualize_stn
from scripts.structures import ConfigData, AdvancedSettings
from scripts.partition import cont_standard_partitioning


advanced = AdvancedSettings(
    best_solution="",
    nr_of_runs=None,
    vertex_size=20,
    arrow_size=1,
    tree_layout=False
)

config = ConfigData(
        problemType="continuous",
        objectiveType="maximization",
        partitionStrategy="standard",

        dPartitioning=0,

        dCSize=50,
        dVSize=25,
        dDistance="hamming",

        cMinBound=25,
        cMaxBound=45,
        cDimension=3,
        cHypercube=5,

        cClusterSize=0,
        cVolumeSize=0,
        cDistance=""
)

files_data = [
    {"path": "../input/stn_input1.txt", "name": "DE", "color": "#13aec3"},
]

graphs = []
for algo in files_data:
    G = create_stn(algo["path"])
    G = cont_standard_partitioning(
        G,
        hypercube_factor=config.cHypercube,
        min_bound=config.cMinBound,
        max_bound=config.cMaxBound
    )
    tag_graph_origin(G, algo["color"])
    graphs.append(G)

legend_entries = {algo["color"]: algo["name"] for algo in files_data}

visualize_stn(
    graphs,
    advanced,
    config,
    output_file="graph_debug.html",
    minmax="minimization",
    legend_entries=legend_entries
)

webbrowser.open_new_tab("graph_debug.html")
