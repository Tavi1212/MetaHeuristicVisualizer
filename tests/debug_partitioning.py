import webbrowser
from scripts.create import create_stn
from scripts.visualize import tag_graph_origin, visualize_stn
from scripts.structures import ConfigData, AdvancedSettings
from scripts.partition import standard_partitioning

# === Advanced visualization settings ===
advanced = AdvancedSettings(
    best_solution="",        # Leave empty if you're not highlighting a best solution
    nr_of_runs=None,
    vertex_size=1.0,         # Scale node sizes properly (1.0 means normal scale)
    arrow_size=1.0,          # Scale arrow thickness
    tree_layout=False
)

# === Configuration for standard partitioning on continuous data ===
config = ConfigData(
    problemType="continuous",
    objectiveType="maximization",       # or "minimization"
    partitionStrategy="partitioning",   # this must match the backend expectation, not "standard"

    dPartitioning=0,
    dCSize=50,
    dVSize=25,
    dDistance="hamming",

    cMinBound=45,
    cMaxBound=55,
    cDimension=3,
    cHypercube=0,                       # log10(length); i.e., 10^0 = cube length 1.0

    cClusterSize=0,
    cVolumeSize=0,
    cDistance=""
)

# === Algorithm file input and color assignment ===
files_data = [
    {
        "path": "../input/stn_input1.txt",
        "name": "DE Algorithm",
        "color": "#32a852"  # Set your desired hex color here
    },
]

graphs = []

# === Load and partition each input STN ===
for algo in files_data:
    G = create_stn(algo["path"])                     # Load raw STN
    G = standard_partitioning(G, config)             # Apply standard (continuous) partitioning
    tag_graph_origin(G, algo["color"])               # Assign node colors for this algorithm
    graphs.append(G)

# === Legend for the visualized algorithms ===
legend_entries = {algo["color"]: algo["name"] for algo in files_data}

# === Render the final STN HTML visualization ===
visualize_stn(
    graphs,
    advanced,
    config,
    output_file="graph_debug.html",
    minmax=config.objectiveType,
    legend_entries=legend_entries
)

# === Open the result in the default web browser ===
webbrowser.open_new_tab("graph_debug.html")
