from werkzeug.utils import secure_filename
from scripts.structures import ConfigData, AdvancedSettings
from flask import session, request
import networkx as nx
import scripts.create as create
import scripts.partition as partition
import scripts.visualize as visualize
import scripts.utils as utils
import os

# ----------- Extracting and storing data from frontend -----------

# Retrieves form data from the session
# Converts it into a ConfigData object
def get_config_from_session():
    config_dict = session.get("config_data")
    if not config_dict:
        raise ValueError("Configuration data missing from session.")
    return ConfigData.from_dict(config_dict)


# Retrieves advanced settings from the session
# Converts them into an AdvancedSettings object
def get_advanced_from_session():
    advanced_dict = session.get("advanced_settings")
    if not advanced_dict:
        raise ValueError("Advanced settings missing from session.")
    return AdvancedSettings.from_dictionary(advanced_dict)


# Extracts uploaded algorithm metadata and files from the request
# Returns a list of dictionaries, one per algorithm
def get_upload_file_data(request, upload_dir="uploads"):
    # Ensure the upload directory exists
    os.makedirs(upload_dir, exist_ok=True)

    # Get the number of submitted algorithms
    total = int(request.form.get("total", 0))
    if total == 0:
        raise ValueError("No algorithms were submitted.")

    files_data = []

    # Loop through all submitted algorithms
    for i in range(total):
        name = request.form.get(f"name_{i}")
        color = request.form.get(f"color_{i}")
        file = request.files.get(f"file_{i}")

        if not file:
            raise ValueError(f"Missing file for algorithm {i}.")

        # Sanitize the filename and save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)

        # Store the algorithm's metadata and saved file path
        files_data.append({
            "name": name,
            "color": color,
            "path": filepath
        })

    return files_data


# Loads all submitted configuration and file data from the current request/session
# Returns deserialized config objects and file metadata
def load_config_and_files():
    try:
        # Load form data into ConfigData and AdvancedSettings objects
        config = get_config_from_session()
        advanced = get_advanced_from_session()
        files_data = get_upload_file_data(request)

        # Cache uploaded algorithm metadata in the session
        session["algorithms"] = files_data

        return config, advanced, files_data

    except ValueError as e:
        raise RuntimeError(f"Config load failed: {e}")


def validate_best_solution_format(best_str: str, config: ConfigData, sample_solution: str | None = None):
    if not best_str.strip():
        return  # Empty best solution is allowed

    try:
        if config.problemType == "continuous":
            best_vec = utils.parse_continuous_solution(best_str)
            sample_vec = utils.parse_continuous_solution(sample_solution)
            if len(best_vec) != len(sample_vec):
                raise ValueError(
                    f"Best solution has {len(best_vec)} values, but sample has {len(sample_vec)}."
                )
            if config.partitionStrategy == "partitioning":
                if len(sample_vec) > config.cDimension:
                    raise ValueError(
                        f"Declared number of dimensions ({config.cDimension}) is less than the actual solution dimensionality ({len(sample_vec)}). "
                    )

        elif config.problemType == "discrete":
            best_vecs = utils.parse_discrete_solutions(best_str)
            if not best_vecs or not isinstance(best_vecs[0], list):
                raise ValueError("Could not parse best solution as a discrete vector.")

            best_vec = best_vecs[0]
            detected_type = utils.detect_solution_type(best_vec)

            if sample_solution:
                sample_vec = utils.parse_discrete_solutions(sample_solution)[0]
                if len(best_vec) != len(sample_vec):
                    raise ValueError("Best solution length does not match sample solution length.")
                if utils.detect_solution_type(sample_vec) != detected_type:
                    raise ValueError("Best solution type does not match sample solution type.")
        else:
            raise ValueError(f"Unknown problem type: {config.problemType}")
    except Exception as e:
        raise ValueError(f"Invalid best solution format: {e}")



def validate_config_data(config: ConfigData, advanced: AdvancedSettings) -> None:
    if config.cMinBound >= config.cMaxBound:
        raise ValueError("Minimum bound must be less than maximum bound.")
    if abs(config.cMinBound) > 1e6 or abs(config.cMaxBound) > 1e6:
        raise ValueError("Bounds must be within [-1e6, 1e6].")



# ----------- Graph construction and visualization -----------

# Creates a list of graphs from uploaded algorithm data
# Each graph is built and tagged with its associated color
def build_graphs(files_data, nr_of_runs: int, best_solution: str, objective_type: str):
    graphs = []
    for algo in files_data:
        G = create.create_stn(algo["path"], nr_of_runs, best_solution, objective_type)
        visualize.tag_graph_origin(G, algo["color"])
        graphs.append(G)
    return graphs


# Applies the selected partitioning strategy and visualizes the results
def apply_partition_and_visualize(graphs, config, advanced, files_data):
    # Prepares legend: maps colors to algorithm names
    legend_entries = {algo["color"]: algo["name"] for algo in files_data}

    # Tags each graph's nodes with its origin color
    for i, G in enumerate(graphs):
        color = files_data[i]["color"]
        for node in G.nodes:
            G.nodes[node]["origin_color"] = color

    if config.partitionStrategy == "clustering":
        # Merge all graphs and apply clustering
        for i, G in enumerate(graphs):
            for node in G.nodes:
                G.nodes[node]["origin_color"] = files_data[i]["color"]

        merged = nx.compose_all(graphs)

        if config.problemType == "continuous":
            final_clusters = partition.continuous_clustering(merged, config)
        else:
            final_clusters = partition.discrete_clustering(merged, config)

        # Store clustered graph into a html output file
        visualize.visualize_clusters(
            final_clusters,
            merged,
            output_file="static/graph_fr.html",
            legend_entries=legend_entries,
            objective_type=config.objectiveType,
            advanced=advanced,
            layout="fr"
        )
        visualize.visualize_clusters(
            final_clusters,
            merged,
            output_file="static/graph_kk.html",
            legend_entries=legend_entries,
            objective_type=config.objectiveType,
            advanced=advanced,
            layout="kk"
        )
        return

    else:
        # Apply individual graph partitioning (standard or Shannon)
        partitioned_graphs = []

        for i, G in enumerate(graphs):
            if config.partitionStrategy == "partitioning":
                partitioned = partition.standard_partitioning(G, config)
                visualize.tag_graph_origin(partitioned, files_data[i]["color"])
                partitioned_graphs.append(partitioned)

            elif config.partitionStrategy == "shannon":
                partitioned = partition.shannon_entropy_partitioning(G, config)
                visualize.tag_graph_origin(partitioned, files_data[i]["color"])
                partitioned_graphs.append(partitioned)

            else:
                # Fallback if no recognized strategy
                visualize.tag_graph_origin(G, files_data[i]["color"])
                partitioned_graphs.append(G)

        # Visualize graph visualizations with both FR and KK layouts
        visualize.visualize_stn(
            partitioned_graphs,
            advanced,
            config,
            output_file="static/graph_fr.html",
            minmax=config.objectiveType,
            legend_entries=legend_entries,
            layout="fr"
        )

        visualize.visualize_stn(
            partitioned_graphs,
            advanced,
            config,
            output_file="static/graph_kk.html",
            minmax=config.objectiveType,
            legend_entries=legend_entries,
            layout="kk"
        )

