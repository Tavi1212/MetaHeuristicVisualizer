from werkzeug.utils import secure_filename
from scripts.structures import ConfigData, AdvancedSettings
from flask import session, request
import networkx as nx
import scripts.create as create
import scripts.partition as partition
import scripts.visualize as visualize
import scripts.utils as utils
import os

def load_config_and_files():
    try:
        config = get_config_from_session()
        advanced = get_advanced_from_session()
        files_data = get_upload_file_data(request)
        session["algorithms"] = files_data
        return config, advanced, files_data
    except ValueError as e:
        raise RuntimeError(f"Config load failed: {e}")

def get_config_from_session():
    config_dict = session.get("config_data")
    if not config_dict:
        raise ValueError("Configuration data missing from session.")
    return ConfigData.from_dict(config_dict)

def get_advanced_from_session():
    advanced_dict = session.get("advanced_settings")
    if not advanced_dict:
        raise ValueError("Advanced settings missing from session.")
    return AdvancedSettings.from_dictionary(advanced_dict)

def get_upload_file_data(request, upload_dir="uploads"):
    os.makedirs(upload_dir, exist_ok=True)

    total = int(request.form.get("total", 0))
    if total == 0:
        raise ValueError("No algorithms were submitted.")

    files_data = []

    for i in range(total):
        name = request.form.get(f"name_{i}")
        color = request.form.get(f"color_{i}")
        file = request.files.get(f"file_{i}")

        if not file:
            raise ValueError(f"Missing file for algorithm {i}.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)

        files_data.append({
            "name": name,
            "color": color,
            "path": filepath
        })

    return files_data

def build_graphs(files_data):
    graphs = []
    for algo in files_data:
        G = create.create_stn(algo["path"])
        visualize.tag_graph_origin(G, algo["color"])
        graphs.append(G)
    return graphs

def apply_partition_and_visualize(graphs, config, advanced, files_data):
    legend_entries = {algo["color"]: algo["name"] for algo in files_data}

    partitioned_graphs = []
    for G in graphs:
        if config.partitionStrategy == "partitioning":
            partitioned = partition.standard_partitioning(G, config)
            partitioned_graphs.append(partitioned)

        elif config.partitionStrategy == "shannon":
            partitioned = partition.shannon_entropy_partitioning(G, config)
            partitioned_graphs.append(partitioned)

        elif config.partitionStrategy == "clustering":
            if config.problemType == "continuous":
                clustered = partition.continuous_clustering(G, config)
            else:
                clustered = partition.discrete_clustering(G, config)
            partitioned_graphs.append(clustered)

        else:
            # Fallback if no recognized partition strategy
            partitioned_graphs.append(G)

    # Final visualization of all processed graphs
    visualize.visualize_stn(
        partitioned_graphs,
        advanced,
        config,
        output_file="static/graph.html",
        minmax=config.objectiveType,
        legend_entries=legend_entries
    )
