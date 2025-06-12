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

    # Tag each graph's nodes with its origin color
    for i, G in enumerate(graphs):
        color = files_data[i]["color"]
        for node in G.nodes:
            G.nodes[node]["origin_color"] = color

    if config.partitionStrategy == "clustering":
        for i, G in enumerate(graphs):
            for node in G.nodes:
                G.nodes[node]["origin_color"] = files_data[i]["color"]

        merged = nx.compose_all(graphs)

        if config.problemType == "continuous":
            final_clusters = partition.continuous_clustering(merged, config)
        else:
            final_clusters = partition.discrete_clustering(merged, config)

        visualize.visualize_clusters(final_clusters, merged, output_file="static/graph.html", legend_entries=legend_entries)
        return

    else:
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

