import os
import networkx as nx
from flask import Flask, request, jsonify, session, render_template, send_from_directory
from scripts.config import get_config_from_session, get_upload_file_data, get_advanced_from_session
from scripts.partition import apply_partitioning_from_config, continuous_clustering, discrete_clustering, cont_standard_partitioning
from scripts.visualize import tag_graph_origin, visualize_stn
from scripts.create import create_stn
from scripts.structures import AdvancedSettings
from scripts.structures import ConfigData


app = Flask(__name__)
app.secret_key = os.urandom(24)


@app.route("/generate_visualization", methods=['POST'])
def generate_visualization():
    try:
        config =   get_config_from_session()
        advanced = get_advanced_from_session()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    files_data = get_upload_file_data(request)

    session["algorithms"] = files_data



    graphs = []
    for algo in files_data:
        G = create_stn(algo["path"])
        tag_graph_origin(G, algo["color"])
        graphs.append(G)

    merged_graph = nx.compose_all(graphs)

    # 📊 Apply clustering if needed
    if config.partitionStrategy == "clustering":
        if config.problemType == "continuous":
            clusters = continuous_clustering(merged_graph, config)
        else:
            clusters = discrete_clustering(merged_graph, config)

        for cluster_id, nodes in enumerate(clusters):
            for node in nodes:
                if node in merged_graph:
                    merged_graph.nodes[node]["cluster"] = cluster_id

        graphs = [merged_graph]
    elif config.partitionStrategy == "partitioning":
        merged_graph = nx.compose_all(graphs)

        H = cont_standard_partitioning(
            merged_graph,
            hypercube_factor=config.cHypercube,
            min_bound=config.cMinBound,
            max_bound=config.cMaxBound
        )
        graphs = [H]

    else:
        # standard partitioning applied per graph
        graphs = [apply_partitioning_from_config(G, config) for G in graphs]

    legend_entries = {algo["color"]: algo["name"] for algo in files_data}
    visualize_stn(graphs, advanced, config, output_file="static/graph.html", minmax=config.objectiveType, legend_entries=legend_entries)

    return jsonify({"success": True})


@app.route("/submit", methods=['POST'])
def submit_config():
    form = request.form
    print("Received form data:", dict(form))

    objective_type = form.get("objective_type", "minimization")
    problem_type = "discrete" if form.get("problem_type") == "discrete" else "continuous"

    strat_key = form.get("strategy")
    if strat_key == "shannon":
        partition_strategy = "shannon"
    elif strat_key == "clustering":
        partition_strategy = "clustering"
    else:
        partition_strategy = "partitioning"

    advanced_settings = AdvancedSettings(
        best_solution=form.get("best_solution", ""),
        nr_of_runs=int(form.get("nr_of_runs") or -1),
        vertex_size=int(form.get("vertex_size") or -1),
        arrow_size=int(form.get("arrow_size") or -1),
        tree_layout="tree_layout" in form
    )

    configData = ConfigData(
        problemType=problem_type,
        objectiveType=objective_type,
        partitionStrategy=partition_strategy,
        dPartitioning=int(form.get("discrete_partitioning") or 0),
        dCSize=int(form.get("disc_cluster_size") or 50),
        dVSize=int(form.get("disc_partitioning") or 25),
        dDistance=form.get("disc_distance_metric", "hamming"),
        cMinBound=int(form.get("min_bound") or -100),
        cMaxBound=int(form.get("max_bound") or 100),
        cDimension=int(form.get("n_dimensions") or 3),
        cHypercube=int(form.get("hyper_cube") or 0),
        cClusterSize=int(form.get("cont_cluster_size") or 50),
        cVolumeSize=int(form.get("cont_volume_size") or 50),
        cDistance=form.get("cont_distance_metric", "euclidean")
    )

    session["config_data"] = configData.toDict()
    session["advanced_settings"] = advanced_settings.to_dict()
    return jsonify({"status": "ok"})

@app.route("/display_graph")
def display_graph():
    return send_from_directory("static","graph.html")

@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template("index.html")


# Main driver
if __name__ == '__main__':
    app.run(debug=True)
