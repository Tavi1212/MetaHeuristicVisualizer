import os
from flask import Flask, request, jsonify, session, render_template, send_from_directory, redirect, url_for, send_file
from scripts import config as cnf
from scripts.structures import AdvancedOptions
from scripts.structures import ConfigData

# Initialize Flask app and secret key for session management
app = Flask(__name__)
app.secret_key = os.urandom(24)


# Route: Generate visualization after submitting config
@app.route("/generate_visualization", methods=['POST'])
def generate_visualization():
    try:
        # Load config, advanced settings, and algorithm file data
        config, advanced, files_data = cnf.load_config_and_files()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Save uploaded algorithm metadata in session
    session["algorithms"] = files_data

    # Build STN graphs from solution data
    graphs = cnf.build_graphs(files_data, advanced.nr_of_runs, advanced.best_solution, config.objectiveType)

    # Apply clustering/partitioning and generate visualization output
    cnf.apply_partition_and_visualize(graphs, config, advanced, files_data)

    return jsonify({"success": True})


# Route: Receive submitted form config from frontend
@app.route("/submit", methods=['POST'])
def submit_config():
    form = request.form
    print("Received form data:", dict(form))

    # Extract high-level problem settings
    objective_type = form.get("objective_type", "minimization")
    problem_type = "discrete" if form.get("problem_type") == "discrete" else "continuous"

    # Determine which strategy the user selected
    strat_key = form.get("strategy")
    if strat_key == "shannon":
        partition_strategy = "shannon"
    elif strat_key == "clustering":
        partition_strategy = "clustering"
    else:
        partition_strategy = "partitioning"

    # Extract advanced (optional) UI settings
    advanced_settings = AdvancedOptions(
        best_solution=form.get("best_solution", ""),
        nr_of_runs=int(form.get("nr_of_runs") or -1),
        vertex_size=float(form.get("vertex_size") or -1),
        arrow_size=float(form.get("arrow_size") or -1),
        tree_layout="tree_layout" in form
    )

    # Extract core config (partitioning and clustering settings)
    configData = ConfigData(
        problemType=problem_type,
        objectiveType=objective_type,
        partitionStrategy=partition_strategy,

        dPartitioning=int(form.get("discrete_partitioning") or 0),
        dCSize=int(form.get("discrete_cluster_size") or 50),
        dVSize=int(form.get("discrete_volume_size") or 25),
        dDistance=form.get("discrete_distance_metric", "hamming"),

        cMinBound=float(form.get("continuous_min_bound") or -100),
        cMaxBound=float(form.get("continuous_max_bound") or 100),
        cDimension=int(form.get("continuous_dimensions") or 3),
        cHypercube=int(form.get("continuous_hypercube") or 0),
        cClusterSize=int(form.get("continuous_cluster_size") or 50),
        cVolumeSize=int(form.get("continuous_volume_size") or 50),
        cDistance=form.get("continuous_distance_metric", "euclidean")
    )

    # Save both objects into session to be reused
    session["config_data"] = configData.toDict()
    session["advanced_settings"] = advanced_settings.to_dict()
    return jsonify({"status": "ok"})


# Route: Load default layout-based graph visualization
@app.route("/display_graph")
def display_graph():
    return send_from_directory("static","graph_fr.html")


# Route: Load specific graph layout by name
@app.route("/display_graph/<layout>")
def display_graph_layout(layout):
    if layout == "fr":
        return send_file("static/graph_fr.html")
    elif layout == "kk":
        return send_file("static/graph_kk.html")
    else:
        return "Unknown layout", 400


# Route: Main landing page
@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template("index.html")


# Main driver
if __name__ == '__main__':
    app.run(debug=True)
