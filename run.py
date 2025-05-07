from flask import Flask, render_template, request, url_for, jsonify
from scripts import settings
from scripts import create
from scripts.create import createSTN
from scripts.structures import ConfigData

app = Flask(__name__)

configData = ConfigData()

@app.route("/", methods=['GET', 'POST'])
def index():
    pdf = False  # By default, no PDF is shown

    if request.method == 'POST':
        if request.form.get('action') == 'load_pdf':
            pdf = True  # Set the variable to display the PDF
        if request.form.get('action') == "select_file":
            userFile = request.files["inputfile"]
            createSTN(userFile)
            pdf = True

    if pdf != True:
        return render_template("index.html", pdf=pdf)
    return render_template("index.html", pdf=pdf)


@app.route("/create_graphs", methods=['POST'])
def create_graphs():
    global graphs
    algorithm_data = request.json.get("algorithms", [])

    for algo in algorithm_data:
        name = algo.get("name")
        file = algo.get("file")
        color = algo.get("color")

        # Create a graph for each algorithm
        #graphs[name] = createGraphFromAlgorithm(file, color)

    return jsonify({"status": "success", "graphs": list(graphs.keys())})

@app.route("/submit", methods=['POST'])
def submit_config():
    global configData

    form = request.form
    configData = ConfigData(
        problemType="Discrete Problem" if form.get("choice2") == "on" else "Continuous Problem",
        partitionStrategy=(
            "Shannon entropy" if form.get("choice3") == "on"
            else "Aglomerative clustering" if form.get("choice3") == "off"
            else form.get("choice4")  # fallback for continuous
        ),
        dPartitioning=int(form.get("disc_partitioning", 0) or 0),
        dCSize=int(form.get("disc_cluster_size", 50) or 50),
        dVSize=int(form.get("disc_partitioning", 25) or 25),
        dDistance=form.get("distance_metric", "hamming"),
        dMinBound=int(form.get("min_bound", -100) or -100),
        cMaxBound=int(form.get("max_bound", 100) or 100),
        cDimension=int(form.get("n_dimensions", 3) or 3),
        cHypercube=int(form.get("hyper_cube", 0) or 0),
        cClusterSize=int(form.get("cont_cluster_size", 50) or 50),
        cVolumeSize=int(form.get("cont_volume_size", 50) or 50),
        cDistance=form.get("distance_metric", "euclidian")
    )

    print(vars(configData))
    return "Config recieved"

# Main driver
if __name__ == '__main__':
    app.run(debug=True)
