from flask import Flask, render_template, request, url_for
from scripts import settings
from scripts import create
from scripts.create import createSTN

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    pdf = False  # By default, no PDF is shown

    if request.method == 'POST':
        if request.form.get('action') == 'load_pdf':
            pdf = True  # Set the variable to display the PDF
        if request.form.get('action') == "select_file":
            userFile = request.files["inputfile"]
            createSTN("input/DE_Rana_3D.txt")
            pdf = True


    if pdf != True:
        return render_template("index.html", pdf=pdf)
    return render_template("index.html", pdf=pdf)

# Main driver
if __name__ == '__main__':
    app.run(debug=True)
