from flask import Flask, render_template, request, url_for
from scripts import settings

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    pdf = False  # By default, no PDF is shown

    if request.method == 'POST':
        if request.form.get('action') == 'load_pdf':
            pdf = True  # Set the variable to display the PDF

    return render_template("index.html", pdf=pdf)

# Main driver
if __name__ == '__main__':
    app.run(debug=True)
