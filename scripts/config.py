from werkzeug.utils import secure_filename
from scripts.structures import ConfigData, AdvancedSettings
from flask import session
import os

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