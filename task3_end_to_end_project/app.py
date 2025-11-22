"""app.py

Simple Flask API to serve the Iris classifier.

Endpoints:
    GET /           -> health check
    POST /predict   -> expects JSON with iris measurements and returns prediction

Run (development):
    export FLASK_APP=app.py
    flask run --host 0.0.0.0 --port 5000

    or

    python app.py
"""

import os
from flask import Flask, request, jsonify
import joblib
import numpy as np

THIS_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(THIS_DIR, "model.joblib")

app = Flask(__name__)

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "Iris model API is running."})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    try:
        features = [
            data["sepal_length"],
            data["sepal_width"],
            data["petal_length"],
            data["petal_width"],
        ]
    except KeyError as e:
        return jsonify({"error": f"Missing key: {str(e)}"}), 400

    arr = np.array(features, dtype=float).reshape(1, -1)
    pred_class = int(model.predict(arr)[0])

    return jsonify({"prediction": pred_class})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
