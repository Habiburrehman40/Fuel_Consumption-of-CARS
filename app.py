import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

model = joblib.load("model.pkl")
with open("options.json") as f:
    OPTIONS = json.load(f)

FEATURES = OPTIONS["features"]
NUMERIC = set(OPTIONS["numeric"])


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/options")
def options():
    return jsonify(OPTIONS)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    row = {}
    for col in FEATURES:
        value = data.get(col)
        if value in (None, ""):
            return jsonify({"error": f"Please fill in: {col.title()}"}), 400
        if col in NUMERIC:
            try:
                value = float(value)
            except ValueError:
                return jsonify({"error": f"{col.title()} must be a number"}), 400
        row[col] = value

    X = pd.DataFrame([row], columns=FEATURES)
    pred = float(model.predict(X)[0])
    return jsonify({"prediction": round(max(pred, 0), 1)})


if __name__ == "__main__":
    app.run(debug=True)
