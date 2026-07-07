"""
Flask API for the Insurance Cost Prediction model.

Run with:  python flask_api.py
Then POST to http://localhost:5000/predict

Example request body (JSON):
{
  "Age": 35, "Diabetes": 0, "BloodPressureProblems": 1, "AnyTransplants": 0,
  "AnyChronicDiseases": 0, "Height": 170, "Weight": 75, "KnownAllergies": 0,
  "HistoryOfCancerInFamily": 0, "NumberOfMajorSurgeries": 1
}
"""
import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_DIR = "../models"
model = joblib.load(f"{MODEL_DIR}/best_model.pkl")
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
FEATURES = joblib.load(f"{MODEL_DIR}/feature_list.pkl")
USES_SCALED = joblib.load(f"{MODEL_DIR}/uses_scaled_input.pkl")

REQUIRED_RAW_FIELDS = [
    "Age", "Diabetes", "BloodPressureProblems", "AnyTransplants",
    "AnyChronicDiseases", "Height", "Weight", "KnownAllergies",
    "HistoryOfCancerInFamily", "NumberOfMajorSurgeries",
]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    missing = [f for f in REQUIRED_RAW_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        height = float(data["Height"])
        weight = float(data["Weight"])
        bmi = weight / ((height / 100) ** 2)
        total_conditions = sum(
            int(data[f]) for f in [
                "Diabetes", "BloodPressureProblems", "AnyTransplants",
                "AnyChronicDiseases", "KnownAllergies", "HistoryOfCancerInFamily",
            ]
        )
        row = {**data, "BMI": bmi, "TotalConditions": total_conditions}
        X = np.array([[float(row[f]) for f in FEATURES]])
    except (ValueError, KeyError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    X_in = scaler.transform(X) if USES_SCALED else X
    pred = float(model.predict(X_in)[0])
    pred = float(np.clip(pred, 15000, 40000))

    return jsonify({
        "predicted_premium_inr": round(pred, 2),
        "bmi": round(bmi, 2),
        "total_risk_conditions": total_conditions,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
