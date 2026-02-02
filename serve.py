from flask import Flask, request, jsonify
import pickle
import numpy as np


MODEL_PATH = "model.bin"

app = Flask("mental-health-predictor")


# =========================
# Load model (lazy loading)
# =========================
def load_artifact():
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    return artifact["model"], artifact["dict_vectorizer"]


# =========================
# Routes
# =========================
@app.route("/", methods=["GET"])
def root():
    return {
        "status": "OK",
        "service": "Global Mental Health & Lifestyle Predictor",
    }


@app.route("/health", methods=["GET"])
def health():
    return {"status": "OK"}


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # Lazy-load model
    if not hasattr(predict, "model"):
        predict.model, predict.dv = load_artifact()

    try:
        X = predict.dv.transform([data])
        prob = predict.model.predict_proba(X)[0, 1]
        pred = int(prob >= 0.5)

        result = {
            "probability": round(float(prob), 4),
            "prediction": pred,
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# Run server
# =========================
if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 9696))
    app.run(host="0.0.0.0", port=port, debug=False)
