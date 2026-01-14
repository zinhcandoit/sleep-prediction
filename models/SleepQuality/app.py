from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("sleep_score_model.joblib")

FEATURES = [
    "sleep_latency_minutes",
    "wake_after_sleep_onset_minutes",
    "sleep_efficiency",
    "sleep_stage_deep",
    "sleep_stage_light",
    "sleep_stage_rem",
    "sleep_stage_awake"
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    try:
        x = np.array([[
            int(data[f]) for f in FEATURES
        ]])
    except KeyError as e:
        return jsonify({
            "error": f"Missing feature: {str(e)}"
        }), 400

    y_pred = model.predict(x)

    return jsonify({
        "sleep_score": float(y_pred[0])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
