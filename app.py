from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, joblib, numpy as np

app = Flask(__name__)
CORS(app)  # allow frontend from other ports like :5500

# Load your trained model
try:
    with open("heart_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Model loaded with pickle")
except Exception as e:
    print("⚠️ Pickle failed:", e)
    model = joblib.load("heart_model.pkl")
    print("✅ Model loaded with joblib")

@app.route("/", methods=["GET"])
def home():
    return "✅ Flask backend is running. Use POST /predict with JSON data."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        features = [
            data["age"], data["sex"], data["cp"], data["trestbps"], data["chol"],
            data["fbs"], data["restecg"], data["thalach"], data["exang"],
            data["oldpeak"], data["slope"], data["ca"], data["thal"]
        ]
        arr = np.array(features).reshape(1, -1)
        risk = float(model.predict_proba(arr)[0, 1])
        return jsonify({"risk": risk})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
