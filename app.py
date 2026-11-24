from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

category_model = joblib.load("category_model.pkl")
priority_model = joblib.load("priority_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    complaint = data.get("complaint", "")

    if not complaint.strip():
        return jsonify({"error": "Complaint is required"}), 400

    category = category_model.predict([complaint])[0]
    priority = priority_model.predict([complaint])[0]

    return jsonify({
        "category": category,
        "priority": priority
    })

if __name__ == "__main__":
    app.run(debug=True)