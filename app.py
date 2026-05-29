from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load trained model
try:
    with open("../ml/model.pkl", "rb") as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

FEATURE_IMPORTANCE = {
    "glucose": 38, "bmi": 21, "age": 14,
    "diabetesPedigree": 10, "bloodPressure": 7,
    "insulin": 5, "skinThickness": 3, "pregnancies": 2,
}

# API input names (camelCase)
MODEL_FIELDS = [
    "pregnancies", "glucose", "bloodPressure",
    "skinThickness", "insulin", "bmi",
    "diabetesPedigree", "age"
]

# CSV column names (as model was trained on)
CSV_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"
]

REQUIRED_FIELDS = ["glucose", "bloodPressure", "bmi", "age"]

OPTIONAL_DEFAULTS = {
    "pregnancies": 0,
    "skinThickness": 20,
    "insulin": 79,
    "diabetesPedigree": 0.47
}

FIELD_RANGES = {
    "pregnancies":      (0, 17),
    "glucose":          (50, 300),
    "bloodPressure":    (30, 140),
    "skinThickness":    (0, 99),
    "insulin":          (0, 846),
    "bmi":              (10, 70),
    "diabetesPedigree": (0.05, 2.5),
    "age":              (1, 120),
}

def get_bmi_category(bmi):
    if bmi < 18.5: return "Underweight"
    elif bmi < 25: return "Normal"
    elif bmi < 30: return "Overweight"
    else: return "Obese"

def get_contributing_factors(data):
    factors = []

    def safe_float(key, default=0):
        try:
            val = data.get(key, default)
            return float(val) if val != "" else default
        except:
            return default

    if safe_float("glucose") > 140:
        factors.append("High glucose level detected")
    if safe_float("bmi") > 30:
        factors.append("BMI indicates obesity")
    if safe_float("age") > 45:
        factors.append("Age is a risk factor (above 45)")
    if safe_float("bloodPressure") > 90:
        factors.append("Elevated blood pressure")
    if safe_float("insulin") > 200:
        factors.append("High insulin level")
    if safe_float("diabetesPedigree") > 0.8:
        factors.append("Strong family history of diabetes")

    return factors if factors else ["No significant individual risk factors detected"]

def get_health_tips(is_diabetic, data):
    def safe_float(key, default=0):
        try:
            val = data.get(key, default)
            return float(val) if val != "" else default
        except:
            return default

    if is_diabetic:
        return [
            "Consult a doctor immediately for clinical evaluation",
            "Reduce sugar and refined carbohydrate intake",
            "Aim for at least 30 minutes of exercise daily",
            "Monitor blood glucose levels regularly",
            "Maintain a healthy weight (target BMI under 25)",
        ]
    else:
        tips = ["Maintain your healthy lifestyle"]
        if safe_float("bmi") > 25:
            tips.append("Work on reducing BMI to below 25")
        if safe_float("glucose") > 100:
            tips.append("Watch your sugar intake, glucose is borderline")
        tips.append("Schedule regular health checkups annually")
        tips.append("Stay physically active and eat a balanced diet")
        return tips


@app.route("/", methods=["GET"])
def home():
    return jsonify({"success": True, "status": "online", "message": "Diabetes Prediction API is running."})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"success": False, "message": "ML model failed to load."}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "Request body is missing."}), 400

        for field in REQUIRED_FIELDS:
            if field not in data or data[field] == "":
                return jsonify({"success": False, "message": f"'{field}' is required."}), 400

        features = {}

        for field in MODEL_FIELDS:
            value = data.get(field)
            if value == "" or value is None:
                value = OPTIONAL_DEFAULTS.get(field)
            try:
                value = float(value)
            except:
                return jsonify({"success": False, "message": f"Invalid value for '{field}'."}), 400

            min_val, max_val = FIELD_RANGES[field]
            if not (min_val <= value <= max_val):
                return jsonify({
                    "success": False,
                    "message": f"'{field}' must be between {min_val} and {max_val}."
                }), 400

            features[field] = value

        input_df = pd.DataFrame([list(features.values())], columns=CSV_COLUMNS)

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        diabetes_chance = round(probability[1] * 100, 2)
        confidence = round(abs(probability[1] - 0.5) * 2 * 100, 2)

        if diabetes_chance >= 70:
            risk_level = "High"
        elif diabetes_chance >= 40:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        return jsonify({
            "success": True,
            "isDiabetic": bool(prediction == 1),
            "diabetesChance": diabetes_chance,
            "confidence": confidence,
            "riskLevel": risk_level,
            "bmiCategory": get_bmi_category(features["bmi"]),
            "contributingFactors": get_contributing_factors(data),
            "healthTips": get_health_tips(bool(prediction == 1), data),
            "featureImportance": FEATURE_IMPORTANCE,
            "modelAccuracy": "76%",
            "message": (
                "High diabetes risk detected."
                if prediction == 1
                else "Low diabetes risk detected."
            )
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": "Internal server error.", "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)