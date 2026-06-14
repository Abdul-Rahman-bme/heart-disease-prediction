from pathlib import Path
import joblib
import pandas as pd


# -----------------------------
# 1. Define project paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "logistic_regression_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "standard_scaler.pkl"


# -----------------------------
# 2. Load saved model and scaler
# -----------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# -----------------------------
# 3. Define feature columns
# -----------------------------
continuous_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

all_features = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


# -----------------------------
# 4. Example patient data
# -----------------------------
# This must follow the same column order used during training.
sample_patient = {
    "age": 58,
    "sex": 1,
    "cp": 4,
    "trestbps": 140,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 120,
    "exang": 1,
    "oldpeak": 2.3,
    "slope": 2,
    "ca": 1,
    "thal": 7
}

patient_df = pd.DataFrame([sample_patient], columns=all_features)


# -----------------------------
# 5. Scale only continuous features
# -----------------------------
patient_scaled = patient_df.copy()
patient_scaled[continuous_features] = scaler.transform(patient_df[continuous_features])


# -----------------------------
# 6. Predict
# -----------------------------
disease_probability = model.predict_proba(patient_scaled)[0][1]

threshold = 0.5
prediction = 1 if disease_probability >= threshold else 0


# -----------------------------
# 7. Show result
# -----------------------------
print("Heart Disease Prediction")
print("------------------------")
print(f"Disease probability: {disease_probability:.2f}")
print(f"Decision threshold: {threshold}")

if prediction == 1:
    print("Prediction: Heart disease present")
else:
    print("Prediction: No heart disease")