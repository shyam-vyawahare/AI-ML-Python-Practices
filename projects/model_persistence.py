"""
model_persistence.py

Unit 6: Applied AI Projects

Objective:
- Save trained ML models and preprocessors
- Load them later for inference
- Simulate real-world ML system persistence
"""

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# -------------------------------
# 1. TRAINING DATA
# -------------------------------

data = {
    "age": [21, 22, 23, 22, 24, 21, 23, 22],
    "cgpa": [8.1, 8.7, 7.9, 8.4, 9.1, 8.0, 7.8, 8.6],
    "internships": [1, 2, 0, 1, 3, 0, 0, 2],
    "placed": [0, 1, 0, 1, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

X = df.drop("placed", axis=1)
y = df["placed"]


# -------------------------------
# 2. PREPROCESSING
# -------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -------------------------------
# 3. MODEL TRAINING
# -------------------------------

model = LogisticRegression()
model.fit(X_scaled, y)


# -------------------------------
# 4. SAVE MODEL & SCALER
# -------------------------------

joblib.dump(model, "placement_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully")


# -------------------------------
# 5. LOAD MODEL & SCALER
# -------------------------------

loaded_model = joblib.load("placement_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")


# -------------------------------
# 6. INFERENCE USING LOADED MODEL
# -------------------------------

def predict_placement(age, cgpa, internships):
    input_data = pd.DataFrame([{
        "age": age,
        "cgpa": cgpa,
        "internships": internships
    }])

    input_scaled = loaded_scaler.transform(input_data)
    prediction = loaded_model.predict(input_scaled)[0]
    probability = loaded_model.predict_proba(input_scaled)[0][1]

    return prediction, probability


# -------------------------------
# 7. TEST PERSISTENCE PIPELINE
# -------------------------------

if __name__ == "__main__":
    test_input = {"age": 23, "cgpa": 8.8, "internships": 2}

    pred, prob = predict_placement(
        test_input["age"],
        test_input["cgpa"],
        test_input["internships"]
    )

    status = "Placed" if pred == 1 else "Not Placed"

    print(f"\nInput: {test_input}")
    print(f"Prediction: {status}")
    print(f"Confidence: {prob:.2f}")

    print("\nmodel_persistence.py executed successfully")
