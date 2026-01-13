"""
inference_pipeline.py

Unit 6: Applied AI Projects

Objective:
- Build a clean inference pipeline
- Simulate real-world ML model usage
- Separate training logic from prediction logic
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# -------------------------------
# 1. TRAINED MODEL SIMULATION
# -------------------------------

# Normally this would be loaded from disk
model = LogisticRegression()
scaler = StandardScaler()

# Training data (same structure as training phase)
train_data = {
    "age": [21, 22, 23, 22, 24, 21, 23, 22],
    "cgpa": [8.1, 8.7, 7.9, 8.4, 9.1, 8.0, 7.8, 8.6],
    "internships": [1, 2, 0, 1, 3, 0, 0, 2],
    "placed": [0, 1, 0, 1, 1, 0, 0, 1]
}

df = pd.DataFrame(train_data)

X_train = df.drop("placed", axis=1)
y_train = df["placed"]

X_train_scaled = scaler.fit_transform(X_train)
model.fit(X_train_scaled, y_train)


# -------------------------------
# 2. INFERENCE FUNCTION
# -------------------------------

def predict_placement(age, cgpa, internships):
    """
    Predict placement outcome for a new student.
    """
    input_data = pd.DataFrame([{
        "age": age,
        "cgpa": cgpa,
        "internships": internships
    }])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return prediction, probability


# -------------------------------
# 3. REAL-WORLD INPUT EXAMPLES
# -------------------------------

new_students = [
    {"age": 22, "cgpa": 8.6, "internships": 2},
    {"age": 24, "cgpa": 7.4, "internships": 0},
    {"age": 23, "cgpa": 9.0, "internships": 3}
]


# -------------------------------
# 4. RUN INFERENCE
# -------------------------------

if __name__ == "__main__":
    print("Placement Prediction Inference\n")

    for student in new_students:
        pred, prob = predict_placement(
            student["age"],
            student["cgpa"],
            student["internships"]
        )

        status = "Placed" if pred == 1 else "Not Placed"
        print(
            f"Input: {student} → Prediction: {status} "
            f"(Confidence: {prob:.2f})"
        )

    print("\ninference_pipeline.py executed successfully")
