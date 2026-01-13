"""
student_placement_prediction.py

Unit 6: Applied AI Projects

Project:
- Predict whether a student gets placed based on academic features

Pipeline:
- Data creation
- Feature engineering
- Train / test split
- Model training
- Evaluation
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -------------------------------
# 1. DATASET (SIMULATED REALISTIC)
# -------------------------------

data = {
    "age": [21, 22, 23, 22, 24, 21, 23, 22, 25, 24, 23, 21],
    "cgpa": [8.1, 8.7, 7.9, 8.4, 9.1, 8.0, 7.8, 8.6, 9.0, 8.9, 7.5, 8.3],
    "internships": [1, 2, 0, 1, 3, 0, 0, 2, 2, 3, 0, 1],
    "placed": [0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)


# -------------------------------
# 2. FEATURE / LABEL SPLIT
# -------------------------------

X = df.drop("placed", axis=1)
y = df["placed"]


# -------------------------------
# 3. TRAIN / TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# -------------------------------
# 4. FEATURE SCALING
# -------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------------------
# 5. MODEL TRAINING
# -------------------------------

model = LogisticRegression()
model.fit(X_train_scaled, y_train)


# -------------------------------
# 6. PREDICTION
# -------------------------------

y_pred = model.predict(X_test_scaled)


# -------------------------------
# 7. EVALUATION
# -------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}


# -------------------------------
# 8. MODEL INTERPRETATION
# -------------------------------

coefficients = pd.Series(
    model.coef_[0],
    index=X.columns
).sort_values(ascending=False)


# -------------------------------
# 9. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("Student Placement Prediction Results\n")

    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

    print("\nFeature Importance (Logistic Coefficients):")
    print(coefficients)

    print("\nstudent_placement_prediction.py executed successfully")
