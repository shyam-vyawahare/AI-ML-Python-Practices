"""
ml_workflow.py

Unit 4: Machine Learning Algorithms

Objective:
- Implement an end-to-end ML workflow
- Understand how ML projects are structured
- Build discipline before jumping into algorithms
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# -------------------------------
# 1. PROBLEM DEFINITION
# -------------------------------
# Binary classification:
# Predict whether a student is placed based on age & CGPA


# -------------------------------
# 2. CREATE / LOAD DATASET
# -------------------------------

data = {
    "age": [21, 22, 23, 22, 24, 21, 23, 22, 25, 24],
    "cgpa": [8.1, 8.7, 7.9, 8.4, 9.1, 8.0, 7.8, 8.6, 9.0, 8.9],
    "placed": [0, 1, 0, 1, 1, 0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)


# -------------------------------
# 3. FEATURE / LABEL SPLIT
# -------------------------------

X = df[["age", "cgpa"]]
y = df["placed"]


# -------------------------------
# 4. TRAIN / TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------------
# 5. BASELINE MODEL
# -------------------------------

model = LogisticRegression()
model.fit(X_train, y_train)


# -------------------------------
# 6. PREDICTION
# -------------------------------

y_pred = model.predict(X_test)


# -------------------------------
# 7. EVALUATION
# -------------------------------

accuracy = accuracy_score(y_test, y_pred)


# -------------------------------
# 8. INTERPRETATION
# -------------------------------

coefficients = model.coef_
intercept = model.intercept_


# -------------------------------
# 9. PIPELINE SANITY CHECK
# -------------------------------

assert X_train.shape[0] == y_train.shape[0], "Training data mismatch"
assert X_test.shape[0] == y_test.shape[0], "Test data mismatch"


# -------------------------------
# 10. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("ML Workflow Executed Successfully")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Model Coefficients: {coefficients}")
    print(f"Model Intercept: {intercept}")
