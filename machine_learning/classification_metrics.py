"""
model_evaluation.py

Unit 4: Machine Learning

Objective:
- Learn common classification evaluation metrics
- Understand confusion matrix components
- Compare different evaluation metrics
- Understand probability thresholds
"""

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# -------------------------------
# 1. LOAD DATASET
# -------------------------------

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Shape:")
print(X.shape)


# -------------------------------
# 2. TRAIN / TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------------
# 3. CREATE ML PIPELINE
# -------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])


# -------------------------------
# 4. TRAIN MODEL
# -------------------------------

model.fit(X_train, y_train)


# -------------------------------
# 5. PREDICTIONS
# -------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# -------------------------------
# 6. BASIC METRICS
# -------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# -------------------------------
# 7. DISPLAY METRICS
# -------------------------------

print("\nModel Evaluation:")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# -------------------------------
# 8. CONFUSION MATRIX
# -------------------------------

matrix = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(matrix)


# -------------------------------
# 9. CLASSIFICATION REPORT
# -------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names
    )
)


# -------------------------------
# 10. THRESHOLD EXPERIMENT
# -------------------------------

thresholds = [0.3, 0.5, 0.7]

print("\nThreshold Experiment:")

for threshold in thresholds:

    threshold_predictions = (
        y_probability >= threshold
    ).astype(int)

    threshold_precision = precision_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    threshold_recall = recall_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    print(
        f"Threshold: {threshold:.1f} | "
        f"Precision: {threshold_precision:.4f} | "
        f"Recall: {threshold_recall:.4f}"
    )


# -------------------------------
# 11. METRIC INTERPRETATION
# -------------------------------

print("\nMetric Interpretation:")

print("- Accuracy  -> Overall correct predictions")
print("- Precision -> How many predicted positives were correct")
print("- Recall    -> How many actual positives were detected")
print("- F1 Score  -> Balance between precision and recall")
print("- ROC-AUC   -> Ranking quality across thresholds")


# -------------------------------
# 12. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print(
        "\nmodel_evaluation.py "
        "executed successfully"
)
