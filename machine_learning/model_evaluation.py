"""
model_evaluation.py

Unit 4: Machine Learning Algorithms

Objective:
- Evaluate classification models correctly
- Understand why accuracy alone is misleading
- Build intuition for precision, recall, and F1-score
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# -------------------------------
# 1. TRUE & PREDICTED LABELS
# -------------------------------

# Example: placement prediction
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])


# -------------------------------
# 2. CONFUSION MATRIX
# -------------------------------

cm = confusion_matrix(y_true, y_pred)

tn, fp, fn, tp = cm.ravel()


# -------------------------------
# 3. BASIC METRICS
# -------------------------------

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)


# -------------------------------
# 4. MANUAL METRIC COMPUTATION
# -------------------------------

manual_accuracy = (tp + tn) / (tp + tn + fp + fn)
manual_precision = tp / (tp + fp) if (tp + fp) != 0 else 0
manual_recall = tp / (tp + fn) if (tp + fn) != 0 else 0


# -------------------------------
# 5. WHY ACCURACY CAN LIE
# -------------------------------

# Highly imbalanced dataset
y_true_imbalanced = np.array([0, 0, 0, 0, 0, 0, 1])
y_pred_imbalanced = np.array([0, 0, 0, 0, 0, 0, 0])

accuracy_imbalanced = accuracy_score(y_true_imbalanced, y_pred_imbalanced)
recall_imbalanced = recall_score(y_true_imbalanced, y_pred_imbalanced, zero_division=0)


# -------------------------------
# 6. OVERFITTING vs UNDERFITTING (SIMULATION)
# -------------------------------

train_accuracy = 0.98
test_accuracy = 0.75

model_status = (
    "Overfitting" if train_accuracy - test_accuracy > 0.15
    else "Good Fit"
)


# -------------------------------
# 7. METRICS SUMMARY
# -------------------------------

metrics_summary = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "Imbalanced Accuracy": accuracy_imbalanced,
    "Imbalanced Recall": recall_imbalanced,
    "Model Status": model_status
}


# -------------------------------
# 8. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("Confusion Matrix:")
    print(cm)
    print("\nMetrics Summary:")
    for key, value in metrics_summary.items():
        print(f"{key}: {value}")

    print("\nmodel_evaluation.py executed successfully")
