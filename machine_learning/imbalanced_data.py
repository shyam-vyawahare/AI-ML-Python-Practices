"""
imbalanced_data.py

Unit 4: Machine Learning

Objective:
- Understand class imbalance
- Compare standard and class-weighted models
- Evaluate imbalanced classification properly
"""

import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# -------------------------------
# 1. CREATE IMBALANCED DATASET
# -------------------------------

X, y = make_classification(
    n_samples=2000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    weights=[0.90, 0.10],
    random_state=42
)


# -------------------------------
# 2. CHECK CLASS DISTRIBUTION
# -------------------------------

unique_classes, class_counts = np.unique(
    y,
    return_counts=True
)

print("Class Distribution:")

for class_value, count in zip(
    unique_classes,
    class_counts
):
    percentage = count / len(y) * 100

    print(
        f"Class {class_value}: "
        f"{count} samples ({percentage:.2f}%)"
    )


# -------------------------------
# 3. TRAIN / TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------------
# 4. STANDARD MODEL
# -------------------------------

standard_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression())
])

standard_model.fit(
    X_train,
    y_train
)

standard_predictions = standard_model.predict(
    X_test
)


# -------------------------------
# 5. BALANCED MODEL
# -------------------------------

balanced_model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            class_weight="balanced"
        )
    )
])

balanced_model.fit(
    X_train,
    y_train
)

balanced_predictions = balanced_model.predict(
    X_test
)


# -------------------------------
# 6. EVALUATION FUNCTION
# -------------------------------

def evaluate_model(name, y_true, predictions):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    print(f"\n{name}")

    print("-" * len(name))

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_true,
            predictions
        )
    )


# -------------------------------
# 7. COMPARE MODELS
# -------------------------------

evaluate_model(
    "Standard Logistic Regression",
    y_test,
    standard_predictions
)

evaluate_model(
    "Class-Weighted Logistic Regression",
    y_test,
    balanced_predictions
)


# -------------------------------
# 8. CLASSIFICATION REPORT
# -------------------------------

print("\nBalanced Model Classification Report:")

print(
    classification_report(
        y_test,
        balanced_predictions
    )
)


# -------------------------------
# 9. WHY CLASS WEIGHTS?
# -------------------------------

print("\nWhy Use Class Weights?")

print("- Gives minority classes greater importance")
print("- Helps improve minority-class recall")
print("- Useful when classes are highly imbalanced")
print("- Avoids simply optimizing for the majority class")


# -------------------------------
# 10. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print(
        "\nimbalanced_data.py "
        "executed successfully"
)
