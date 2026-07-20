"""
cross_validation.py

Unit 4: Machine Learning

Objective:
- Learn K-Fold Cross Validation
- Learn Stratified K-Fold Cross Validation
- Evaluate model performance more reliably
"""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score
)

# -------------------------------
# 1. LOAD DATASET
# -------------------------------

iris = load_iris()
X = iris.data
y = iris.target

# -------------------------------
# 2. CREATE MODEL
# -------------------------------

model = LogisticRegression(max_iter=300)

# -------------------------------
# 3. K-FOLD CROSS VALIDATION
# -------------------------------

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

kfold_scores = cross_val_score(
    model,
    X,
    y,
    cv=kfold,
    scoring="accuracy"
)

print("K-Fold Cross Validation")

print("Scores:", kfold_scores)
print("Average Accuracy:", kfold_scores.mean())

# -------------------------------
# 4. STRATIFIED K-FOLD
# -------------------------------

stratified = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

stratified_scores = cross_val_score(
    model,
    X,
    y,
    cv=stratified,
    scoring="accuracy"
)

print("\nStratified K-Fold")

print("Scores:", stratified_scores)
print("Average Accuracy:", stratified_scores.mean())

# -------------------------------
# 5. COMPARISON
# -------------------------------

print("\nComparison")

print(f"K-Fold Mean Accuracy       : {kfold_scores.mean():.4f}")
print(f"Stratified Mean Accuracy   : {stratified_scores.mean():.4f}")

# -------------------------------
# 6. WHY CROSS VALIDATION?
# -------------------------------

print("\nBenefits")

print("- More reliable model evaluation")
print("- Uses the entire dataset")
print("- Reduces bias from one train-test split")
print("- Standard practice in machine learning")

# -------------------------------
# 7. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\ncross_validation.py executed successfully")
