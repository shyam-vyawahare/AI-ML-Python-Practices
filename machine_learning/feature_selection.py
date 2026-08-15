"""
feature_selection.py

Unit 4: Machine Learning

Objective:
- Learn feature selection techniques
- Remove irrelevant or low-information features
- Compare statistical feature selection methods
"""

import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    VarianceThreshold
)


# -------------------------------
# 1. LOAD DATASET
# -------------------------------

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target

print("Original Dataset Shape:")
print(X.shape)


# -------------------------------
# 2. CORRELATION ANALYSIS
# -------------------------------

correlations = X.corrwith(
    pd.Series(y)
).abs().sort_values(
    ascending=False
)

print("\nTop Features by Correlation:")
print(correlations.head(10))


# -------------------------------
# 3. VARIANCE THRESHOLD
# -------------------------------

variance_selector = VarianceThreshold(
    threshold=0.01
)

X_variance = variance_selector.fit_transform(X)

selected_variance_features = X.columns[
    variance_selector.get_support()
]

print("\nAfter Variance Filtering:")
print(f"Features Remaining: {X_variance.shape[1]}")

print("\nSelected Features:")
print(selected_variance_features.tolist())


# -------------------------------
# 4. SELECT K BEST - ANOVA
# -------------------------------

k = 10

kbest_selector = SelectKBest(
    score_func=f_classif,
    k=k
)

X_kbest = kbest_selector.fit_transform(
    X,
    y
)

selected_kbest_features = X.columns[
    kbest_selector.get_support()
]

print("\nTop Features Using ANOVA:")
print(selected_kbest_features.tolist())


# -------------------------------
# 5. FEATURE SCORES
# -------------------------------

feature_scores = pd.DataFrame({
    "Feature": X.columns,
    "Score": kbest_selector.scores_
})

feature_scores = feature_scores.sort_values(
    "Score",
    ascending=False
)

print("\nTop 10 ANOVA Feature Scores:")
print(feature_scores.head(10))


# -------------------------------
# 6. MUTUAL INFORMATION
# -------------------------------

mi_selector = SelectKBest(
    score_func=mutual_info_classif,
    k=10
)

X_mi = mi_selector.fit_transform(
    X,
    y
)

selected_mi_features = X.columns[
    mi_selector.get_support()
]

print("\nTop Features Using Mutual Information:")
print(selected_mi_features.tolist())


# -------------------------------
# 7. COMPARE METHODS
# -------------------------------

print("\nFeature Selection Comparison:")

print(
    f"Original Features      : {X.shape[1]}"
)

print(
    f"Variance Selected      : {X_variance.shape[1]}"
)

print(
    f"ANOVA Selected         : {X_kbest.shape[1]}"
)

print(
    f"Mutual Information    : {X_mi.shape[1]}"
)


# -------------------------------
# 8. WHY FEATURE SELECTION?
# -------------------------------

print("\nBenefits:")
print("- Reduces unnecessary features")
print("- Can improve model generalization")
print("- Reduces training time")
print("- Makes models easier to interpret")
print("- Can reduce noise and overfitting")


# -------------------------------
# 9. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\nfeature_selection.py executed successfully")
