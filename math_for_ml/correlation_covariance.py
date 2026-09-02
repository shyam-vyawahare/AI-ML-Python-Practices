"""
correlation_covariance.py

Unit 3: Math for Machine Learning

Objective:
- Understand covariance and correlation
- Compute covariance matrices
- Analyze relationships between features
- Connect covariance to PCA
"""

import numpy as np


# -------------------------------
# 1. CREATE DATASET
# -------------------------------

np.random.seed(42)

study_hours = np.array([
    2, 3, 4, 5, 6, 7, 8, 9
])

exam_scores = np.array([
    50, 55, 60, 66, 70, 78, 84, 90
])

sleep_hours = np.array([
    8, 7, 7, 6, 7, 6, 6, 5
])

print("Study Hours:")
print(study_hours)

print("\nExam Scores:")
print(exam_scores)

print("\nSleep Hours:")
print(sleep_hours)


# -------------------------------
# 2. COVARIANCE
# -------------------------------

cov_study_score = np.cov(
    study_hours,
    exam_scores,
    ddof=1
)[0, 1]

print("\nCovariance:")
print(
    f"Study Hours vs Exam Scores: "
    f"{cov_study_score:.4f}"
)


# -------------------------------
# 3. CORRELATION
# -------------------------------

corr_study_score = np.corrcoef(
    study_hours,
    exam_scores
)[0, 1]

print("\nCorrelation:")
print(
    f"Study Hours vs Exam Scores: "
    f"{corr_study_score:.4f}"
)


# -------------------------------
# 4. NEGATIVE CORRELATION
# -------------------------------

corr_sleep_score = np.corrcoef(
    sleep_hours,
    exam_scores
)[0, 1]

print(
    "\nSleep Hours vs Exam Scores:"
)

print(
    f"Correlation: "
    f"{corr_sleep_score:.4f}"
)


# -------------------------------
# 5. MANUAL CORRELATION
# -------------------------------

study_mean = np.mean(study_hours)
score_mean = np.mean(exam_scores)

numerator = np.sum(
    (study_hours - study_mean)
    *
    (exam_scores - score_mean)
)

denominator = np.sqrt(
    np.sum(
        (study_hours - study_mean) ** 2
    )
    *
    np.sum(
        (exam_scores - score_mean) ** 2
    )
)

manual_correlation = (
    numerator / denominator
)

print("\nManual Pearson Correlation:")
print(
    f"{manual_correlation:.4f}"
)


# -------------------------------
# 6. COVARIANCE MATRIX
# -------------------------------

features = np.column_stack(
    (
        study_hours,
        exam_scores,
        sleep_hours
    )
)

covariance_matrix = np.cov(
    features,
    rowvar=False
)

print("\nCovariance Matrix:")
print(
    np.round(
        covariance_matrix,
        4
    )
)


# -------------------------------
# 7. CORRELATION MATRIX
# -------------------------------

correlation_matrix = np.corrcoef(
    features,
    rowvar=False
)

print("\nCorrelation Matrix:")
print(
    np.round(
        correlation_matrix,
        4
    )
)


# -------------------------------
# 8. STANDARDIZE FEATURES
# -------------------------------

means = np.mean(
    features,
    axis=0
)

stds = np.std(
    features,
    axis=0,
    ddof=1
)

standardized = (
    features - means
) / stds

print("\nStandardized Features:")
print(
    np.round(
        standardized,
        3
    )
)


# -------------------------------
# 9. COVARIANCE OF STANDARDIZED DATA
# -------------------------------

standardized_covariance = np.cov(
    standardized,
    rowvar=False
)

print(
    "\nCovariance Matrix "
    "After Standardization:"
)

print(
    np.round(
        standardized_covariance,
        4
    )
)


# -------------------------------
# 10. PCA CONNECTION
# -------------------------------

eigenvalues, eigenvectors = np.linalg.eig(
    standardized_covariance
)

sorted_indices = np.argsort(
    eigenvalues
)[::-1]

eigenvalues = eigenvalues[
    sorted_indices
]

eigenvectors = eigenvectors[
    :,
    sorted_indices
]

print("\nPCA Connection:")

print("Eigenvalues:")
print(
    np.round(
        eigenvalues,
        4
    )
)

print("\nPrincipal Directions:")
print(
    np.round(
        eigenvectors,
        4
    )
)


# -------------------------------
# 11. INTERPRETATION
# -------------------------------

print("\nInterpretation:")

print(
    "- Positive covariance → "
    "variables tend to increase together"
)

print(
    "- Negative covariance → "
    "one variable tends to increase as another decreases"
)

print(
    "- Correlation → "
    "standardized measure of linear relationship"
)

print(
    "- Correlation ranges from -1 to +1"
)

print(
    "- PCA uses the covariance structure "
    "to find principal directions"
)


# -------------------------------
# 12. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print(
        "\ncorrelation_covariance.py "
        "executed successfully"
)
