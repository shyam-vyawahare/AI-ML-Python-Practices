"""
pipeline_preprocessing.py

Unit 4: Machine Learning

Objective:
- Learn how to build a machine learning pipeline
- Apply feature scaling automatically
- Prevent data leakage
- Evaluate the pipeline using cross-validation
"""

from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. LOAD DATASET
# -------------------------------

wine = load_wine()
X = wine.data
y = wine.target

print(f"Dataset Shape: {X.shape}")

# -------------------------------
# 2. CREATE PIPELINE
# -------------------------------

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# -------------------------------
# 3. TRAIN PIPELINE
# -------------------------------

pipeline.fit(X, y)

print("\nPipeline trained successfully.")

# -------------------------------
# 4. CROSS VALIDATION
# -------------------------------

scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\nCross Validation Scores:")
print(scores)

print(f"\nAverage Accuracy: {scores.mean():.4f}")

# -------------------------------
# 5. MAKE PREDICTIONS
# -------------------------------

sample = X[:5]

predictions = pipeline.predict(sample)

print("\nPredictions:")
print(predictions)

print("\nActual Labels:")
print(y[:5])

# -------------------------------
# 6. WHY USE PIPELINES?
# -------------------------------

print("\nBenefits of Pipelines:")
print("- Prevents data leakage")
print("- Automates preprocessing")
print("- Makes code cleaner")
print("- Easy integration with GridSearchCV")
print("- Standard practice in production ML")

# -------------------------------
# 7. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\npipeline_preprocessing.py executed successfully")
