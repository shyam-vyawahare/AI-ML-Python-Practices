"""
hyperparameter_tuning.py

Unit 4: Machine Learning

Objective:
- Learn GridSearchCV
- Learn RandomizedSearchCV
- Find the best hyperparameters for a model
"""

from scipy.stats import randint
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV
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

model = RandomForestClassifier(random_state=42)

# -------------------------------
# 3. GRID SEARCH
# -------------------------------

grid_params = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 5, 7, None],
    "criterion": ["gini", "entropy"]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=grid_params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X, y)

print("========== Grid Search ==========")
print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Accuracy:")
print(f"{grid_search.best_score_:.4f}")

# -------------------------------
# 4. RANDOMIZED SEARCH
# -------------------------------

random_params = {
    "n_estimators": randint(50, 200),
    "max_depth": [None, 3, 5, 7, 10],
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 5)
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=random_params,
    n_iter=10,
    cv=5,
    random_state=42,
    scoring="accuracy",
    n_jobs=-1
)

random_search.fit(X, y)

print("\n========== Randomized Search ==========")
print("Best Parameters:")
print(random_search.best_params_)

print("\nBest Accuracy:")
print(f"{random_search.best_score_:.4f}")

# -------------------------------
# 5. COMPARISON
# -------------------------------

print("\nComparison")
print("Grid Search explores every parameter combination.")
print("Randomized Search samples a subset of combinations.")
print("Randomized Search is usually faster for large search spaces.")

# -------------------------------
# 6. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\nhyperparameter_tuning.py executed successfully")
