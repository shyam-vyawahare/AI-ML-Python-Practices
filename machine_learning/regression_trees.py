"""
regression_trees.py

Unit 4: Machine Learning

Objective:
- Practice tree-based regression models
- Compare Decision Tree, Random Forest,
  and Gradient Boosting
- Analyze regression performance
"""

import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# -------------------------------
# 1. LOAD DATASET
# -------------------------------

data = load_diabetes()

X = data.data
y = data.target

feature_names = data.feature_names

print("Dataset Shape:")
print(X.shape)


# -------------------------------
# 2. TRAIN / TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------------
# 3. CREATE MODELS
# -------------------------------

models = {
    "Decision Tree": DecisionTreeRegressor(
        max_depth=4,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# -------------------------------
# 4. TRAIN & EVALUATE
# -------------------------------

results = {}

for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# -------------------------------
# 5. DISPLAY RESULTS
# -------------------------------

print("\nRegression Model Comparison:")

for name, metrics in results.items():

    print(f"\n{name}")

    print(
        f"MAE  : {metrics['MAE']:.4f}"
    )

    print(
        f"RMSE : {metrics['RMSE']:.4f}"
    )

    print(
        f"R²   : {metrics['R2']:.4f}"
    )


# -------------------------------
# 6. RANDOM FOREST FEATURE IMPORTANCE
# -------------------------------

forest = models["Random Forest"]

importance = forest.feature_importances_

ranking = sorted(
    zip(feature_names, importance),
    key=lambda item: item[1],
    reverse=True
)

print("\nRandom Forest Feature Importance:")

for feature, score in ranking:
    print(
        f"{feature:10s}: {score:.4f}"
    )


# -------------------------------
# 7. SAMPLE PREDICTIONS
# -------------------------------

sample_predictions = forest.predict(
    X_test[:5]
)

print("\nSample Predictions:")

for actual, predicted in zip(
    y_test[:5],
    sample_predictions
):
    print(
        f"Actual: {actual:.2f} | "
        f"Predicted: {predicted:.2f}"
    )


# -------------------------------
# 8. CONCEPT SUMMARY
# -------------------------------

print("\nConcept Summary:")

print(
    "- Decision Tree -> Learns non-linear decision rules"
)

print(
    "- Random Forest -> Combines many decision trees"
)

print(
    "- Gradient Boosting -> Builds trees sequentially"
)

print(
    "- Feature Importance -> Estimates feature contribution"
)


# -------------------------------
# 9. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print(
        "\nregression_trees.py "
        "executed successfully"
)
