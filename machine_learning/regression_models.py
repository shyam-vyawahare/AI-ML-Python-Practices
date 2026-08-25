"""
regression_models.py

Unit 4: Machine Learning

Objective:
- Practice common regression models
- Compare regularized linear models
- Evaluate regression performance
"""

import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
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
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),

    "Lasso Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.1))
    ])
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
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


# -------------------------------
# 5. DISPLAY RESULTS
# -------------------------------

print("\nRegression Model Comparison:")

for model_name, metrics in results.items():

    print(f"\n{model_name}")

    print(
        f"MAE  : {metrics['MAE']:.4f}"
    )

    print(
        f"MSE  : {metrics['MSE']:.4f}"
    )

    print(
        f"RMSE : {metrics['RMSE']:.4f}"
    )

    print(
        f"R²   : {metrics['R2']:.4f}"
    )


# -------------------------------
# 6. METRIC INTERPRETATION
# -------------------------------

print("\nMetric Interpretation:")

print("- MAE  -> Average absolute prediction error")
print("- MSE  -> Penalizes larger errors more heavily")
print("- RMSE -> Error in the target's original scale")
print("- R²   -> Proportion of variance explained by the model")


# -------------------------------
# 7. REGULARIZATION
# -------------------------------

print("\nRegularization:")

print("- Ridge uses L2 regularization")
print("- Lasso uses L1 regularization")
print("- Regularization can reduce overfitting")


# -------------------------------
# 8. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print(
        "\nregression_models.py "
        "executed successfully"
)
