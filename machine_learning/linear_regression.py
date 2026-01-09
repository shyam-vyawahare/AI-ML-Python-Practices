"""
linear_regression.py

Unit 4: Machine Learning Algorithms

Objective:
- Implement Linear Regression from scratch
- Use Gradient Descent for optimization
- Compare with scikit-learn implementation
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# -------------------------------
# 1. CREATE DATASET
# -------------------------------

# y = 2x + 1 (with slight noise)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 5, 7, 9, 11])


# -------------------------------
# 2. LINEAR REGRESSION (SCRATCH)
# -------------------------------

class LinearRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):
            y_pred = X @ self.w + self.b

            dw = (-2 / n_samples) * (X.T @ (y - y_pred))
            db = (-2 / n_samples) * np.sum(y - y_pred)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return X @ self.w + self.b


# -------------------------------
# 3. TRAIN SCRATCH MODEL
# -------------------------------

scratch_model = LinearRegressionScratch(lr=0.1, epochs=1000)
scratch_model.fit(X, y)

y_pred_scratch = scratch_model.predict(X)


# -------------------------------
# 4. TRAIN SCIKIT-LEARN MODEL
# -------------------------------

sk_model = LinearRegression()
sk_model.fit(X, y)

y_pred_sklearn = sk_model.predict(X)


# -------------------------------
# 5. EVALUATION
# -------------------------------

mse_scratch = mean_squared_error(y, y_pred_scratch)
mse_sklearn = mean_squared_error(y, y_pred_sklearn)


# -------------------------------
# 6. COMPARISON
# -------------------------------

results = {
    "Scratch Weights": scratch_model.w,
    "Scratch Bias": scratch_model.b,
    "Sklearn Weights": sk_model.coef_,
    "Sklearn Bias": sk_model.intercept_,
    "Scratch MSE": mse_scratch,
    "Sklearn MSE": mse_sklearn
}


# -------------------------------
# 7. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    for key, value in results.items():
        print(f"{key}: {value}")

    print("linear_regression.py executed successfully")
