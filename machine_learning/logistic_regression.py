"""
logistic_regression.py

Unit 4: Machine Learning Algorithms

Objective:
- Implement Logistic Regression from scratch
- Understand sigmoid, probability & decision boundaries
- Compare with scikit-learn implementation
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# -------------------------------
# 1. SIGMOID FUNCTION
# -------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# -------------------------------
# 2. BINARY CROSS ENTROPY LOSS
# -------------------------------

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )


# -------------------------------
# 3. LOGISTIC REGRESSION (SCRATCH)
# -------------------------------

class LogisticRegressionScratch:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):
            linear_output = X @ self.w + self.b
            y_pred = sigmoid(linear_output)

            dw = (1 / n_samples) * (X.T @ (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


# -------------------------------
# 4. CREATE DATASET
# -------------------------------

# Simple binary classification dataset
X = np.array([
    [21, 8.1],
    [22, 8.7],
    [23, 7.9],
    [24, 9.1],
    [22, 8.4],
    [25, 9.0]
])

y = np.array([0, 1, 0, 1, 1, 1])


# -------------------------------
# 5. TRAIN SCRATCH MODEL
# -------------------------------

scratch_model = LogisticRegressionScratch(lr=0.1, epochs=1000)
scratch_model.fit(X, y)

y_pred_scratch = scratch_model.predict(X)


# -------------------------------
# 6. TRAIN SCIKIT-LEARN MODEL
# -------------------------------

sk_model = LogisticRegression()
sk_model.fit(X, y)

y_pred_sklearn = sk_model.predict(X)


# -------------------------------
# 7. EVALUATION
# -------------------------------

accuracy_scratch = accuracy_score(y, y_pred_scratch)
accuracy_sklearn = accuracy_score(y, y_pred_sklearn)


# -------------------------------
# 8. COMPARISON
# -------------------------------

results = {
    "Scratch Accuracy": accuracy_scratch,
    "Sklearn Accuracy": accuracy_sklearn,
    "Scratch Weights": scratch_model.w,
    "Scratch Bias": scratch_model.b,
    "Sklearn Weights": sk_model.coef_,
    "Sklearn Bias": sk_model.intercept_
}


# -------------------------------
# 9. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    for key, value in results.items():
        print(f"{key}: {value}")

    print("logistic_regression.py executed successfully")
