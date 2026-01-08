"""
gradients.py

Unit 3: Math for Machine Learning

Objective:
- Understand gradients numerically and analytically
- Implement gradient descent from scratch
- Build intuition for optimization in ML models
"""

import numpy as np


# -------------------------------
# 1. SIMPLE FUNCTION
# -------------------------------

def f(x):
    return x ** 2


# -------------------------------
# 2. ANALYTICAL GRADIENT
# -------------------------------

def df_dx(x):
    return 2 * x


# -------------------------------
# 3. NUMERICAL GRADIENT
# -------------------------------

def numerical_gradient(func, x, h=1e-5):
    return (func(x + h) - func(x - h)) / (2 * h)


# -------------------------------
# 4. GRADIENT COMPARISON
# -------------------------------

x = 3.0
analytical = df_dx(x)
numerical = numerical_gradient(f, x)


# -------------------------------
# 5. GRADIENT DESCENT (1D)
# -------------------------------

def gradient_descent(start_x, lr=0.1, steps=50):
    x = start_x
    history = []

    for _ in range(steps):
        grad = df_dx(x)
        x = x - lr * grad
        history.append(x)

    return x, history


# -------------------------------
# 6. COST FUNCTION (MSE)
# -------------------------------

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# -------------------------------
# 7. GRADIENT FOR LINEAR REGRESSION
# -------------------------------

def compute_gradients(X, y, w, b):
    n = len(y)
    y_pred = X @ w + b

    dw = (-2 / n) * (X.T @ (y - y_pred))
    db = (-2 / n) * np.sum(y - y_pred)

    return dw, db


# -------------------------------
# 8. GRADIENT DESCENT (LINEAR REG)
# -------------------------------

def train_linear_regression(X, y, lr=0.01, epochs=100):
    w = np.zeros(X.shape[1])
    b = 0

    for _ in range(epochs):
        dw, db = compute_gradients(X, y, w, b)
        w -= lr * dw
        b -= lr * db

    return w, b


# -------------------------------
# 9. SAMPLE TRAINING DATA
# -------------------------------

X = np.array([
    [1],
    [2],
    [3],
    [4]
])

y = np.array([3, 5, 7, 9])  # y = 2x + 1


# -------------------------------
# 10. TRAIN MODEL
# -------------------------------

weights, bias = train_linear_regression(X, y)


# -------------------------------
# 11. LEARNING RATE EFFECT
# -------------------------------

def simulate_learning_rates():
    rates = [0.001, 0.01, 0.1]
    results = {}

    for lr in rates:
        w, b = train_linear_regression(X, y, lr=lr, epochs=100)
        results[lr] = (w, b)

    return results


# -------------------------------
# 12. CONVERGENCE CHECK
# -------------------------------

predictions = X @ weights + bias
final_loss = mse(y, predictions)


# -------------------------------
# 13. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print(f"Analytical gradient: {analytical}")
    print(f"Numerical gradient: {numerical}")

    print(f"Trained weight: {weights}")
    print(f"Trained bias: {bias}")
    print(f"Final MSE Loss: {final_loss:.4f}")

    print("gradients.py executed successfully")
