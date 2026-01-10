"""
loss_functions.py

Unit 5: Deep Learning Fundamentals

Objective:
- Implement common deep learning loss functions
- Understand how loss guides neural network learning
- Build intuition for classification vs regression losses
"""

import numpy as np


# -------------------------------
# 1. MEAN SQUARED ERROR (MSE)
# -------------------------------

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)


# -------------------------------
# 2. BINARY CROSS ENTROPY (BCE)
# -------------------------------

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    return -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )


def bce_derivative(y_true, y_pred):
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    return (y_pred - y_true) / (y_pred * (1 - y_pred) + epsilon)


# -------------------------------
# 3. CATEGORICAL CROSS ENTROPY
# -------------------------------

def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


# -------------------------------
# 4. SOFTMAX + CROSS ENTROPY (STABLE)
# -------------------------------

def softmax(z):
    z_stable = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def softmax_cross_entropy_loss(y_true, logits):
    probs = softmax(logits)
    return categorical_cross_entropy(y_true, probs)


# -------------------------------
# 5. REAL DL EXAMPLES
# -------------------------------

# Regression example
y_true_reg = np.array([[3.0], [5.0], [7.0]])
y_pred_reg = np.array([[2.5], [5.5], [6.8]])

mse_loss = mean_squared_error(y_true_reg, y_pred_reg)


# Binary classification example
y_true_bin = np.array([[1], [0], [1], [1]])
y_pred_bin = np.array([[0.9], [0.2], [0.7], [0.6]])

bce_loss = binary_cross_entropy(y_true_bin, y_pred_bin)


# Multi-class classification example
y_true_cat = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

logits = np.array([
    [2.0, 1.0, 0.1],
    [0.5, 1.5, 0.3],
    [0.2, 0.1, 2.5]
])

ce_loss = softmax_cross_entropy_loss(y_true_cat, logits)


# -------------------------------
# 6. LOSS SELECTION GUIDELINE
# -------------------------------

"""
Regression        -> Mean Squared Error
Binary Class      -> Binary Cross Entropy
Multi-Class       -> Softmax + Cross Entropy
"""


# -------------------------------
# 7. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print(f"MSE Loss: {mse_loss:.4f}")
    print(f"BCE Loss: {bce_loss:.4f}")
    print(f"Categorical CE Loss: {ce_loss:.4f}")

    print("loss_functions.py executed successfully")
