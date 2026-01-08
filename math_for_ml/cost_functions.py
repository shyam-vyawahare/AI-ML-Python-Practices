"""
cost_functions.py

Unit 3: Math for Machine Learning

Objective:
- Implement common loss functions from scratch
- Understand how models measure error
- Build intuition for optimization targets
"""

import numpy as np


# -------------------------------
# 1. MEAN ABSOLUTE ERROR (MAE)
# -------------------------------

def mean_absolute_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


# -------------------------------
# 2. MEAN SQUARED ERROR (MSE)
# -------------------------------

def mean_squared_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)


# -------------------------------
# 3. ROOT MEAN SQUARED ERROR (RMSE)
# -------------------------------

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# -------------------------------
# 4. HUBER LOSS (ROBUST REGRESSION)
# -------------------------------

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta

    squared_loss = 0.5 * error ** 2
    linear_loss = delta * (np.abs(error) - 0.5 * delta)

    return np.mean(np.where(is_small_error, squared_loss, linear_loss))


# -------------------------------
# 5. BINARY CROSS ENTROPY (LOG LOSS)
# -------------------------------

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-8  # numerical stability
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    return -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )


# -------------------------------
# 6. CATEGORICAL CROSS ENTROPY
# -------------------------------

def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


# -------------------------------
# 7. ZERO-ONE LOSS (CLASSIFICATION)
# -------------------------------

def zero_one_loss(y_true, y_pred_labels):
    return np.mean(y_true != y_pred_labels)


# -------------------------------
# 8. REAL ML PATTERN
# -------------------------------

y_true = np.array([1, 0, 1, 1])
y_pred_prob = np.array([0.9, 0.2, 0.7, 0.6])
y_pred_label = np.array([1, 0, 1, 0])


mae = mean_absolute_error(y_true, y_pred_label)
mse = mean_squared_error(y_true, y_pred_prob)
bce = binary_cross_entropy(y_true, y_pred_prob)
zol = zero_one_loss(y_true, y_pred_label)


# -------------------------------
# 9. LOSS COMPARISON INSIGHT
# -------------------------------

losses = {
    "MAE": mae,
    "MSE": mse,
    "Binary Cross Entropy": bce,
    "Zero-One Loss": zol
}


# -------------------------------
# 10. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    for name, value in losses.items():
        print(f"{name}: {value:.4f}")

    print("cost_functions.py executed successfully")
