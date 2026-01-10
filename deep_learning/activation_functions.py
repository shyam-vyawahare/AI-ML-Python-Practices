"""
activation_functions.py

Unit 5: Deep Learning Fundamentals

Objective:
- Implement common activation functions
- Understand their behavior and gradients
- Build intuition for deep network training
"""

import numpy as np


# -------------------------------
# 1. SIGMOID
# -------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    return a * (1 - a)


# -------------------------------
# 2. TANH
# -------------------------------

def tanh(z):
    return np.tanh(z)


def tanh_derivative(a):
    return 1 - a ** 2


# -------------------------------
# 3. RELU (MOST IMPORTANT)
# -------------------------------

def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


# -------------------------------
# 4. LEAKY RELU
# -------------------------------

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)


def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)


# -------------------------------
# 5. SOFTMAX (OUTPUT LAYER)
# -------------------------------

def softmax(z):
    z_stable = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# -------------------------------
# 6. ACTIVATION COMPARISON
# -------------------------------

z = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])

sig = sigmoid(z)
th = tanh(z)
re = relu(z)
lre = leaky_relu(z)


# -------------------------------
# 7. GRADIENT BEHAVIOR CHECK
# -------------------------------

sig_grad = sigmoid_derivative(sig)
tanh_grad = tanh_derivative(th)
relu_grad = relu_derivative(z)


# -------------------------------
# 8. REAL DL PATTERN
# -------------------------------

# Hidden layers → ReLU
# Output layer → Sigmoid (binary) or Softmax (multi-class)


# -------------------------------
# 9. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("Sigmoid:", sig)
    print("Tanh:", th)
    print("ReLU:", re)
    print("Leaky ReLU:", lre)

    print("\nGradient Comparison")
    print("Sigmoid Grad:", sig_grad)
    print("Tanh Grad:", tanh_grad)
    print("ReLU Grad:", relu_grad)

    print("\nactivation_functions.py executed successfully")
