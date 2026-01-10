"""
training_loop.py

Unit 5: Deep Learning Fundamentals

Objective:
- Build a complete training loop
- Understand epoch-based learning
- Observe loss convergence and learning rate effects
"""

import numpy as np


# -------------------------------
# 1. ACTIVATION & LOSS
# -------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    return a * (1 - a)


def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )


# -------------------------------
# 2. SIMPLE NEURAL NETWORK
# -------------------------------

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, lr=0.1):
        self.lr = lr

        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, 1)
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.output = sigmoid(self.z2)

        return self.output

    def backward(self, X, y):
        error = self.output - y

        d_output = error * sigmoid_derivative(self.output)

        dW2 = self.a1.T @ d_output
        db2 = np.sum(d_output, axis=0, keepdims=True)

        d_hidden = d_output @ self.W2.T * sigmoid_derivative(self.a1)

        dW1 = X.T @ d_hidden
        db1 = np.sum(d_hidden, axis=0, keepdims=True)

        # Gradient descent update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1


# -------------------------------
# 3. DATASET (BINARY CLASSIFICATION)
# -------------------------------

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])


# -------------------------------
# 4. TRAINING LOOP
# -------------------------------

def train(model, X, y, epochs=1000):
    for epoch in range(epochs):
        y_pred = model.forward(X)
        loss = binary_cross_entropy(y, y_pred)
        model.backward(X, y)

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")


# -------------------------------
# 5. LEARNING RATE EXPERIMENT
# -------------------------------

learning_rates = [0.01, 0.1, 0.5]

for lr in learning_rates:
    print(f"\nTraining with learning rate = {lr}")
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, lr=lr)
    train(nn, X, y, epochs=500)


# -------------------------------
# 6. FINAL PREDICTIONS
# -------------------------------

final_predictions = nn.forward(X)


# -------------------------------
# 7. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\nFinal Predictions:")
    print(final_predictions.round(3))
    print("training_loop.py executed successfully")
