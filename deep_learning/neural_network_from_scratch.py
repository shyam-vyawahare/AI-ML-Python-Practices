"""
neural_network_from_scratch.py

Unit 5: Deep Learning Fundamentals

Objective:
- Build a simple neural network from scratch
- Implement forward propagation & backpropagation
- Understand how weights learn via gradients
"""

import numpy as np


# -------------------------------
# 1. ACTIVATION FUNCTIONS
# -------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    return a * (1 - a)


# -------------------------------
# 2. LOSS FUNCTION (MSE)
# -------------------------------

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# -------------------------------
# 3. SIMPLE NEURAL NETWORK
# -------------------------------

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1):
        self.lr = lr

        # Weight initialization
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    # ---------------------------
    # FORWARD PROPAGATION
    # ---------------------------
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2

    # ---------------------------
    # BACKPROPAGATION
    # ---------------------------
    def backward(self, X, y, output):
        error = y - output
        d_output = error * sigmoid_derivative(output)

        d_W2 = self.a1.T @ d_output
        d_b2 = np.sum(d_output, axis=0, keepdims=True)

        d_hidden = d_output @ self.W2.T * sigmoid_derivative(self.a1)

        d_W1 = X.T @ d_hidden
        d_b1 = np.sum(d_hidden, axis=0, keepdims=True)

        # Update weights
        self.W2 += self.lr * d_W2
        self.b2 += self.lr * d_b2
        self.W1 += self.lr * d_W1
        self.b1 += self.lr * d_b1

    # ---------------------------
    # TRAINING LOOP
    # ---------------------------
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = mean_squared_error(y, output)
            self.backward(X, y, output)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")


# -------------------------------
# 4. CREATE DATASET
# -------------------------------

# XOR-like pattern (non-linear problem)
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
# 5. TRAIN NETWORK
# -------------------------------

nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, lr=0.5)
nn.train(X, y, epochs=1000)


# -------------------------------
# 6. TEST NETWORK
# -------------------------------

predictions = nn.forward(X)


# -------------------------------
# 7. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\nFinal Predictions:")
    print(predictions.round(3))
    print("neural_network_from_scratch.py executed successfully")
