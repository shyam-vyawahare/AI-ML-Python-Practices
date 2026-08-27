"""
optimization_algorithms.py

Unit 3: Math for Machine Learning

Objective:
- Understand optimization in machine learning
- Implement Gradient Descent variants from scratch
- Observe how parameters converge toward the minimum
"""

import numpy as np


# -------------------------------
# 1. CREATE DATA
# -------------------------------

np.random.seed(42)

X = np.linspace(0, 10, 50)

noise = np.random.normal(
    0,
    1,
    size=X.shape
)

y = 2 * X + 3 + noise


# -------------------------------
# 2. ADD BIAS TERM
# -------------------------------

X_matrix = np.column_stack(
    (np.ones(len(X)), X)
)


# -------------------------------
# 3. PREDICTION FUNCTION
# -------------------------------

def predict(X, weights):
    return X @ weights


# -------------------------------
# 4. MEAN SQUARED ERROR
# -------------------------------

def mse_loss(y_true, y_pred):
    return np.mean(
        (y_true - y_pred) ** 2
    )


# -------------------------------
# 5. GRADIENT CALCULATION
# -------------------------------

def calculate_gradient(X, y, weights):

    predictions = predict(
        X,
        weights
    )

    errors = predictions - y

    gradient = (
        2 / len(y)
    ) * X.T @ errors

    return gradient


# -------------------------------
# 6. BATCH GRADIENT DESCENT
# -------------------------------

def batch_gradient_descent(
    X,
    y,
    learning_rate=0.01,
    epochs=100
):

    weights = np.zeros(
        X.shape[1]
    )

    losses = []

    for epoch in range(epochs):

        gradient = calculate_gradient(
            X,
            y,
            weights
        )

        weights -= (
            learning_rate * gradient
        )

        predictions = predict(
            X,
            weights
        )

        loss = mse_loss(
            y,
            predictions
        )

        losses.append(loss)

    return weights, losses


# -------------------------------
# 7. STOCHASTIC GRADIENT DESCENT
# -------------------------------

def stochastic_gradient_descent(
    X,
    y,
    learning_rate=0.01,
    epochs=100
):

    weights = np.zeros(
        X.shape[1]
    )

    losses = []

    for epoch in range(epochs):

        indices = np.random.permutation(
            len(X)
        )

        for index in indices:

            x_i = X[index:index + 1]
            y_i = y[index:index + 1]

            gradient = calculate_gradient(
                x_i,
                y_i,
                weights
            )

            weights -= (
                learning_rate * gradient
            )

        predictions = predict(
            X,
            weights
        )

        loss = mse_loss(
            y,
            predictions
        )

        losses.append(loss)

    return weights, losses


# -------------------------------
# 8. MINI-BATCH GRADIENT DESCENT
# -------------------------------

def mini_batch_gradient_descent(
    X,
    y,
    learning_rate=0.01,
    epochs=100,
    batch_size=8
):

    weights = np.zeros(
        X.shape[1]
    )

    losses = []

    for epoch in range(epochs):

        indices = np.random.permutation(
            len(X)
        )

        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start in range(
            0,
            len(X),
            batch_size
        ):

            end = start + batch_size

            X_batch = X_shuffled[
                start:end
            ]

            y_batch = y_shuffled[
                start:end
            ]

            gradient = calculate_gradient(
                X_batch,
                y_batch,
                weights
            )

            weights -= (
                learning_rate * gradient
            )

        predictions = predict(
            X,
            weights
        )

        loss = mse_loss(
            y,
            predictions
        )

        losses.append(loss)

    return weights, losses


# -------------------------------
# 9. RUN OPTIMIZERS
# -------------------------------

batch_weights, batch_losses = (
    batch_gradient_descent(
        X_matrix,
        y
    )
)

sgd_weights, sgd_losses = (
    stochastic_gradient_descent(
        X_matrix,
        y
    )
)

mini_weights, mini_losses = (
    mini_batch_gradient_descent(
        X_matrix,
        y
    )
)


# -------------------------------
# 10. COMPARE RESULTS
# -------------------------------

print("Optimization Results")

print("\nBatch Gradient Descent:")
print("Weights:", np.round(batch_weights, 4))
print("Final Loss:", round(batch_losses[-1], 4))

print("\nStochastic Gradient Descent:")
print("Weights:", np.round(sgd_weights, 4))
print("Final Loss:", round(sgd_losses[-1], 4))

print("\nMini-Batch Gradient Descent:")
print("Weights:", np.round(mini_weights, 4))
print("Final Loss:", round(mini_losses[-1], 4))


# -------------------------------
# 11. EXPECTED RELATIONSHIP
# -------------------------------

print("\nExpected Relationship:")
print("y ≈ 2x + 3")

print("\nLearned Parameters:")
print(
    f"Batch      → "
    f"y ≈ {batch_weights[1]:.2f}x "
    f"+ {batch_weights[0]:.2f}"
)

print(
    f"SGD        → "
    f"y ≈ {sgd_weights[1]:.2f}x "
    f"+ {sgd_weights[0]:.2f}"
)

print(
    f"Mini-Batch  → "
    f"y ≈ {mini_weights[1]:.2f}x "
    f"+ {mini_weights[0]:.2f}"
)


# -------------------------------
# 12. CONCEPT SUMMARY
# -------------------------------

print("\nConcept Summary:")

print(
    "- Batch GD → Uses the entire dataset per update"
)

print(
    "- SGD → Uses one sample per update"
)

print(
    "- Mini-Batch GD → Uses a small batch per update"
)

print(
    "- Learning Rate → Controls the update size"
)

print(
    "- Goal → Minimize the loss function"
)


# -------------------------------
# 13. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print(
        "\noptimization_algorithms.py "
        "executed successfully"
)
