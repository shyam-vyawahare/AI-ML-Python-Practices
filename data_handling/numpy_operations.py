"""
numpy_operations.py

Unit 2: Data Handling for AI & ML

Objective:
- Perform numerical operations used in ML algorithms
- Understand axis-based computation
- Build intuition for matrix math and vector operations
"""

import numpy as np


# -------------------------------
# 1. AXIS CONCEPT (CRITICAL)
# -------------------------------

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

row_sum = np.sum(matrix, axis=1)   # sum across columns
col_sum = np.sum(matrix, axis=0)   # sum across rows


# -------------------------------
# 2. MEAN, VARIANCE, STD
# -------------------------------

data = np.array([10, 20, 30, 40, 50])

mean = np.mean(data)
variance = np.var(data)
std_dev = np.std(data)


# -------------------------------
# 3. ELEMENT-WISE OPERATIONS
# -------------------------------

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

add = a + b
subtract = a - b
multiply = a * b
divide = a / b


# -------------------------------
# 4. DOT PRODUCT (ML CORE)
# -------------------------------

vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])

dot_product = np.dot(vector1, vector2)


# -------------------------------
# 5. MATRIX MULTIPLICATION
# -------------------------------

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

matrix_product = A @ B  # preferred over np.dot for matrices


# -------------------------------
# 6. TRANSPOSE
# -------------------------------

transposed = matrix.T


# -------------------------------
# 7. SHAPE ALIGNMENT (VERY IMPORTANT)
# -------------------------------

X = np.array([[1], [2], [3]])
y = np.array([10, 20, 30])

y_reshaped = y.reshape(-1, 1)


# -------------------------------
# 8. CLIPPING VALUES (ML SAFETY)
# -------------------------------

predictions = np.array([0.01, 0.5, 0.99, 1.2, -0.2])

clipped = np.clip(predictions, 0.0, 1.0)


# -------------------------------
# 9. WHERE CONDITION (VECTORIZED IF)
# -------------------------------

labels = np.where(predictions >= 0.5, 1, 0)


# -------------------------------
# 10. CUMULATIVE OPERATIONS
# -------------------------------

values = np.array([1, 2, 3, 4])

cumulative_sum = np.cumsum(values)
cumulative_product = np.cumprod(values)


# -------------------------------
# 11. SORTING & ARG SORT
# -------------------------------

scores = np.array([88, 92, 75, 91])

sorted_scores = np.sort(scores)
sorted_indices = np.argsort(scores)


# -------------------------------
# 12. REAL-WORLD ML PATTERN
# -------------------------------

# Linear model: y = Xw + b
X = np.array([[1, 2],
              [3, 4],
              [5, 6]])

weights = np.array([0.5, 1.5])
bias = 2

y_pred = X @ weights + bias


# -------------------------------
# 13. NUMERICAL STABILITY
# -------------------------------

logits = np.array([1000, 1001, 1002])

stable_logits = logits - np.max(logits)
softmax = np.exp(stable_logits) / np.sum(np.exp(stable_logits))


# -------------------------------
# 14. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("numpy_operations.py executed successfully")
