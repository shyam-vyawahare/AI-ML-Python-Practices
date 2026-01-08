"""
linear_algebra.py

Unit 3: Math for Machine Learning

Objective:
- Build linear algebra intuition using NumPy
- Understand vectors, matrices, and transformations
- Prepare foundation for ML models and neural networks
"""

import numpy as np


# -------------------------------
# 1. SCALARS, VECTORS, MATRICES
# -------------------------------

scalar = 5

vector = np.array([1, 2, 3])

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])


# -------------------------------
# 2. SHAPES & DIMENSIONS
# -------------------------------

vector_shape = vector.shape
matrix_shape = matrix.shape

vector_dim = vector.ndim
matrix_dim = matrix.ndim


# -------------------------------
# 3. VECTOR OPERATIONS
# -------------------------------

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

vector_add = v1 + v2
vector_sub = v1 - v2
vector_mul = v1 * v2  # element-wise


# -------------------------------
# 4. DOT PRODUCT (CORE ML OPERATION)
# -------------------------------

dot_product = np.dot(v1, v2)
dot_product_alt = v1 @ v2


# -------------------------------
# 5. MATRIX MULTIPLICATION
# -------------------------------

A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

matrix_product = A @ B


# -------------------------------
# 6. TRANSPOSE
# -------------------------------

A_transpose = A.T


# -------------------------------
# 7. IDENTITY MATRIX
# -------------------------------

identity = np.eye(3)
result_identity = identity @ np.array([1, 2, 3])


# -------------------------------
# 8. MATRIX × VECTOR (ML PREDICTION)
# -------------------------------

X = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

weights = np.array([0.5, 1.5])

linear_output = X @ weights


# -------------------------------
# 9. ADDING BIAS (IMPORTANT)
# -------------------------------

bias = 2
linear_output_with_bias = linear_output + bias


# -------------------------------
# 10. RESHAPING VECTORS
# -------------------------------

column_vector = np.array([1, 2, 3]).reshape(-1, 1)
row_vector = np.array([1, 2, 3]).reshape(1, -1)


# -------------------------------
# 11. MATRIX SUM & MEAN (AXIS)
# -------------------------------

sum_rows = np.sum(matrix, axis=1)
sum_cols = np.sum(matrix, axis=0)

mean_rows = np.mean(matrix, axis=1)
mean_cols = np.mean(matrix, axis=0)


# -------------------------------
# 12. NORM (VECTOR MAGNITUDE)
# -------------------------------

vector_norm = np.linalg.norm(vector)


# -------------------------------
# 13. REAL ML PATTERN
# -------------------------------

# y = Xw + b
X = np.array([
    [1, 1],
    [1, 2],
    [1, 3]
])

w = np.array([0.5, 1.0])
b = 0.2

y_pred = X @ w + b


# -------------------------------
# 14. SHAPE SAFETY CHECK
# -------------------------------

assert X.shape[1] == w.shape[0], "Shape mismatch in matrix multiplication"


# -------------------------------
# 15. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("linear_algebra.py executed successfully")
