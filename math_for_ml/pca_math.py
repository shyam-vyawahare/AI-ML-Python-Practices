"""
pca_math.py

Unit 3: Math for Machine Learning

Objective:
- Implement Principal Component Analysis (PCA)
- Reduce dimensionality using linear algebra
- Understand the mathematical foundation of PCA
"""

import numpy as np


# -------------------------------
# 1. SAMPLE DATASET
# -------------------------------

X = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2.0, 1.6],
    [1.0, 1.1],
    [1.5, 1.6],
    [1.1, 0.9]
])

print("Original Data:")
print(X)


# -------------------------------
# 2. STANDARDIZE DATA
# -------------------------------

X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

print("\nCentered Data:")
print(np.round(X_centered, 2))


# -------------------------------
# 3. COVARIANCE MATRIX
# -------------------------------

covariance_matrix = np.cov(X_centered.T)

print("\nCovariance Matrix:")
print(np.round(covariance_matrix, 4))


# -------------------------------
# 4. EIGEN DECOMPOSITION
# -------------------------------

eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

print("\nEigenvalues:")
print(np.round(eigenvalues, 4))

print("\nEigenvectors:")
print(np.round(eigenvectors, 4))


# -------------------------------
# 5. SORT COMPONENTS
# -------------------------------

sorted_indices = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

print("\nSorted Eigenvalues:")
print(np.round(eigenvalues, 4))


# -------------------------------
# 6. SELECT FIRST PRINCIPAL COMPONENT
# -------------------------------

principal_component = eigenvectors[:, 0]

print("\nPrincipal Component:")
print(np.round(principal_component, 4))


# -------------------------------
# 7. PROJECT DATA
# -------------------------------

projected_data = X_centered @ principal_component

print("\nProjected Data:")
print(np.round(projected_data, 4))


# -------------------------------
# 8. EXPLAINED VARIANCE
# -------------------------------

explained_variance = eigenvalues / np.sum(eigenvalues)

print("\nExplained Variance Ratio:")
print(np.round(explained_variance, 4))


# -------------------------------
# 9. APPLICATIONS
# -------------------------------

print("\nApplications:")
print("- Feature Reduction")
print("- Data Visualization")
print("- Image Compression")
print("- Noise Reduction")
print("- Faster Model Training")


# -------------------------------
# 10. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\npca_math.py executed successfully")
