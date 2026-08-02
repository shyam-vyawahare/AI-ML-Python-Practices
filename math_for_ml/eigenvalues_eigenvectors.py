"""
eigenvalues_eigenvectors.py

Unit 3: Math for Machine Learning

Objective:
- Compute eigenvalues and eigenvectors
- Verify the eigenvalue equation
- Understand their importance in ML
"""

import numpy as np


# -------------------------------
# 1. CREATE A MATRIX
# -------------------------------

A = np.array([
    [4, 2],
    [1, 3]
])

print("Matrix A:")
print(A)


# -------------------------------
# 2. COMPUTE EIGENVALUES & EIGENVECTORS
# -------------------------------

eigenvalues, eigenvectors = np.linalg.eig(A)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)


# -------------------------------
# 3. VERIFY Av = λv
# -------------------------------

print("\nVerification of Av = λv")

for i in range(len(eigenvalues)):
    eigenvalue = eigenvalues[i]
    eigenvector = eigenvectors[:, i]

    left = A @ eigenvector
    right = eigenvalue * eigenvector

    print(f"\nEigenvalue {i + 1}: {eigenvalue:.4f}")
    print("A × v =", np.round(left, 4))
    print("λ × v =", np.round(right, 4))


# -------------------------------
# 4. MATRIX RECONSTRUCTION
# -------------------------------

P = eigenvectors
D = np.diag(eigenvalues)
P_inv = np.linalg.inv(P)

A_reconstructed = P @ D @ P_inv

print("\nReconstructed Matrix:")
print(np.round(A_reconstructed, 4))


# -------------------------------
# 5. PRACTICAL ML EXAMPLE
# -------------------------------

covariance_matrix = np.array([
    [3.0, 1.5],
    [1.5, 1.0]
])

eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)

principal_component = eig_vecs[:, np.argmax(eig_vals)]

print("\nPrincipal Component:")
print(np.round(principal_component, 4))


# -------------------------------
# 6. WHY ARE THEY IMPORTANT?
# -------------------------------

print("\nApplications:")
print("- Principal Component Analysis (PCA)")
print("- Dimensionality Reduction")
print("- Image Compression")
print("- Recommendation Systems")
print("- Spectral Clustering")


# -------------------------------
# 7. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\neigenvalues_eigenvectors.py executed successfully")
