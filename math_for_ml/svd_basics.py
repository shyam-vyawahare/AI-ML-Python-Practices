"""
svd_basics.py

Unit 3: Math for Machine Learning

Objective:
- Learn Singular Value Decomposition (SVD)
- Reconstruct matrices using SVD
- Understand low-rank approximation
"""

import numpy as np


# -------------------------------
# 1. CREATE MATRIX
# -------------------------------

A = np.array([
    [3, 2, 2],
    [2, 3, -2]
], dtype=float)

print("Original Matrix:")
print(A)


# -------------------------------
# 2. COMPUTE SVD
# -------------------------------

U, S, VT = np.linalg.svd(A)

print("\nU Matrix:")
print(np.round(U, 4))

print("\nSingular Values:")
print(np.round(S, 4))

print("\nVᵀ Matrix:")
print(np.round(VT, 4))


# -------------------------------
# 3. RECONSTRUCT MATRIX
# -------------------------------

Sigma = np.zeros((A.shape[0], A.shape[1]))
np.fill_diagonal(Sigma, S)

A_reconstructed = U @ Sigma @ VT

print("\nReconstructed Matrix:")
print(np.round(A_reconstructed, 4))


# -------------------------------
# 4. LOW-RANK APPROXIMATION
# -------------------------------

k = 1

U_k = U[:, :k]
S_k = np.diag(S[:k])
VT_k = VT[:k, :]

A_low_rank = U_k @ S_k @ VT_k

print(f"\nRank-{k} Approximation:")
print(np.round(A_low_rank, 4))


# -------------------------------
# 5. RECONSTRUCTION ERROR
# -------------------------------

error = np.linalg.norm(A - A_low_rank)

print("\nApproximation Error:")
print(round(error, 4))


# -------------------------------
# 6. APPLICATIONS
# -------------------------------

print("\nApplications:")
print("- Recommendation Systems")
print("- Image Compression")
print("- NLP")
print("- Dimensionality Reduction")
print("- Latent Semantic Analysis (LSA)")


# -------------------------------
# 7. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\nsvd_basics.py executed successfully")
