"""
vector_norms.py

Unit 3: Math for Machine Learning

Objective:
- Learn vector norms
- Compute common distance metrics
- Understand their applications in ML
"""

import numpy as np


# -------------------------------
# 1. CREATE VECTORS
# -------------------------------

vector_a = np.array([2, 4, 6])
vector_b = np.array([1, 5, 7])

print("Vector A:", vector_a)
print("Vector B:", vector_b)


# -------------------------------
# 2. L1 NORM (MANHATTAN NORM)
# -------------------------------

l1_norm = np.linalg.norm(vector_a, ord=1)

print("\nL1 Norm:", l1_norm)


# -------------------------------
# 3. L2 NORM (EUCLIDEAN NORM)
# -------------------------------

l2_norm = np.linalg.norm(vector_a)

print("L2 Norm:", round(l2_norm, 4))


# -------------------------------
# 4. INFINITY NORM
# -------------------------------

inf_norm = np.linalg.norm(vector_a, ord=np.inf)

print("Infinity Norm:", inf_norm)


# -------------------------------
# 5. EUCLIDEAN DISTANCE
# -------------------------------

euclidean = np.linalg.norm(vector_a - vector_b)

print("\nEuclidean Distance:", round(euclidean, 4))


# -------------------------------
# 6. MANHATTAN DISTANCE
# -------------------------------

manhattan = np.sum(np.abs(vector_a - vector_b))

print("Manhattan Distance:", manhattan)


# -------------------------------
# 7. COSINE SIMILARITY
# -------------------------------

cosine_similarity = np.dot(vector_a, vector_b) / (
    np.linalg.norm(vector_a) *
    np.linalg.norm(vector_b)
)

print("Cosine Similarity:", round(cosine_similarity, 4))


# -------------------------------
# 8. PRACTICAL ML EXAMPLE
# -------------------------------

query_embedding = np.array([0.2, 0.8, 0.5])
document_embedding = np.array([0.3, 0.7, 0.4])

similarity = np.dot(
    query_embedding,
    document_embedding
) / (
    np.linalg.norm(query_embedding) *
    np.linalg.norm(document_embedding)
)

print("\nEmbedding Similarity:", round(similarity, 4))


# -------------------------------
# 9. WHEN TO USE WHICH?
# -------------------------------

print("\nApplications:")
print("- L1 Norm          -> Sparse models (Lasso)")
print("- L2 Norm          -> Regularization (Ridge)")
print("- Euclidean        -> KNN, Clustering")
print("- Manhattan        -> Grid-based distances")
print("- Cosine Similarity-> NLP, Embeddings, RAG")


# -------------------------------
# 10. EXECUTION CHECK
# -------------------------------

if __name__ == "__main__":
    print("\nvector_norms.py executed successfully")
