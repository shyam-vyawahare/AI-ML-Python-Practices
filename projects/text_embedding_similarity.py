"""
Text Embedding Similarity

Practice:
- Sentence embeddings
- Semantic similarity
- Cosine similarity
- Comparing related and unrelated text
- Similarity thresholds
- Finding the most similar sentence

Requirement:
    pip install sentence-transformers
"""

import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# 1. Load Embedding Model
# ---------------------------------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ---------------------------------------------------------
# 2. Example Sentences
# ---------------------------------------------------------

sentences = [
    "I love programming in Python.",
    "Python is my favorite programming language.",
    "I enjoy developing software.",
    "The weather is very pleasant today.",
    "Machine learning models learn patterns from data.",
    "Artificial intelligence can learn from examples.",
]


# ---------------------------------------------------------
# 3. Generate Embeddings
# ---------------------------------------------------------

embeddings = model.encode(
    sentences,
    convert_to_numpy=True,
)

print("\nEmbedding Shape:")
print(embeddings.shape)

print(
    "\nEmbedding Dimensions:",
    embeddings.shape[1],
)


# ---------------------------------------------------------
# 4. Inspect One Embedding
# ---------------------------------------------------------

print("\nFirst Sentence:")
print(sentences[0])

print("\nFirst Embedding:")
print(embeddings[0])


# ---------------------------------------------------------
# 5. Compare Two Sentences
# ---------------------------------------------------------

sentence_a = "I love programming in Python."
sentence_b = "Python is my favorite programming language."

embedding_a = model.encode(
    [sentence_a],
    convert_to_numpy=True,
)

embedding_b = model.encode(
    [sentence_b],
    convert_to_numpy=True,
)

similarity = cosine_similarity(
    embedding_a,
    embedding_b,
)[0][0]

print("\nTwo-Sentence Similarity:")
print(f"Sentence A: {sentence_a}")
print(f"Sentence B: {sentence_b}")
print(f"Similarity: {similarity:.4f}")


# ---------------------------------------------------------
# 6. Compare Unrelated Sentences
# ---------------------------------------------------------

sentence_c = "I love programming in Python."
sentence_d = "The weather is very pleasant today."

embedding_c = model.encode(
    [sentence_c],
    convert_to_numpy=True,
)

embedding_d = model.encode(
    [sentence_d],
    convert_to_numpy=True,
)

similarity = cosine_similarity(
    embedding_c,
    embedding_d,
)[0][0]

print("\nUnrelated Sentence Similarity:")
print(f"Sentence A: {sentence_c}")
print(f"Sentence B: {sentence_d}")
print(f"Similarity: {similarity:.4f}")


# ---------------------------------------------------------
# 7. Calculate Complete Similarity Matrix
# ---------------------------------------------------------

similarity_matrix = cosine_similarity(
    embeddings
)

print("\nSimilarity Matrix:")

np.set_printoptions(
    precision=3,
    suppress=True,
)

print(similarity_matrix)


# ---------------------------------------------------------
# 8. Display Sentence Similarities
# ---------------------------------------------------------

print("\nSentence Similarities:")

for i in range(len(sentences)):

    for j in range(i + 1, len(sentences)):

        score = similarity_matrix[i][j]

        print(
            f"{i} ↔ {j}: "
            f"{score:.4f}"
        )


# ---------------------------------------------------------
# 9. Find Most Similar Sentence
# ---------------------------------------------------------

query = (
    "I want to learn programming with Python."
)

query_embedding = model.encode(
    [query],
    convert_to_numpy=True,
)

scores = cosine_similarity(
    query_embedding,
    embeddings,
)[0]

best_index = int(
    np.argmax(scores)
)

print("\nSemantic Search:")

print("Query:")
print(query)

print("\nMost Similar Sentence:")
print(sentences[best_index])

print(
    f"Similarity: {scores[best_index]:.4f}"
)


# ---------------------------------------------------------
# 10. Rank All Sentences
# ---------------------------------------------------------

ranked_indices = np.argsort(
    scores
)[::-1]

print("\nRanked Results:")

for rank, index in enumerate(
    ranked_indices,
    start=1,
):

    print(
        f"{rank}. "
        f"{sentences[index]} "
        f"-> {scores[index]:.4f}"
    )


# ---------------------------------------------------------
# 11. Reusable Similarity Function
# ---------------------------------------------------------

def calculate_similarity(
    text_a,
    text_b,
):
    """
    Calculate semantic similarity between
    two pieces of text.
    """

    vectors = model.encode(
        [text_a, text_b],
        convert_to_numpy=True,
    )

    score = cosine_similarity(
        [vectors[0]],
        [vectors[1]],
    )[0][0]

    return float(score)


# ---------------------------------------------------------
# 12. Test Reusable Function
# ---------------------------------------------------------

text_a = (
    "Machine learning learns from data."
)

text_b = (
    "AI models identify patterns in datasets."
)

score = calculate_similarity(
    text_a,
    text_b,
)

print("\nReusable Similarity Function:")

print(f"Text A: {text_a}")
print(f"Text B: {text_b}")
print(f"Similarity: {score:.4f}")


# ---------------------------------------------------------
# 13. Similarity Threshold
# ---------------------------------------------------------

def are_semantically_similar(
    text_a,
    text_b,
    threshold=0.50,
):
    """
    Determine whether two texts are sufficiently
    similar according to a chosen threshold.
    """

    score = calculate_similarity(
        text_a,
        text_b,
    )

    return score >= threshold, score


text_a = "How does machine learning work?"
text_b = "How do AI models learn from data?"

is_similar, score = are_semantically_similar(
    text_a,
    text_b,
)

print("\nSimilarity Threshold:")

print(f"Score: {score:.4f}")
print(f"Similar: {is_similar}")


# ---------------------------------------------------------
# 14. Embedding-Based Search Function
# ---------------------------------------------------------

def semantic_search(
    query,
    documents,
    top_k=3,
):
    """
    Search documents using sentence embeddings.
    """

    document_embeddings = model.encode(
        documents,
        convert_to_numpy=True,
    )

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
    )

    scores = cosine_similarity(
        query_embedding,
        document_embeddings,
    )[0]

    ranked_indices = np.argsort(
        scores
    )[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        results.append(
            {
                "document": documents[index],
                "score": float(scores[index]),
            }
        )

    return results


# ---------------------------------------------------------
# 15. Mini Semantic Search Engine
# ---------------------------------------------------------

documents = [
    "Python is commonly used for artificial intelligence.",
    "Neural networks are powerful models for deep learning.",
    "Docker packages applications into isolated containers.",
    "Vector databases store embeddings for similarity search.",
    "Natural language processing works with human language.",
    "Git helps developers track changes in source code.",
]

query = (
    "Which technology can be used to search embeddings?"
)

results = semantic_search(
    query,
    documents,
    top_k=3,
)

print("\nMini Semantic Search Engine:")

print("Query:", query)

for rank, result in enumerate(
    results,
    start=1,
):

    print(
        f"\n{rank}. {result['document']}"
    )

    print(
        f"   Score: {result['score']:.4f}"
    )


# ---------------------------------------------------------
# 16. Understand the AI Retrieval Pipeline
# ---------------------------------------------------------

print(
    """
\nEmbedding Retrieval Pipeline:

Documents
    ↓
Embedding Model
    ↓
Dense Vectors
    ↓
Vector Representation
    │
    │
User Query
    ↓
Embedding Model
    ↓
Query Vector
    ↓
Cosine Similarity
    ↓
Similarity Scores
    ↓
Ranking
    ↓
Top-K Documents
"""
)


# ---------------------------------------------------------
# 17. Important Concepts
# ---------------------------------------------------------

print("\nImportant Concepts:")

print(
    "1. Embeddings represent text as numerical vectors."
)

print(
    "2. Similar meanings can produce similar vectors."
)

print(
    "3. Cosine similarity measures vector similarity."
)

print(
    "4. Top-K retrieval returns the most relevant results."
)

print(
    "5. Embeddings enable semantic rather than exact "
    "keyword-based retrieval."
)

print(
    "6. Modern RAG systems commonly use this retrieval approach."
)
