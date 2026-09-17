"""
Semantic Search Practice

Practice:
- TF-IDF vectorization
- Text embeddings
- Cosine similarity
- Query-document similarity
- Ranking search results
- Top-K retrieval
- Reusable search function

Note:
This uses TF-IDF vectors rather than neural embeddings.
It is a lightweight way to understand the retrieval
concept used in larger AI/RAG systems.
"""

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# 1. Document Collection
# ---------------------------------------------------------

documents = [
    "Python is widely used for machine learning and artificial intelligence.",
    "Machine learning models learn patterns from data.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing allows computers to understand text.",
    "Computer vision enables machines to analyze images and videos.",
    "Docker is used to package applications into containers.",
    "Git is a version control system used by software developers.",
    "Cloud computing provides scalable computing resources.",
    "Retrieval augmented generation combines document retrieval with language models.",
    "Vector databases are commonly used to store and search embeddings.",
]


print("Number of documents:", len(documents))


# ---------------------------------------------------------
# 2. Create TF-IDF Vectorizer
# ---------------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
)


# ---------------------------------------------------------
# 3. Convert Documents into Vectors
# ---------------------------------------------------------

document_vectors = vectorizer.fit_transform(
    documents
)

print("\nDocument Vector Shape:")
print(document_vectors.shape)


# ---------------------------------------------------------
# 4. Inspect Vocabulary
# ---------------------------------------------------------

print("\nVocabulary Size:")
print(len(vectorizer.get_feature_names_out()))


# ---------------------------------------------------------
# 5. Search Function
# ---------------------------------------------------------

def search(query, top_k=3):
    """
    Search documents using cosine similarity.

    Parameters:
        query: User's search query.
        top_k: Number of results to return.

    Returns:
        Ranked list of documents and similarity scores.
    """

    # Convert query into the same vector space
    query_vector = vectorizer.transform(
        [query]
    )

    # Calculate similarity between query and
    # every document.
    similarities = cosine_similarity(
        query_vector,
        document_vectors,
    )[0]

    # Sort indices from highest similarity
    # to lowest similarity.
    ranked_indices = np.argsort(
        similarities
    )[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        results.append(
            {
                "document_id": int(index),
                "score": float(similarities[index]),
                "text": documents[index],
            }
        )

    return results


# ---------------------------------------------------------
# 6. Perform a Search
# ---------------------------------------------------------

query = "How does artificial intelligence learn from data?"

results = search(
    query,
    top_k=3,
)

print("\nSearch Query:")
print(query)

print("\nTop Results:")

for rank, result in enumerate(
    results,
    start=1,
):
    print(
        f"\nRank {rank}"
        f"\nScore: {result['score']:.4f}"
        f"\nDocument: {result['text']}"
    )


# ---------------------------------------------------------
# 7. Test Multiple Queries
# ---------------------------------------------------------

queries = [
    "neural networks and deep learning",
    "how computers understand language",
    "software container technology",
    "searching vector embeddings",
]


print("\n" + "=" * 60)
print("MULTIPLE SEARCH QUERIES")
print("=" * 60)

for query in queries:

    print(f"\nQuery: {query}")

    results = search(
        query,
        top_k=2,
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{result['text']} "
            f"(score={result['score']:.4f})"
        )


# ---------------------------------------------------------
# 8. Similarity Matrix
# ---------------------------------------------------------
# Compare every document with every other document.

similarity_matrix = cosine_similarity(
    document_vectors
)

print("\nSimilarity Matrix Shape:")
print(similarity_matrix.shape)


# ---------------------------------------------------------
# 9. Find Most Similar Document Pair
# ---------------------------------------------------------

best_score = -1
best_pair = None

for i in range(len(documents)):

    for j in range(i + 1, len(documents)):

        score = similarity_matrix[i, j]

        if score > best_score:
            best_score = score
            best_pair = (i, j)


print("\nMost Similar Document Pair:")

if best_pair is not None:

    first, second = best_pair

    print(f"Document 1: {documents[first]}")
    print(f"Document 2: {documents[second]}")
    print(f"Similarity: {best_score:.4f}")


# ---------------------------------------------------------
# 10. Apply a Similarity Threshold
# ---------------------------------------------------------

def search_with_threshold(
    query,
    threshold=0.10,
    top_k=5,
):
    """
    Return only documents whose similarity
    exceeds the specified threshold.
    """

    query_vector = vectorizer.transform(
        [query]
    )

    similarities = cosine_similarity(
        query_vector,
        document_vectors,
    )[0]

    ranked_indices = np.argsort(
        similarities
    )[::-1]

    results = []

    for index in ranked_indices:

        score = similarities[index]

        if score >= threshold:

            results.append(
                {
                    "document_id": int(index),
                    "score": float(score),
                    "text": documents[index],
                }
            )

        if len(results) >= top_k:
            break

    return results


# ---------------------------------------------------------
# 11. Threshold-Based Search
# ---------------------------------------------------------

query = "quantum physics"

threshold_results = search_with_threshold(
    query,
    threshold=0.10,
)

print("\nThreshold Search:")
print("Query:", query)

if threshold_results:

    for result in threshold_results:
        print(
            f"- {result['text']} "
            f"(score={result['score']:.4f})"
        )

else:
    print("No sufficiently similar documents found.")


# ---------------------------------------------------------
# 12. Simple Search Result Formatter
# ---------------------------------------------------------

def display_results(
    query,
    top_k=3,
):
    """
    Display formatted search results.
    """

    print("\n" + "-" * 60)
    print(f"QUERY: {query}")
    print("-" * 60)

    results = search(
        query,
        top_k=top_k,
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"\n[{rank}] "
            f"Score: {result['score']:.4f}"
        )

        print(result["text"])


# ---------------------------------------------------------
# 13. Test Search Interface
# ---------------------------------------------------------

display_results(
    "What is retrieval augmented generation?",
    top_k=3,
)

display_results(
    "How are images analyzed by computers?",
    top_k=3,
)


# ---------------------------------------------------------
# 14. Understand the Retrieval Pipeline
# ---------------------------------------------------------

print(
    """
\nRetrieval Pipeline:

Documents
    ↓
TF-IDF Vectorization
    ↓
Document Vectors
    ↓
       User Query
           ↓
    Query Vectorization
           ↓
    Cosine Similarity
           ↓
    Similarity Scores
           ↓
       Ranking
           ↓
       Top-K Results
"""
)


# ---------------------------------------------------------
# 15. Important AI Engineering Concepts
# ---------------------------------------------------------

print("\nImportant Concepts:")

print(
    "1. Documents must be converted into numerical vectors."
)

print(
    "2. The query must use the same vectorizer."
)

print(
    "3. Cosine similarity measures vector similarity."
)

print(
    "4. Higher similarity generally means stronger lexical relevance."
)

print(
    "5. Top-K retrieval selects the most relevant documents."
)

print(
    "6. A similarity threshold can filter weak results."
)

print(
    "7. This is the basic architecture behind a retrieval system."
)

print(
    "8. Modern RAG systems commonly replace TF-IDF with "
    "dense neural embeddings."
)
