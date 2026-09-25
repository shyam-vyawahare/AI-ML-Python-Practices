"""
RAG Hybrid Search Practice

Combines:
    1. Semantic similarity
    2. Keyword relevance

The goal is to improve retrieval by combining
semantic understanding with exact keyword matching.

Pipeline:

Query
  ↓
Semantic Search
  ↓
Keyword Search
  ↓
Score Normalization
  ↓
Weighted Score Fusion
  ↓
Top-K Results
"""

import re
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Document:
    doc_id: str
    text: str


@dataclass
class SearchResult:
    document: Document
    semantic_score: float
    keyword_score: float
    hybrid_score: float


# ============================================================
# TEXT PROCESSING
# ============================================================

def tokenize(text: str) -> List[str]:
    """
    Convert text into normalized tokens.
    """

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


# ============================================================
# KEYWORD SEARCH
# ============================================================

def keyword_score(
    query: str,
    document: str
) -> float:
    """
    Calculate a simple keyword relevance score.

    The score is based on the percentage of unique
    query terms appearing in the document.
    """

    query_tokens = set(tokenize(query))
    document_tokens = set(tokenize(document))

    if not query_tokens:
        return 0.0

    matches = query_tokens.intersection(
        document_tokens
    )

    return len(matches) / len(query_tokens)


# ============================================================
# EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# VECTOR UTILITIES
# ============================================================

def normalize_vectors(
    vectors: np.ndarray
) -> np.ndarray:
    """
    Normalize vectors for cosine similarity.
    """

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    norms = np.maximum(
        norms,
        1e-12
    )

    return vectors / norms


def cosine_similarity(
    query_vector: np.ndarray,
    document_vectors: np.ndarray
) -> np.ndarray:
    """
    Calculate cosine similarity.
    """

    query_vector = normalize_vectors(
        query_vector.reshape(1, -1)
    )[0]

    document_vectors = normalize_vectors(
        document_vectors
    )

    return document_vectors @ query_vector


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query: str,
    documents: List[Document]
) -> Dict[str, float]:
    """
    Calculate semantic similarity for every document.
    """

    query_embedding = model.encode(
        query,
        convert_to_numpy=True
    )

    document_embeddings = model.encode(
        [doc.text for doc in documents],
        convert_to_numpy=True
    )

    scores = cosine_similarity(
        query_embedding,
        document_embeddings
    )

    return {
        document.doc_id: float(score)
        for document, score
        in zip(documents, scores)
    }


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def min_max_normalize(
    scores: Dict[str, float]
) -> Dict[str, float]:
    """
    Normalize scores into the range [0, 1].
    """

    if not scores:
        return {}

    values = np.array(
        list(scores.values()),
        dtype=float
    )

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:

        return {
            key: 1.0
            for key in scores
        }

    normalized = (
        (values - minimum)
        /
        (maximum - minimum)
    )

    return {
        key: float(value)
        for key, value
        in zip(scores.keys(), normalized)
    }


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    query: str,
    documents: List[Document],
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    top_k: int = 5
) -> List[SearchResult]:
    """
    Perform hybrid search.

    Final score:

        semantic_score * semantic_weight
        +
        keyword_score * keyword_weight
    """

    # --------------------------------------------------------
    # Semantic retrieval
    # --------------------------------------------------------

    semantic_scores = semantic_search(
        query,
        documents
    )

    # --------------------------------------------------------
    # Keyword retrieval
    # --------------------------------------------------------

    keyword_scores = {
        doc.doc_id: keyword_score(
            query,
            doc.text
        )
        for doc in documents
    }

    # --------------------------------------------------------
    # Normalize semantic scores
    # --------------------------------------------------------

    normalized_semantic = min_max_normalize(
        semantic_scores
    )

    # Keyword scores are already in [0, 1]
    normalized_keyword = keyword_scores

    # --------------------------------------------------------
    # Score fusion
    # --------------------------------------------------------

    results = []

    for document in documents:

        semantic = normalized_semantic[
            document.doc_id
        ]

        keyword = normalized_keyword[
            document.doc_id
        ]

        hybrid = (
            semantic * semantic_weight
            +
            keyword * keyword_weight
        )

        results.append(
            SearchResult(
                document=document,
                semantic_score=semantic,
                keyword_score=keyword,
                hybrid_score=hybrid
            )
        )

    # --------------------------------------------------------
    # Rank results
    # --------------------------------------------------------

    results.sort(
        key=lambda result: result.hybrid_score,
        reverse=True
    )

    return results[:top_k]


# ============================================================
# SEMANTIC-ONLY SEARCH
# ============================================================

def semantic_only_search(
    query: str,
    documents: List[Document],
    top_k: int = 5
) -> List[SearchResult]:
    """
    Perform semantic-only retrieval.

    Used for comparison with hybrid search.
    """

    scores = semantic_search(
        query,
        documents
    )

    normalized_scores = min_max_normalize(
        scores
    )

    results = []

    for document in documents:

        score = normalized_scores[
            document.doc_id
        ]

        results.append(
            SearchResult(
                document=document,
                semantic_score=score,
                keyword_score=0.0,
                hybrid_score=score
            )
        )

    results.sort(
        key=lambda result: result.hybrid_score,
        reverse=True
    )

    return results[:top_k]


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    results: List[SearchResult],
    title: str
):
    """
    Display ranked search results.
    """

    print("\n")
    print("=" * 75)
    print(title)
    print("=" * 75)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRank {rank}"
        )

        print(
            f"Document: {result.document.doc_id}"
        )

        print(
            f"Semantic Score: "
            f"{result.semantic_score:.4f}"
        )

        print(
            f"Keyword Score:  "
            f"{result.keyword_score:.4f}"
        )

        print(
            f"Hybrid Score:   "
            f"{result.hybrid_score:.4f}"
        )

        print(
            f"Text: {result.document.text}"
        )


# ============================================================
# WEIGHT EXPERIMENT
# ============================================================

def compare_weights(
    query: str,
    documents: List[Document]
):
    """
    Compare different hybrid-search configurations.
    """

    configurations = [
        (1.0, 0.0),
        (0.8, 0.2),
        (0.7, 0.3),
        (0.5, 0.5),
        (0.3, 0.7),
    ]

    print("\n")
    print("=" * 75)
    print("HYBRID SEARCH WEIGHT EXPERIMENT")
    print("=" * 75)

    for semantic_weight, keyword_weight in configurations:

        results = hybrid_search(
            query=query,
            documents=documents,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            top_k=3
        )

        print(
            f"\nSemantic={semantic_weight:.1f} | "
            f"Keyword={keyword_weight:.1f}"
        )

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"{rank}. "
                f"{result.document.doc_id} "
                f"-> "
                f"{result.hybrid_score:.4f}"
            )


# ============================================================
# SAMPLE KNOWLEDGE BASE
# ============================================================

documents = [

    Document(
        "doc_01",
        """
        Machine learning allows computers to learn
        patterns from data without being explicitly
        programmed for every task.
        """
    ),

    Document(
        "doc_02",
        """
        Gradient descent is an optimization algorithm
        used to minimize the loss function while training
        machine learning models.
        """
    ),

    Document(
        "doc_03",
        """
        Neural networks contain interconnected neurons
        organized into layers and are widely used in
        deep learning applications.
        """
    ),

    Document(
        "doc_04",
        """
        Python is commonly used for artificial intelligence,
        machine learning, data science and automation.
        """
    ),

    Document(
        "doc_05",
        """
        Retrieval augmented generation combines document
        retrieval with large language models to generate
        grounded answers.
        """
    ),

    Document(
        "doc_06",
        """
        Vector databases store numerical representations
        of documents and allow similarity-based retrieval.
        """
    ),

    Document(
        "doc_07",
        """
        Fine tuning modifies a pretrained neural network
        using task-specific training data.
        """
    ),

    Document(
        "doc_08",
        """
        TF-IDF is a traditional information retrieval
        technique that measures the importance of terms
        within documents.
        """
    ),
]


# ============================================================
# MAIN EXPERIMENT
# ============================================================

if __name__ == "__main__":

    query = (
        "How does retrieval augmented generation "
        "use document retrieval?"
    )

    print("\n")
    print("=" * 75)
    print("RAG HYBRID SEARCH")
    print("=" * 75)

    print(
        f"\nQuery: {query}"
    )

    # --------------------------------------------------------
    # Semantic-only retrieval
    # --------------------------------------------------------

    semantic_results = semantic_only_search(
        query=query,
        documents=documents,
        top_k=5
    )

    display_results(
        semantic_results,
        "SEMANTIC-ONLY SEARCH"
    )

    # --------------------------------------------------------
    # Hybrid retrieval
    # --------------------------------------------------------

    hybrid_results = hybrid_search(
        query=query,
        documents=documents,
        semantic_weight=0.7,
        keyword_weight=0.3,
        top_k=5
    )

    display_results(
        hybrid_results,
        "HYBRID SEARCH"
    )

    # --------------------------------------------------------
    # Weight experiment
    # --------------------------------------------------------

    compare_weights(
        query=query,
        documents=documents
    )
