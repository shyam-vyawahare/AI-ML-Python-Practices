"""
RAG Multi-Query Retrieval Practice

Generates multiple retrieval queries from one user query,
runs semantic retrieval for each query, and combines the
results into one ranked result set.

Pipeline:

User Query
    ↓
Query Expansion
    ↓
Multiple Retrieval Queries
    ↓
Semantic Search
    ↓
Result Fusion
    ↓
Deduplication
    ↓
Final Top-K
"""

import re
from dataclasses import dataclass
from typing import Dict, List

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
class RetrievalResult:
    document: Document
    score: float
    matched_queries: int


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
# QUERY EXPANSION
# ============================================================

def generate_queries(
    query: str
) -> List[str]:
    """
    Generate multiple retrieval-oriented versions
    of the original query.
    """

    query = query.strip()

    queries = [
        query,

        f"Explain {query}",

        f"{query} detailed explanation",

        f"{query} concepts and fundamentals",

        f"{query} practical applications",
    ]

    # Remove duplicates while preserving order.

    unique_queries = []

    for item in queries:

        if item not in unique_queries:
            unique_queries.append(item)

    return unique_queries


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
# DOCUMENT INDEX
# ============================================================

def build_index(
    documents: List[Document]
):
    """
    Create document embeddings once.

    This avoids re-encoding the same documents
    for every query.
    """

    embeddings = model.encode(
        [document.text for document in documents],
        convert_to_numpy=True
    )

    return normalize_vectors(
        embeddings
    )


# ============================================================
# SINGLE QUERY RETRIEVAL
# ============================================================

def retrieve(
    query: str,
    documents: List[Document],
    document_embeddings: np.ndarray,
    top_k: int = 3
):
    """
    Retrieve documents for one query.
    """

    query_embedding = model.encode(
        query,
        convert_to_numpy=True
    )

    scores = cosine_similarity(
        query_embedding,
        document_embeddings
    )

    ranked_indices = np.argsort(
        scores
    )[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        results.append(
            (
                documents[index],
                float(scores[index])
            )
        )

    return results


# ============================================================
# RESULT FUSION
# ============================================================

def fuse_results(
    query_results: Dict[str, List],
    top_k: int = 5
) -> List[RetrievalResult]:
    """
    Combine results from multiple queries.

    A document receives additional support when it appears
    in multiple query result sets.

    Final score:

        average similarity
        + query coverage bonus
    """

    document_scores = {}

    document_matches = {}

    document_objects = {}

    # --------------------------------------------------------
    # Collect scores
    # --------------------------------------------------------

    for query, results in query_results.items():

        for document, score in results:

            doc_id = document.doc_id

            document_objects[doc_id] = document

            if doc_id not in document_scores:

                document_scores[doc_id] = []

            document_scores[doc_id].append(
                score
            )

            if doc_id not in document_matches:

                document_matches[doc_id] = 0

            document_matches[doc_id] += 1

    # --------------------------------------------------------
    # Calculate fused scores
    # --------------------------------------------------------

    fused_results = []

    total_queries = len(
        query_results
    )

    for doc_id, scores in document_scores.items():

        average_score = (
            sum(scores)
            /
            len(scores)
        )

        query_coverage = (
            len(scores)
            /
            total_queries
        )

        # Coverage provides a small bonus when a document
        # is relevant to multiple generated queries.

        fused_score = (
            average_score * 0.8
            +
            query_coverage * 0.2
        )

        fused_results.append(
            RetrievalResult(
                document=document_objects[doc_id],
                score=fused_score,
                matched_queries=document_matches[doc_id]
            )
        )

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    fused_results.sort(
        key=lambda result: result.score,
        reverse=True
    )

    return fused_results[:top_k]


# ============================================================
# DISPLAY
# ============================================================

def display_results(
    results: List[RetrievalResult],
    title: str
):
    """
    Display ranked results.
    """

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRank {rank}"
        )

        print(
            f"Document: "
            f"{result.document.doc_id}"
        )

        print(
            f"Fused Score: "
            f"{result.score:.4f}"
        )

        print(
            f"Matched Queries: "
            f"{result.matched_queries}"
        )

        print(
            f"Text: "
            f"{result.document.text}"
        )


# ============================================================
# SINGLE QUERY BASELINE
# ============================================================

def single_query_baseline(
    query: str,
    documents: List[Document],
    embeddings: np.ndarray,
    top_k: int
):
    """
    Retrieve using only the original query.

    Used as a baseline for comparison.
    """

    results = retrieve(
        query=query,
        documents=documents,
        document_embeddings=embeddings,
        top_k=top_k
    )

    converted = [
        RetrievalResult(
            document=document,
            score=score,
            matched_queries=1
        )
        for document, score in results
    ]

    return converted


# ============================================================
# MULTI-QUERY RETRIEVAL
# ============================================================

def multi_query_retrieve(
    query: str,
    documents: List[Document],
    embeddings: np.ndarray,
    per_query_k: int = 3,
    final_k: int = 5
):
    """
    Complete multi-query retrieval pipeline.
    """

    generated_queries = generate_queries(
        query
    )

    print("\n")
    print("=" * 70)
    print("GENERATED QUERIES")
    print("=" * 70)

    for index, generated_query in enumerate(
        generated_queries,
        start=1
    ):

        print(
            f"{index}. {generated_query}"
        )

    # --------------------------------------------------------
    # Retrieve for every generated query
    # --------------------------------------------------------

    query_results = {}

    for generated_query in generated_queries:

        results = retrieve(
            query=generated_query,
            documents=documents,
            document_embeddings=embeddings,
            top_k=per_query_k
        )

        query_results[
            generated_query
        ] = results

    # --------------------------------------------------------
    # Fuse results
    # --------------------------------------------------------

    return fuse_results(
        query_results=query_results,
        top_k=final_k
    )


# ============================================================
# SAMPLE KNOWLEDGE BASE
# ============================================================

documents = [

    Document(
        "doc_01",
        """
        Retrieval augmented generation combines document
        retrieval with a large language model to provide
        answers grounded in external knowledge.
        """
    ),

    Document(
        "doc_02",
        """
        RAG systems first retrieve relevant documents from
        a knowledge base and then provide the retrieved
        information to a language model.
        """
    ),

    Document(
        "doc_03",
        """
        Vector databases store numerical embeddings and
        allow systems to search for semantically similar
        documents.
        """
    ),

    Document(
        "doc_04",
        """
        Query rewriting improves retrieval by transforming
        vague or conversational questions into clearer
        search queries.
        """
    ),

    Document(
        "doc_05",
        """
        Reranking improves retrieval quality by reordering
        initially retrieved documents according to their
        relevance to the user query.
        """
    ),

    Document(
        "doc_06",
        """
        Large language models generate natural language
        responses based on instructions and provided context.
        """
    ),

    Document(
        "doc_07",
        """
        Hybrid search combines keyword matching and semantic
        similarity to improve information retrieval.
        """
    ),

    Document(
        "doc_08",
        """
        Chunking divides large documents into smaller pieces
        so retrieval systems can locate relevant information
        more precisely.
        """
    ),
]


# ============================================================
# MAIN EXPERIMENT
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("RAG MULTI-QUERY RETRIEVAL")
    print("=" * 70)

    query = (
        "How does RAG retrieve information "
        "to generate answers?"
    )

    # --------------------------------------------------------
    # Build document index
    # --------------------------------------------------------

    print(
        "\nBuilding document index..."
    )

    document_embeddings = build_index(
        documents
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline = single_query_baseline(
        query=query,
        documents=documents,
        embeddings=document_embeddings,
        top_k=5
    )

    display_results(
        baseline,
        "SINGLE-QUERY BASELINE"
    )

    # --------------------------------------------------------
    # Multi-query retrieval
    # --------------------------------------------------------

    multi_results = multi_query_retrieve(
        query=query,
        documents=documents,
        embeddings=document_embeddings,
        per_query_k=3,
        final_k=5
    )

    display_results(
        multi_results,
        "MULTI-QUERY RESULTS"
    )
