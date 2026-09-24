"""
RAG Reranking Practice

Pipeline:

Query
  ↓
Initial Retrieval
  ↓
Candidate Chunks
  ↓
Hybrid Reranking
  ↓
Final Top-K Context

Uses:
- sentence-transformers
- NumPy
- keyword overlap
- semantic similarity
"""

import re
from dataclasses import dataclass
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Chunk:
    chunk_id: str
    text: str


@dataclass
class RankedChunk:
    chunk: Chunk
    semantic_score: float
    keyword_score: float
    final_score: float


# ============================================================
# TEXT PROCESSING
# ============================================================

def tokenize(text: str) -> List[str]:
    """
    Convert text into normalized word tokens.
    """

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


def keyword_overlap(
    query: str,
    text: str
) -> float:
    """
    Calculate keyword overlap between query and chunk.

    Score range:
        0.0 → no overlap
        1.0 → all query keywords found
    """

    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))

    if not query_tokens:
        return 0.0

    overlap = query_tokens.intersection(
        text_tokens
    )

    return len(overlap) / len(query_tokens)


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

    norms = np.maximum(norms, 1e-12)

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
# INITIAL RETRIEVAL
# ============================================================

def retrieve_candidates(
    query: str,
    chunks: List[Chunk],
    top_n: int = 5
):
    """
    Retrieve initial candidates using embeddings.
    """

    query_embedding = model.encode(
        query,
        convert_to_numpy=True
    )

    chunk_embeddings = model.encode(
        [chunk.text for chunk in chunks],
        convert_to_numpy=True
    )

    scores = cosine_similarity(
        query_embedding,
        chunk_embeddings
    )

    ranked_indices = np.argsort(
        scores
    )[::-1]

    candidates = []

    for index in ranked_indices[:top_n]:

        candidates.append(
            (
                chunks[index],
                float(scores[index])
            )
        )

    return candidates


# ============================================================
# RERANKING
# ============================================================

def rerank(
    query: str,
    candidates,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3
) -> List[RankedChunk]:
    """
    Rerank retrieved candidates.

    Final score:

        semantic_score * semantic_weight
        +
        keyword_score * keyword_weight
    """

    ranked_chunks = []

    for chunk, semantic_score in candidates:

        keyword_score = keyword_overlap(
            query,
            chunk.text
        )

        final_score = (
            semantic_score * semantic_weight
            +
            keyword_score * keyword_weight
        )

        ranked_chunks.append(
            RankedChunk(
                chunk=chunk,
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                final_score=final_score
            )
        )

    ranked_chunks.sort(
        key=lambda item: item.final_score,
        reverse=True
    )

    return ranked_chunks


# ============================================================
# DISPLAY
# ============================================================

def print_results(
    results: List[RankedChunk],
    title: str
):
    """
    Display ranked results.
    """

    print("\n" + "=" * 70)
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
            f"Chunk ID: {result.chunk.chunk_id}"
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
            f"Final Score:    "
            f"{result.final_score:.4f}"
        )

        print(
            f"Text: {result.chunk.text}"
        )


# ============================================================
# SAMPLE DOCUMENT
# ============================================================

chunks = [

    Chunk(
        "chunk_01",
        """
        Machine learning is a branch of artificial
        intelligence that allows systems to learn
        patterns from data.
        """
    ),

    Chunk(
        "chunk_02",
        """
        Supervised learning trains a machine learning
        model using labeled training data.
        """
    ),

    Chunk(
        "chunk_03",
        """
        Deep learning uses neural networks with
        multiple layers to learn complex patterns.
        """
    ),

    Chunk(
        "chunk_04",
        """
        Gradient descent is an optimization algorithm
        used to minimize the loss function during
        machine learning model training.
        """
    ),

    Chunk(
        "chunk_05",
        """
        Overfitting happens when a model learns the
        training data too closely and performs poorly
        on unseen data.
        """
    ),

    Chunk(
        "chunk_06",
        """
        Feature engineering transforms raw data into
        useful features that improve machine learning
        model performance.
        """
    ),

    Chunk(
        "chunk_07",
        """
        Classification predicts discrete categories,
        while regression predicts continuous numerical
        values.
        """
    ),
]


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment(
    query: str,
    top_n: int = 5,
    final_k: int = 3
):

    print("\n")
    print("=" * 70)
    print("RAG RERANKING EXPERIMENT")
    print("=" * 70)

    print(
        f"\nQuery: {query}"
    )

    # --------------------------------------------------------
    # STEP 1: INITIAL RETRIEVAL
    # --------------------------------------------------------

    candidates = retrieve_candidates(
        query=query,
        chunks=chunks,
        top_n=top_n
    )

    print_results(
        [
            RankedChunk(
                chunk=chunk,
                semantic_score=score,
                keyword_score=0.0,
                final_score=score
            )
            for chunk, score in candidates
        ],
        "INITIAL RETRIEVAL"
    )

    # --------------------------------------------------------
    # STEP 2: RERANK
    # --------------------------------------------------------

    reranked = rerank(
        query=query,
        candidates=candidates
    )

    # --------------------------------------------------------
    # STEP 3: FINAL TOP-K
    # --------------------------------------------------------

    final_results = reranked[:final_k]

    print_results(
        final_results,
        "RERANKED RESULTS"
    )


# ============================================================
# COMPARE WEIGHTS
# ============================================================

def compare_weights(
    query: str,
    candidates
):
    """
    Compare different semantic/keyword weight combinations.
    """

    configurations = [
        (1.0, 0.0),
        (0.8, 0.2),
        (0.7, 0.3),
        (0.5, 0.5),
    ]

    print("\n")
    print("=" * 70)
    print("RERANKING WEIGHT EXPERIMENT")
    print("=" * 70)

    for semantic_weight, keyword_weight in configurations:

        results = rerank(
            query=query,
            candidates=candidates,
            semantic_weight=semantic_weight,
            keyword_weight=keyword
        )

        print(
            f"\nSemantic={semantic_weight:.1f} | "
            f"Keyword={keyword_weight:.1f}"
        )

        for rank, result in enumerate(
            results[:3],
            start=1
        ):

            print(
                f"{rank}. "
                f"{result.chunk.chunk_id} "
                f"-> "
                f"{result.final_score:.4f}"
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    query = (
        "How does gradient descent train "
        "a machine learning model?"
    )

    # Run complete reranking experiment
    run_experiment(
        query=query,
        top_n=5,
        final_k=3
    )

    # Retrieve candidates again for weight experiment
    candidates = retrieve_candidates(
        query=query,
        chunks=chunks,
        top_n=5
    )

    compare_weights(
        query=query,
        candidates=candidates
    )
