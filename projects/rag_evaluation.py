"""
RAG Retrieval Evaluation Practice

Evaluates retrieval quality using:
- Precision@K
- Recall@K
- Hit Rate@K
- Mean Reciprocal Rank (MRR)

The evaluation uses known relevant chunk IDs as ground truth.
"""

from dataclasses import dataclass
from typing import List, Dict


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class RetrievalResult:
    query: str
    retrieved_chunks: List[str]
    relevant_chunks: List[str]


# ============================================================
# METRICS
# ============================================================

def precision_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int
) -> float:
    """
    Precision@K = relevant retrieved chunks / K
    """

    top_k = retrieved[:k]

    if not top_k:
        return 0.0

    relevant_retrieved = sum(
        chunk in relevant
        for chunk in top_k
    )

    return relevant_retrieved / len(top_k)


def recall_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int
) -> float:
    """
    Recall@K = relevant retrieved chunks / total relevant chunks
    """

    if not relevant:
        return 0.0

    top_k = retrieved[:k]

    relevant_retrieved = sum(
        chunk in relevant
        for chunk in top_k
    )

    return relevant_retrieved / len(relevant)


def hit_rate_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int
) -> float:
    """
    Hit Rate@K = 1 if at least one relevant chunk
    appears in top-K, otherwise 0.
    """

    top_k = retrieved[:k]

    return float(
        any(chunk in relevant for chunk in top_k)
    )


def reciprocal_rank(
    retrieved: List[str],
    relevant: List[str]
) -> float:
    """
    Reciprocal Rank = 1 / rank of first relevant result.
    """

    for rank, chunk in enumerate(retrieved, start=1):

        if chunk in relevant:
            return 1 / rank

    return 0.0


# ============================================================
# MRR
# ============================================================

def mean_reciprocal_rank(
    results: List[RetrievalResult]
) -> float:
    """
    MRR = average reciprocal rank across queries.
    """

    if not results:
        return 0.0

    scores = [
        reciprocal_rank(
            result.retrieved_chunks,
            result.relevant_chunks
        )
        for result in results
    ]

    return sum(scores) / len(scores)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_retrieval(
    results: List[RetrievalResult],
    k: int
) -> Dict[str, float]:

    precision_scores = []
    recall_scores = []
    hit_scores = []

    for result in results:

        precision_scores.append(
            precision_at_k(
                result.retrieved_chunks,
                result.relevant_chunks,
                k
            )
        )

        recall_scores.append(
            recall_at_k(
                result.retrieved_chunks,
                result.relevant_chunks,
                k
            )
        )

        hit_scores.append(
            hit_rate_at_k(
                result.retrieved_chunks,
                result.relevant_chunks,
                k
            )
        )

    return {
        "precision@k": sum(precision_scores)
        / len(precision_scores),

        "recall@k": sum(recall_scores)
        / len(recall_scores),

        "hit_rate@k": sum(hit_scores)
        / len(hit_scores),

        "mrr": mean_reciprocal_rank(results)
    }


# ============================================================
# SAMPLE GROUND TRUTH
# ============================================================

evaluation_dataset = [

    RetrievalResult(
        query="What is machine learning?",
        retrieved_chunks=[
            "chunk_03",
            "chunk_01",
            "chunk_07",
            "chunk_05"
        ],
        relevant_chunks=[
            "chunk_01",
            "chunk_03"
        ]
    ),

    RetrievalResult(
        query="What is supervised learning?",
        retrieved_chunks=[
            "chunk_08",
            "chunk_04",
            "chunk_02",
            "chunk_09"
        ],
        relevant_chunks=[
            "chunk_02",
            "chunk_04"
        ]
    ),

    RetrievalResult(
        query="What is overfitting?",
        retrieved_chunks=[
            "chunk_06",
            "chunk_10",
            "chunk_03",
            "chunk_01"
        ],
        relevant_chunks=[
            "chunk_06"
        ]
    ),

    RetrievalResult(
        query="What is gradient descent?",
        retrieved_chunks=[
            "chunk_11",
            "chunk_05",
            "chunk_12",
            "chunk_03"
        ],
        relevant_chunks=[
            "chunk_12"
        ]
    )
]


# ============================================================
# QUERY-LEVEL EVALUATION
# ============================================================

def evaluate_queries(
    results: List[RetrievalResult],
    k: int
) -> None:

    print("=" * 60)
    print(f"QUERY LEVEL EVALUATION — TOP {k}")
    print("=" * 60)

    for result in results:

        precision = precision_at_k(
            result.retrieved_chunks,
            result.relevant_chunks,
            k
        )

        recall = recall_at_k(
            result.retrieved_chunks,
            result.relevant_chunks,
            k
        )

        hit = hit_rate_at_k(
            result.retrieved_chunks,
            result.relevant_chunks,
            k
        )

        rr = reciprocal_rank(
            result.retrieved_chunks,
            result.relevant_chunks
        )

        print(f"\nQuery: {result.query}")

        print(f"Retrieved: {result.retrieved_chunks[:k]}")
        print(f"Relevant:  {result.relevant_chunks}")

        print(f"Precision@{k}: {precision:.2f}")
        print(f"Recall@{k}:    {recall:.2f}")
        print(f"Hit Rate@{k}:  {hit:.2f}")
        print(f"Reciprocal Rank: {rr:.2f}")


# ============================================================
# COMPARE DIFFERENT TOP-K VALUES
# ============================================================

def compare_k_values(
    results: List[RetrievalResult],
    k_values: List[int]
) -> None:

    print("\n")
    print("=" * 60)
    print("TOP-K COMPARISON")
    print("=" * 60)

    for k in k_values:

        metrics = evaluate_retrieval(
            results,
            k
        )

        print(
            f"\nK = {k}"
        )

        print(
            f"Precision@{k}: "
            f"{metrics['precision@k']:.2f}"
        )

        print(
            f"Recall@{k}: "
            f"{metrics['recall@k']:.2f}"
        )

        print(
            f"Hit Rate@{k}: "
            f"{metrics['hit_rate@k']:.2f}"
        )

        print(
            f"MRR: "
            f"{metrics['mrr']:.2f}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\nRAG RETRIEVAL EVALUATION")
    print("=" * 60)

    # Detailed evaluation
    evaluate_queries(
        evaluation_dataset,
        k=3
    )

    # Compare retrieval configurations
    compare_k_values(
        evaluation_dataset,
        k_values=[1, 2, 3, 4]
    )

    # Overall evaluation
    metrics = evaluate_retrieval(
        evaluation_dataset,
        k=3
    )

    print("\n")
    print("=" * 60)
    print("OVERALL RETRIEVAL QUALITY")
    print("=" * 60)

    for metric, score in metrics.items():

        print(
            f"{metric}: {score:.3f}"
        )
