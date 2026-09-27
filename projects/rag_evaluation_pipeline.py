"""
RAG Evaluation Pipeline

Evaluates a complete RAG system using:

Retrieval metrics:
    - Precision@K
    - Recall@K
    - Hit Rate@K
    - Reciprocal Rank
    - MRR

Generation metrics:
    - Claim support
    - Grounding score
    - Grounding status

This is a lightweight, dependency-free evaluation framework
designed for learning RAG quality engineering.
"""

import re
from dataclasses import dataclass
from typing import List


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class RAGTestCase:
    query: str

    retrieved_chunks: List[str]
    relevant_chunks: List[str]

    answer: str
    context: str


@dataclass
class QueryEvaluation:
    query: str

    precision: float
    recall: float
    hit_rate: float
    reciprocal_rank: float

    grounding_score: float
    grounding_status: str


# ============================================================
# TOKENIZATION
# ============================================================

STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "with",
    "that",
    "this",
    "it",
    "as",
    "by",
    "from",
    "can",
    "be",
    "used",
    "uses",
    "using",
    "into",
    "its",
}


def tokenize(text: str) -> List[str]:
    """
    Convert text into normalized tokens.
    """

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


def meaningful_tokens(text: str) -> List[str]:
    """
    Remove common stopwords.
    """

    return [
        token
        for token in tokenize(text)
        if token not in STOPWORDS
    ]


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def precision_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int
) -> float:
    """
    Percentage of top-K retrieved chunks
    that are relevant.
    """

    top_k = retrieved[:k]

    if not top_k:
        return 0.0

    relevant_count = sum(
        chunk in relevant
        for chunk in top_k
    )

    return relevant_count / len(top_k)


def recall_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int
) -> float:
    """
    Percentage of relevant chunks successfully retrieved.
    """

    if not relevant:
        return 0.0

    top_k = retrieved[:k]

    relevant_count = sum(
        chunk in relevant
        for chunk in top_k
    )

    return relevant_count / len(relevant)


def hit_rate_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int
) -> float:
    """
    Returns 1 if at least one relevant chunk
    appears in top-K.
    """

    top_k = retrieved[:k]

    return float(
        any(
            chunk in relevant
            for chunk in top_k
        )
    )


def reciprocal_rank(
    retrieved: List[str],
    relevant: List[str]
) -> float:
    """
    Reciprocal rank of the first relevant result.
    """

    for rank, chunk in enumerate(
        retrieved,
        start=1
    ):

        if chunk in relevant:
            return 1 / rank

    return 0.0


# ============================================================
# GROUNDING METRICS
# ============================================================

def extract_claims(
    answer: str
) -> List[str]:
    """
    Split an answer into sentence-level claims.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        answer.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def claim_support_score(
    claim: str,
    context: str
) -> float:
    """
    Estimate claim support using token overlap.
    """

    claim_terms = set(
        meaningful_tokens(claim)
    )

    context_terms = set(
        meaningful_tokens(context)
    )

    if not claim_terms:
        return 0.0

    matched_terms = (
        claim_terms.intersection(
            context_terms
        )
    )

    return (
        len(matched_terms)
        /
        len(claim_terms)
    )


def evaluate_grounding(
    answer: str,
    context: str
):
    """
    Calculate average claim support and classify
    the answer's grounding.
    """

    claims = extract_claims(answer)

    if not claims:
        return 0.0, "NO CLAIMS"

    scores = [
        claim_support_score(
            claim,
            context
        )
        for claim in claims
    ]

    grounding_score = (
        sum(scores) / len(scores)
    )

    strongly_supported = sum(
        score >= 0.5
        for score in scores
    )

    support_ratio = (
        strongly_supported
        /
        len(scores)
    )

    if support_ratio >= 0.8:
        status = "GROUNDED"

    elif support_ratio >= 0.5:
        status = "PARTIALLY GROUNDED"

    else:
        status = "POTENTIAL HALLUCINATION"

    return grounding_score, status


# ============================================================
# SINGLE QUERY EVALUATION
# ============================================================

def evaluate_test_case(
    test_case: RAGTestCase,
    k: int = 3
) -> QueryEvaluation:
    """
    Evaluate retrieval and generation quality
    for one query.
    """

    precision = precision_at_k(
        test_case.retrieved_chunks,
        test_case.relevant_chunks,
        k
    )

    recall = recall_at_k(
        test_case.retrieved_chunks,
        test_case.relevant_chunks,
        k
    )

    hit_rate = hit_rate_at_k(
        test_case.retrieved_chunks,
        test_case.relevant_chunks,
        k
    )

    rr = reciprocal_rank(
        test_case.retrieved_chunks,
        test_case.relevant_chunks
    )

    grounding_score, grounding_status = (
        evaluate_grounding(
            test_case.answer,
            test_case.context
        )
    )

    return QueryEvaluation(
        query=test_case.query,

        precision=precision,
        recall=recall,
        hit_rate=hit_rate,
        reciprocal_rank=rr,

        grounding_score=grounding_score,
        grounding_status=grounding_status
    )


# ============================================================
# PIPELINE EVALUATION
# ============================================================

def evaluate_pipeline(
    test_cases: List[RAGTestCase],
    k: int = 3
):
    """
    Evaluate all test cases.
    """

    evaluations = []

    for test_case in test_cases:

        evaluation = evaluate_test_case(
            test_case,
            k=k
        )

        evaluations.append(
            evaluation
        )

    return evaluations


# ============================================================
# AGGREGATE METRICS
# ============================================================

def aggregate_metrics(
    evaluations: List[QueryEvaluation]
):
    """
    Calculate average metrics across all queries.
    """

    if not evaluations:
        return {}

    return {
        "precision@k": sum(
            item.precision
            for item in evaluations
        ) / len(evaluations),

        "recall@k": sum(
            item.recall
            for item in evaluations
        ) / len(evaluations),

        "hit_rate@k": sum(
            item.hit_rate
            for item in evaluations
        ) / len(evaluations),

        "mrr": sum(
            item.reciprocal_rank
            for item in evaluations
        ) / len(evaluations),

        "grounding_score": sum(
            item.grounding_score
            for item in evaluations
        ) / len(evaluations),
    }


# ============================================================
# QUALITY SUMMARY
# ============================================================

def quality_summary(
    metrics
):
    """
    Produce a simple diagnostic summary.

    This is NOT a universal quality score.
    It is only a learning-oriented diagnostic.
    """

    print("\n")
    print("=" * 70)
    print("RAG QUALITY DIAGNOSTICS")
    print("=" * 70)

    precision = metrics["precision@k"]
    recall = metrics["recall@k"]
    hit_rate = metrics["hit_rate@k"]
    mrr = metrics["mrr"]
    grounding = metrics["grounding_score"]

    print(
        f"\nPrecision@K : {precision:.3f}"
    )

    print(
        f"Recall@K    : {recall:.3f}"
    )

    print(
        f"Hit Rate@K  : {hit_rate:.3f}"
    )

    print(
        f"MRR         : {mrr:.3f}"
    )

    print(
        f"Grounding   : {grounding:.3f}"
    )

    print("\nDiagnostic:")

    if recall < 0.5:
        print(
            "- Retrieval is missing many relevant chunks."
        )

    if precision < 0.5:
        print(
            "- Retrieval is returning many irrelevant chunks."
        )

    if mrr < 0.5:
        print(
            "- Relevant information often appears too low."
        )

    if grounding < 0.5:
        print(
            "- Generated answers contain weakly supported claims."
        )

    if (
        precision >= 0.5
        and recall >= 0.5
        and grounding >= 0.5
    ):
        print(
            "- Retrieval and grounding signals are reasonably strong."
        )


# ============================================================
# REPORT
# ============================================================

def print_query_report(
    evaluations: List[QueryEvaluation]
):
    """
    Display per-query evaluation results.
    """

    print("\n")
    print("=" * 70)
    print("QUERY EVALUATION REPORT")
    print("=" * 70)

    for index, item in enumerate(
        evaluations,
        start=1
    ):

        print(
            f"\nQuery {index}: "
            f"{item.query}"
        )

        print(
            f"Precision@K: "
            f"{item.precision:.2f}"
        )

        print(
            f"Recall@K: "
            f"{item.recall:.2f}"
        )

        print(
            f"Hit Rate@K: "
            f"{item.hit_rate:.2f}"
        )

        print(
            f"Reciprocal Rank: "
            f"{item.reciprocal_rank:.2f}"
        )

        print(
            f"Grounding Score: "
            f"{item.grounding_score:.2f}"
        )

        print(
            f"Grounding Status: "
            f"{item.grounding_status}"
        )


# ============================================================
# SAMPLE DATASET
# ============================================================

test_cases = [

    RAGTestCase(
        query="What is machine learning?",

        retrieved_chunks=[
            "chunk_03",
            "chunk_01",
            "chunk_07",
        ],

        relevant_chunks=[
            "chunk_01",
            "chunk_03",
        ],

        answer=(
            "Machine learning allows computers to learn "
            "patterns from data. It can be used to build "
            "models without explicitly programming every rule."
        ),

        context=(
            "Machine learning allows computers to learn "
            "patterns from data without being explicitly "
            "programmed for every task."
        )
    ),

    RAGTestCase(
        query="What is gradient descent?",

        retrieved_chunks=[
            "chunk_05",
            "chunk_04",
            "chunk_02",
        ],

        relevant_chunks=[
            "chunk_04",
        ],

        answer=(
            "Gradient descent is an optimization algorithm "
            "used to minimize a loss function during model "
            "training."
        ),

        context=(
            "Gradient descent is an optimization algorithm "
            "used to minimize the loss function while training "
            "machine learning models."
        )
    ),

    RAGTestCase(
        query="What is RAG?",

        retrieved_chunks=[
            "chunk_09",
            "chunk_08",
            "chunk_10",
        ],

        relevant_chunks=[
            "chunk_09",
        ],

        answer=(
            "RAG combines document retrieval with a language "
            "model to generate grounded answers. It guarantees "
            "that every generated answer is completely accurate."
        ),

        context=(
            "Retrieval augmented generation combines document "
            "retrieval with a language model to generate "
            "grounded answers."
        )
    ),

    RAGTestCase(
        query="What is overfitting?",

        retrieved_chunks=[
            "chunk_11",
            "chunk_12",
            "chunk_06",
        ],

        relevant_chunks=[
            "chunk_12",
        ],

        answer=(
            "Overfitting occurs when a model learns the "
            "training data too closely and performs poorly "
            "on unseen data."
        ),

        context=(
            "Overfitting occurs when a machine learning model "
            "learns the training data too closely and performs "
            "poorly on unseen data."
        )
    ),
]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("RAG EVALUATION PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Evaluate complete dataset
    # --------------------------------------------------------

    evaluations = evaluate_pipeline(
        test_cases,
        k=3
    )

    # --------------------------------------------------------
    # Query-level report
    # --------------------------------------------------------

    print_query_report(
        evaluations
    )

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    metrics = aggregate_metrics(
        evaluations
    )

    print("\n")
    print("=" * 70)
    print("AGGREGATE RAG METRICS")
    print("=" * 70)

    for name, value in metrics.items():

        print(
            f"{name}: {value:.3f}"
        )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    quality_summary(
        metrics
    )
