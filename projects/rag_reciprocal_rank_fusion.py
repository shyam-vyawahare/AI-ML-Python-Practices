"""
RAG Reciprocal Rank Fusion (RRF) Practice

Combines multiple ranked retrieval results using
Reciprocal Rank Fusion.

RRF formula:

    RRF(d) = Σ 1 / (k + rank(d))

Where:

    d    = document
    rank = document position in a result list
    k    = ranking constant

The important idea:

RRF combines rankings rather than raw similarity scores.
"""

from dataclasses import dataclass
from typing import Dict, List


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Document:
    doc_id: str
    text: str


@dataclass
class RankedDocument:
    doc_id: str
    rank: int
    score: float


@dataclass
class RRFResult:
    document: Document
    rrf_score: float
    appearances: int


# ============================================================
# SAMPLE DOCUMENTS
# ============================================================

documents = [

    Document(
        "doc_01",
        """
        Retrieval augmented generation combines document
        retrieval with language models to generate grounded
        answers.
        """
    ),

    Document(
        "doc_02",
        """
        RAG systems retrieve relevant documents from a
        knowledge base before generating an answer.
        """
    ),

    Document(
        "doc_03",
        """
        Vector databases store embeddings and support
        similarity-based document retrieval.
        """
    ),

    Document(
        "doc_04",
        """
        Query rewriting transforms vague questions into
        retrieval-friendly search queries.
        """
    ),

    Document(
        "doc_05",
        """
        Reranking reorders retrieved documents according
        to their relevance to a query.
        """
    ),

    Document(
        "doc_06",
        """
        Hybrid search combines keyword matching with
        semantic similarity.
        """
    ),

    Document(
        "doc_07",
        """
        Chunking divides large documents into smaller
        units for more precise retrieval.
        """
    ),

    Document(
        "doc_08",
        """
        Large language models generate responses using
        instructions and contextual information.
        """
    ),
]


# ============================================================
# SAMPLE RETRIEVAL RESULTS
# ============================================================

semantic_results = [
    "doc_01",
    "doc_03",
    "doc_02",
    "doc_05",
    "doc_07",
]


keyword_results = [
    "doc_02",
    "doc_06",
    "doc_01",
    "doc_04",
    "doc_05",
]


query_variation_results = [
    "doc_01",
    "doc_02",
    "doc_05",
    "doc_06",
    "doc_03",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_document_map(
    documents: List[Document]
) -> Dict[str, Document]:
    """
    Create a document ID → Document mapping.
    """

    return {
        document.doc_id: document
        for document in documents
    }


def display_ranked_results(
    results: List[str],
    title: str
):
    """
    Display one ranked result list.
    """

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

    for rank, doc_id in enumerate(
        results,
        start=1
    ):

        print(
            f"{rank}. {doc_id}"
        )


# ============================================================
# RRF SCORE
# ============================================================

def calculate_rrf_score(
    rankings: List[List[str]],
    k: int = 60
) -> Dict[str, float]:
    """
    Calculate Reciprocal Rank Fusion scores.

    Formula:

        RRF(d) = Σ 1 / (k + rank)

    Rank starts from 1.
    """

    scores = {}

    for ranking in rankings:

        for rank, doc_id in enumerate(
            ranking,
            start=1
        ):

            contribution = (
                1 / (k + rank)
            )

            scores[doc_id] = (
                scores.get(doc_id, 0.0)
                +
                contribution
            )

    return scores


# ============================================================
# COUNT APPEARANCES
# ============================================================

def count_appearances(
    rankings: List[List[str]]
) -> Dict[str, int]:
    """
    Count how many ranking lists contain each document.
    """

    appearances = {}

    for ranking in rankings:

        for doc_id in ranking:

            appearances[doc_id] = (
                appearances.get(doc_id, 0)
                + 1
            )

    return appearances


# ============================================================
# RRF FUSION
# ============================================================

def reciprocal_rank_fusion(
    rankings: List[List[str]],
    documents: List[Document],
    k: int = 60,
    top_n: int = 5
) -> List[RRFResult]:
    """
    Fuse multiple rankings using RRF.
    """

    scores = calculate_rrf_score(
        rankings,
        k=k
    )

    appearances = count_appearances(
        rankings
    )

    document_map = create_document_map(
        documents
    )

    sorted_documents = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    results = []

    for doc_id, score in sorted_documents[:top_n]:

        results.append(
            RRFResult(
                document=document_map[doc_id],
                rrf_score=score,
                appearances=appearances[doc_id]
            )
        )

    return results


# ============================================================
# DISPLAY RRF RESULTS
# ============================================================

def display_rrf_results(
    results: List[RRFResult]
):
    """
    Display RRF ranked results.
    """

    print("\n")
    print("=" * 70)
    print("RRF FUSED RESULTS")
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
            f"RRF Score: "
            f"{result.rrf_score:.6f}"
        )

        print(
            f"Appearances: "
            f"{result.appearances}"
        )

        print(
            f"Text: "
            f"{result.document.text.strip()}"
        )


# ============================================================
# RANK CONTRIBUTION ANALYSIS
# ============================================================

def analyze_rank_contributions(
    rankings: List[List[str]],
    k: int = 60
):
    """
    Show how each ranking contributes to the final RRF score.
    """

    contributions = {}

    for ranking_index, ranking in enumerate(
        rankings,
        start=1
    ):

        for rank, doc_id in enumerate(
            ranking,
            start=1
        ):

            score = 1 / (k + rank)

            if doc_id not in contributions:

                contributions[doc_id] = {}

            contributions[doc_id][
                f"ranking_{ranking_index}"
            ] = score

    print("\n")
    print("=" * 70)
    print("RRF CONTRIBUTION ANALYSIS")
    print("=" * 70)

    for doc_id, data in contributions.items():

        total = sum(
            data.values()
        )

        print(
            f"\n{doc_id}"
        )

        for ranking_name, score in data.items():

            print(
                f"  {ranking_name}: "
                f"{score:.6f}"
            )

        print(
            f"  Total: "
            f"{total:.6f}"
        )


# ============================================================
# K PARAMETER EXPERIMENT
# ============================================================

def compare_k_values(
    rankings: List[List[str]],
    documents: List[Document]
):
    """
    Compare different RRF k values.
    """

    k_values = [
        1,
        10,
        30,
        60,
        100,
    ]

    print("\n")
    print("=" * 70)
    print("RRF K PARAMETER EXPERIMENT")
    print("=" * 70)

    for k in k_values:

        results = reciprocal_rank_fusion(
            rankings=rankings,
            documents=documents,
            k=k,
            top_n=5
        )

        print(
            f"\nK = {k}"
        )

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"{rank}. "
                f"{result.document.doc_id} "
                f"-> "
                f"{result.rrf_score:.6f}"
            )


# ============================================================
# UNIQUE DOCUMENTS
# ============================================================

def get_all_documents(
    rankings: List[List[str]]
) -> List[str]:
    """
    Return all unique document IDs appearing
    in any ranking.
    """

    unique_documents = []

    for ranking in rankings:

        for doc_id in ranking:

            if doc_id not in unique_documents:

                unique_documents.append(
                    doc_id
                )

    return unique_documents


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("RECIPROCAL RANK FUSION FOR RAG")
    print("=" * 70)

    rankings = [
        semantic_results,
        keyword_results,
        query_variation_results,
    ]

    # --------------------------------------------------------
    # Display individual rankings
    # --------------------------------------------------------

    display_ranked_results(
        semantic_results,
        "SEMANTIC SEARCH RANKING"
    )

    display_ranked_results(
        keyword_results,
        "KEYWORD SEARCH RANKING"
    )

    display_ranked_results(
        query_variation_results,
        "QUERY VARIATION RANKING"
    )

    # --------------------------------------------------------
    # Show unique documents
    # --------------------------------------------------------

    unique_documents = get_all_documents(
        rankings
    )

    print("\n")
    print("=" * 70)
    print("UNIQUE DOCUMENTS")
    print("=" * 70)

    print(
        unique_documents
    )

    # --------------------------------------------------------
    # RRF fusion
    # --------------------------------------------------------

    results = reciprocal_rank_fusion(
        rankings=rankings,
        documents=documents,
        k=60,
        top_n=5
    )

    display_rrf_results(
        results
    )

    # --------------------------------------------------------
    # Contribution analysis
    # --------------------------------------------------------

    analyze_rank_contributions(
        rankings,
        k=60
    )

    # --------------------------------------------------------
    # K experiment
    # --------------------------------------------------------

    compare_k_values(
        rankings,
        documents
    )
