"""
RAG Retrieval Thresholding
--------------------------
Filters retrieved documents using configurable similarity thresholds.

Features:
- Similarity score filtering
- Configurable minimum threshold
- Top-K result limiting
- Adaptive threshold selection
- Retrieval statistics
- JSON report export

No external dependencies required.

Note:
Uses simulated similarity scores for demonstration.
Production systems should use calibrated embedding similarity scores.
"""

from dataclasses import dataclass, asdict
from typing import List
import json


@dataclass
class RetrievedDocument:
    document_id: str
    content: str
    similarity: float


@dataclass
class ThresholdReport:
    threshold: float
    total_documents: int
    accepted_documents: int
    rejected_documents: int
    average_similarity: float
    accepted: List[dict]
    rejected: List[dict]


class RetrievalThresholdFilter:
    """Filter retrieved documents using similarity scores."""

    def __init__(self, threshold: float = 0.65):
        self.set_threshold(threshold)

    def set_threshold(self, threshold: float) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")

        self.threshold = threshold

    def filter_documents(
        self,
        documents: List[RetrievedDocument],
        top_k: int = None,
    ) -> ThresholdReport:

        if top_k is not None and top_k < 0:
            raise ValueError("top_k cannot be negative.")

        seen_ids = set()

        for document in documents:
            if not document.document_id.strip():
                raise ValueError("Document IDs cannot be empty.")

            if document.document_id in seen_ids:
                raise ValueError(
                    f"Duplicate document ID: {document.document_id}"
                )

            seen_ids.add(document.document_id)

            if not 0.0 <= document.similarity <= 1.0:
                raise ValueError(
                    f"Invalid similarity score for "
                    f"{document.document_id}"
                )

        ranked_documents = sorted(
            documents,
            key=lambda item: item.similarity,
            reverse=True,
        )

        accepted = [
            document
            for document in ranked_documents
            if document.similarity >= self.threshold
        ]

        if top_k is not None:
            accepted = accepted[:top_k]

        accepted_ids = {
            document.document_id for document in accepted
        }

        rejected = [
            document
            for document in ranked_documents
            if document.document_id not in accepted_ids
        ]

        average_similarity = (
            sum(doc.similarity for doc in accepted) / len(accepted)
            if accepted else 0.0
        )

        return ThresholdReport(
            threshold=self.threshold,
            total_documents=len(documents),
            accepted_documents=len(accepted),
            rejected_documents=len(rejected),
            average_similarity=round(average_similarity, 3),
            accepted=[asdict(doc) for doc in accepted],
            rejected=[asdict(doc) for doc in rejected],
        )

    @staticmethod
    def suggest_threshold(
        documents: List[RetrievedDocument],
    ) -> float:
        """
        Suggest a threshold based on the median score.

        This is only a demonstration heuristic, not a
        statistically calibrated threshold-selection method.
        """

        if not documents:
            return 0.5

        scores = sorted(
            document.similarity for document in documents
        )

        middle = len(scores) // 2

        if len(scores) % 2 == 0:
            median = (
                scores[middle - 1] + scores[middle]
            ) / 2
        else:
            median = scores[middle]

        return round(median, 3)


def save_report(
    report: ThresholdReport,
    filepath: str,
) -> None:

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(asdict(report), file, indent=4)

    print(f"\nReport saved to: {filepath}")


def main():
    documents = [
        RetrievedDocument(
            "D1",
            "Neural networks learn patterns from training data.",
            0.94,
        ),
        RetrievedDocument(
            "D2",
            "Gradient descent optimizes model parameters.",
            0.89,
        ),
        RetrievedDocument(
            "D3",
            "Backpropagation calculates gradients.",
            0.78,
        ),
        RetrievedDocument(
            "D4",
            "Database indexing improves query performance.",
            0.52,
        ),
        RetrievedDocument(
            "D5",
            "Weather forecasting uses atmospheric measurements.",
            0.31,
        ),
    ]

    threshold = 0.65

    retrieval_filter = RetrievalThresholdFilter(
        threshold=threshold
    )

    report = retrieval_filter.filter_documents(
        documents=documents,
        top_k=3,
    )

    print("RAG RETRIEVAL THRESHOLDING")
    print("=" * 50)

    print(f"Threshold: {report.threshold}")
    print(f"Total documents: {report.total_documents}")
    print(f"Accepted: {report.accepted_documents}")
    print(f"Rejected: {report.rejected_documents}")
    print(
        f"Average accepted similarity: "
        f"{report.average_similarity}"
    )

    print("\nACCEPTED DOCUMENTS")

    for document in report.accepted:
        print(
            f"{document['document_id']} - "
            f"{document['similarity']:.2f}"
        )

    print("\nREJECTED DOCUMENTS")

    for document in report.rejected:
        print(
            f"{document['document_id']} - "
            f"{document['similarity']:.2f}"
        )

    suggested = RetrievalThresholdFilter.suggest_threshold(
        documents
    )

    print(f"\nSuggested median threshold: {suggested}")

    save_report(report, "retrieval_threshold_report.json")


if __name__ == "__main__":
    main()
