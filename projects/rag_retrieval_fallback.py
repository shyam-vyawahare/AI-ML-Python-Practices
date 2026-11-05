"""
RAG Retrieval Fallback
----------------------
Implements fault-tolerant retrieval using alternative strategies.

Features:
- Primary retriever execution
- Automatic fallback strategies
- Exception handling
- Empty-result detection
- Minimum relevance validation
- Retrieval attempt logging
- JSON execution report

No external dependencies required.

Note:
Retrieval strategies are simulated for demonstration.
"""

from dataclasses import dataclass, asdict
from typing import List, Callable
import json


@dataclass
class RetrievedDocument:
    document_id: str
    content: str
    score: float


@dataclass
class RetrievalAttempt:
    strategy: str
    status: str
    result_count: int
    error: str


@dataclass
class FallbackReport:
    query: str
    successful_strategy: str
    fallback_count: int
    final_status: str
    documents: List[dict]
    attempts: List[dict]


class RetrievalFallback:
    """Execute alternative retrievers when earlier strategies fail."""

    def __init__(self, minimum_score: float = 0.6):
        if not 0 <= minimum_score <= 1:
            raise ValueError("Minimum score must be between 0 and 1.")

        self.minimum_score = minimum_score

    def primary_retriever(
        self, query: str
    ) -> List[RetrievedDocument]:
        """Simulated primary vector retriever."""

        return [
            RetrievedDocument(
                "D1",
                "Weak vector search result.",
                0.35,
            )
        ]

    def keyword_retriever(
        self, query: str
    ) -> List[RetrievedDocument]:
        """Simulated keyword-based fallback."""

        return [
            RetrievedDocument(
                "D2",
                "Keyword matching result.",
                0.72,
            ),
            RetrievedDocument(
                "D3",
                "Additional keyword result.",
                0.81,
            ),
        ]

    def secondary_vector_retriever(
        self, query: str
    ) -> List[RetrievedDocument]:
        """Simulated secondary vector database."""

        return [
            RetrievedDocument(
                "D4",
                "Secondary vector database result.",
                0.88,
            )
        ]

    def validate_results(
        self,
        documents: List[RetrievedDocument],
    ) -> List[RetrievedDocument]:
        """Keep results that meet the minimum score."""

        return [
            document
            for document in documents
            if document.score >= self.minimum_score
        ]

    def retrieve(self, query: str) -> FallbackReport:
        """Try retrieval strategies until one succeeds."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        strategies: List[
            tuple[str, Callable]
        ] = [
            ("primary_vector", self.primary_retriever),
            ("keyword_fallback", self.keyword_retriever),
            ("secondary_vector", self.secondary_vector_retriever),
        ]

        attempts = []
        successful_strategy = "none"
        final_documents = []

        for strategy_name, retriever in strategies:
            try:
                raw_documents = retriever(query)

                documents = self.validate_results(
                    raw_documents
                )

                if documents:
                    status = "success"
                    successful_strategy = strategy_name
                    final_documents = documents
                else:
                    status = "insufficient_results"

                attempts.append(
                    RetrievalAttempt(
                        strategy=strategy_name,
                        status=status,
                        result_count=len(documents),
                        error="",
                    )
                )

                if documents:
                    break

            except Exception as error:
                attempts.append(
                    RetrievalAttempt(
                        strategy=strategy_name,
                        status="failed",
                        result_count=0,
                        error=str(error),
                    )
                )

        return FallbackReport(
            query=query,
            successful_strategy=successful_strategy,
            fallback_count=max(0, len(attempts) - 1),
            final_status=(
                "success" if final_documents else "failed"
            ),
            documents=[
                asdict(document)
                for document in final_documents
            ],
            attempts=[
                asdict(attempt)
                for attempt in attempts
            ],
        )


def save_report(
    report: FallbackReport,
    filepath: str,
) -> None:
    """Save retrieval execution details."""

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(asdict(report), file, indent=4)

    print(f"\nFallback report saved to: {filepath}")


def main():
    retriever = RetrievalFallback(
        minimum_score=0.6
    )

    query = "How does a neural network learn?"

    report = retriever.retrieve(query)

    print("RAG RETRIEVAL FALLBACK")
    print("=" * 50)

    print(f"Query: {report.query}")
    print(f"Successful strategy: {report.successful_strategy}")
    print(f"Fallback attempts: {report.fallback_count}")
    print(f"Final status: {report.final_status}")

    print("\nRETRIEVAL ATTEMPTS")

    for attempt in report.attempts:
        print(f"\nStrategy: {attempt['strategy']}")
        print(f"Status: {attempt['status']}")
        print(f"Results: {attempt['result_count']}")

        if attempt["error"]:
            print(f"Error: {attempt['error']}")

    print("\nFINAL RESULTS")

    for document in report.documents:
        print(
            f"{document['document_id']} "
            f"| Score: {document['score']}"
        )

    save_report(
        report,
        "retrieval_fallback_report.json"
    )


if __name__ == "__main__":
    main()
