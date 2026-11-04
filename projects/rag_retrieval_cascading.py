"""
RAG Cascading Retrieval
-----------------------
Implements a multi-stage retrieval pipeline.

Stages:
1. Keyword retrieval
2. Dense retrieval
3. Hybrid retrieval

The pipeline escalates when a stage fails to retrieve
enough sufficiently relevant documents.

Features:
- Multiple retrieval stages
- Configurable relevance thresholds
- Minimum result requirements
- Retrieval-stage tracking
- Execution statistics
- JSON report export

No external dependencies required.

Note:
Retrieval results are simulated. Replace the retriever
functions with actual search implementations in production.
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
class CascadeReport:
    query: str
    selected_stage: str
    escalation_count: int
    documents: List[dict]
    stage_results: List[dict]
    final_status: str


class CascadingRetriever:
    """Execute retrieval strategies progressively."""

    def __init__(
        self,
        minimum_results: int = 2,
        relevance_threshold: float = 0.6,
    ):
        if minimum_results < 1:
            raise ValueError("minimum_results must be at least 1.")

        if not 0 <= relevance_threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1.")

        self.minimum_results = minimum_results
        self.relevance_threshold = relevance_threshold

    def keyword_retrieval(
        self, query: str
    ) -> List[RetrievedDocument]:
        """Simulated keyword retrieval."""

        return [
            RetrievedDocument("D1", "Keyword matching document.", 0.72),
            RetrievedDocument("D2", "Related keyword document.", 0.54),
        ]

    def dense_retrieval(
        self, query: str
    ) -> List[RetrievedDocument]:
        """Simulated embedding-based retrieval."""

        return [
            RetrievedDocument("D3", "Semantic matching document.", 0.91),
            RetrievedDocument("D4", "Contextually related document.", 0.83),
            RetrievedDocument("D5", "Weak semantic match.", 0.42),
        ]

    def hybrid_retrieval(
        self, query: str
    ) -> List[RetrievedDocument]:
        """Simulated combined keyword and dense retrieval."""

        return [
            RetrievedDocument("D6", "Strong hybrid result.", 0.95),
            RetrievedDocument("D7", "Relevant hybrid result.", 0.89),
            RetrievedDocument("D8", "Additional useful result.", 0.78),
        ]

    def is_sufficient(
        self,
        documents: List[RetrievedDocument],
    ) -> bool:
        """Check whether enough relevant documents were retrieved."""

        relevant_count = sum(
            document.score >= self.relevance_threshold
            for document in documents
        )

        return relevant_count >= self.minimum_results

    def retrieve(self, query: str) -> CascadeReport:
        """Execute retrieval stages until sufficient results are found."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        stages: List[tuple[str, Callable]] = [
            ("keyword", self.keyword_retrieval),
            ("dense", self.dense_retrieval),
            ("hybrid", self.hybrid_retrieval),
        ]

        stage_results = []
        selected_documents = []
        selected_stage = "none"

        for stage_name, retriever in stages:
            documents = retriever(query)

            relevant_documents = [
                document
                for document in documents
                if document.score >= self.relevance_threshold
            ]

            stage_results.append({
                "stage": stage_name,
                "retrieved_count": len(documents),
                "relevant_count": len(relevant_documents),
                "sufficient": self.is_sufficient(documents),
            })

            selected_documents = relevant_documents
            selected_stage = stage_name

            if self.is_sufficient(documents):
                break

        successful = self.is_sufficient(
            [
                RetrievedDocument(**document)
                for document in [
                    asdict(item) for item in selected_documents
                ]
            ]
        )

        return CascadeReport(
            query=query,
            selected_stage=selected_stage,
            escalation_count=len(stage_results) - 1,
            documents=[
                asdict(document)
                for document in selected_documents
            ],
            stage_results=stage_results,
            final_status=(
                "sufficient" if successful else "insufficient"
            ),
        )


def save_report(report: CascadeReport, filepath: str) -> None:
    """Export cascade execution details."""

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(asdict(report), file, indent=4)

    print(f"\nCascade report saved to: {filepath}")


def main():
    retriever = CascadingRetriever(
        minimum_results=2,
        relevance_threshold=0.6,
    )

    query = "How does a neural network learn?"

    report = retriever.retrieve(query)

    print("RAG CASCADING RETRIEVAL")
    print("=" * 50)

    print(f"Query: {report.query}")
    print(f"Selected stage: {report.selected_stage}")
    print(f"Escalations: {report.escalation_count}")
    print(f"Final status: {report.final_status}")

    print("\nSTAGE EXECUTION")

    for stage in report.stage_results:
        print(
            f"{stage['stage']}: "
            f"{stage['relevant_count']} relevant results "
            f"| Sufficient: {stage['sufficient']}"
        )

    print("\nFINAL DOCUMENTS")

    for document in report.documents:
        print(
            f"{document['document_id']} "
            f"| Score: {document['score']}"
        )

    save_report(report, "retrieval_cascade_report.json")


if __name__ == "__main__":
    main()
