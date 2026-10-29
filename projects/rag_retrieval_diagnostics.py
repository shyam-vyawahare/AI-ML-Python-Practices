"""
RAG Retrieval Diagnostics
-------------------------
Analyze retrieved documents before generating an answer.

Features:
- Query-term overlap scoring
- Duplicate document detection
- Unique-term diversity measurement
- Retrieval ranking diagnostics
- JSON report export

No external dependencies required.

Note:
Scores are lexical heuristics, not semantic relevance judgments.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict
from collections import Counter
import hashlib
import json
import re


STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was",
    "were", "be", "been", "to", "of", "in", "on", "at", "for",
    "by", "with", "from", "as", "it", "this", "that", "these",
    "those", "what", "which", "who", "when", "where", "why",
    "how", "does", "do", "can", "could", "would", "should",
}


@dataclass
class DocumentDiagnostic:
    document_id: str
    rank: int
    relevance_score: float
    matched_terms: List[str]
    missing_query_terms: List[str]
    duplicate_of: str
    content_hash: str


@dataclass
class RetrievalReport:
    query: str
    total_documents: int
    unique_documents: int
    duplicate_count: int
    average_relevance: float
    unique_term_ratio: float
    low_relevance_documents: List[str]
    diagnostics: List[DocumentDiagnostic]


class RetrievalDiagnostics:
    """Diagnose basic quality issues in retrieved documents."""

    def __init__(self, low_relevance_threshold: float = 0.2):
        if not 0 <= low_relevance_threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1.")

        self.low_relevance_threshold = low_relevance_threshold

    @staticmethod
    def tokenize(text: str) -> List[str]:
        tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        return [token for token in tokens if token not in STOP_WORDS]

    @staticmethod
    def content_hash(text: str) -> str:
        """Generate a stable hash for normalized document content."""

        normalized = " ".join(text.lower().split())
        return hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

    def diagnose(
        self,
        query: str,
        documents: List[Dict[str, str]],
    ) -> RetrievalReport:
        """
        Each document must contain:
        {
            "id": "doc_1",
            "content": "Document text..."
        }
        """

        query_terms = set(self.tokenize(query))

        if not query_terms:
            raise ValueError("Query must contain meaningful terms.")

        seen_hashes = {}
        diagnostics = []
        all_document_terms = set()
        low_relevance = []
        relevance_scores = []

        for rank, document in enumerate(documents, start=1):
            document_id = document["id"]
            content = document["content"]

            if not document_id.strip():
                raise ValueError("Document IDs cannot be empty.")

            content_digest = self.content_hash(content)

            duplicate_of = seen_hashes.get(content_digest, "")

            if not duplicate_of:
                seen_hashes[content_digest] = document_id

            document_terms = set(self.tokenize(content))
            all_document_terms.update(document_terms)

            matched_terms = query_terms & document_terms
            missing_terms = query_terms - document_terms

            score = len(matched_terms) / len(query_terms)
            score = round(score, 3)

            relevance_scores.append(score)

            if score < self.low_relevance_threshold:
                low_relevance.append(document_id)

            diagnostics.append(
                DocumentDiagnostic(
                    document_id=document_id,
                    rank=rank,
                    relevance_score=score,
                    matched_terms=sorted(matched_terms),
                    missing_query_terms=sorted(missing_terms),
                    duplicate_of=duplicate_of,
                    content_hash=content_digest,
                )
            )

        if not documents:
            average_relevance = 0.0
            unique_term_ratio = 0.0
        else:
            average_relevance = round(
                sum(relevance_scores) / len(relevance_scores),
                3,
            )

            total_term_occurrences = sum(
                len(self.tokenize(document["content"]))
                for document in documents
            )

            unique_term_ratio = round(
                len(all_document_terms) / total_term_occurrences,
                3,
            ) if total_term_occurrences else 0.0

        unique_documents = len(seen_hashes)

        return RetrievalReport(
            query=query,
            total_documents=len(documents),
            unique_documents=unique_documents,
            duplicate_count=len(documents) - unique_documents,
            average_relevance=average_relevance,
            unique_term_ratio=unique_term_ratio,
            low_relevance_documents=low_relevance,
            diagnostics=diagnostics,
        )


def save_report(report: RetrievalReport, filepath: str) -> None:
    """Export diagnostics as JSON."""

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(asdict(report), file, indent=4)

    print(f"\nReport saved to: {filepath}")


def main():
    query = "How does a neural network learn using gradient descent?"

    documents = [
        {
            "id": "doc_1",
            "content": (
                "A neural network learns by adjusting its weights "
                "using gradients computed during backpropagation."
            ),
        },
        {
            "id": "doc_2",
            "content": (
                "Gradient descent updates model parameters to "
                "minimize a loss function."
            ),
        },
        {
            "id": "doc_3",
            "content": (
                "Gradient descent updates model parameters to "
                "minimize a loss function."
            ),
        },
        {
            "id": "doc_4",
            "content": (
                "A database index improves query performance "
                "by reducing unnecessary data scans."
            ),
        },
    ]

    analyzer = RetrievalDiagnostics(
        low_relevance_threshold=0.2
    )

    report = analyzer.diagnose(query, documents)

    print("RAG RETRIEVAL DIAGNOSTICS")
    print("=" * 45)
    print(f"Query: {report.query}")
    print(f"Retrieved documents: {report.total_documents}")
    print(f"Unique documents: {report.unique_documents}")
    print(f"Duplicate documents: {report.duplicate_count}")
    print(f"Average lexical relevance: {report.average_relevance:.1%}")
    print(f"Unique-term ratio: {report.unique_term_ratio:.1%}")

    print("\nDocument details:")

    for item in report.diagnostics:
        print(f"\nRank {item.rank}: {item.document_id}")
        print(f"  Relevance: {item.relevance_score:.1%}")
        print(f"  Matched terms: {item.matched_terms}")

        if item.duplicate_of:
            print(f"  Duplicate of: {item.duplicate_of}")

        if item.document_id in report.low_relevance_documents:
            print("  Warning: Low lexical relevance")

    print("\nLow-relevance documents:")
    print(report.low_relevance_documents or "None")

    save_report(report, "retrieval_diagnostics.json")


if __name__ == "__main__":
    main()
