"""
RAG Retrieval Diversity
-----------------------
Implements Maximal Marginal Relevance (MMR) to select
relevant and diverse documents.

Features:
- Cosine similarity calculation
- Query-document relevance scoring
- Document-document redundancy scoring
- MMR-based document selection
- Configurable diversity parameter

No external dependencies required.

Note:
This implementation uses manually supplied feature vectors
for demonstration. Production systems should use embeddings.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List
import json
import math


@dataclass
class SelectionResult:
    document_id: str
    relevance_score: float
    redundancy_score: float
    mmr_score: float
    selection_rank: int


class MMRRetriever:
    """Select documents using Maximal Marginal Relevance."""

    def __init__(self, diversity: float = 0.5):
        if not 0.0 <= diversity <= 1.0:
            raise ValueError("Diversity must be between 0 and 1.")

        self.diversity = diversity

    @staticmethod
    def cosine_similarity(
        vector_a: List[float],
        vector_b: List[float],
    ) -> float:
        """Calculate cosine similarity between two vectors."""

        if len(vector_a) != len(vector_b):
            raise ValueError("Vector dimensions must match.")

        dot_product = sum(
            a * b for a, b in zip(vector_a, vector_b)
        )

        norm_a = math.sqrt(sum(a * a for a in vector_a))
        norm_b = math.sqrt(sum(b * b for b in vector_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def select(
        self,
        query_vector: List[float],
        document_vectors: Dict[str, List[float]],
        top_k: int = 3,
    ) -> List[SelectionResult]:
        """Select relevant and diverse documents using MMR."""

        if top_k < 0:
            raise ValueError("top_k cannot be negative.")

        if top_k == 0 or not document_vectors:
            return []

        if len(document_vectors) != len(set(document_vectors)):
            raise ValueError("Document IDs must be unique.")

        if len(query_vector) == 0:
            raise ValueError("Query vector cannot be empty.")

        for document_id, vector in document_vectors.items():
            if len(vector) != len(query_vector):
                raise ValueError(
                    f"Invalid vector dimensions for {document_id}."
                )

        relevance = {
            document_id: self.cosine_similarity(
                query_vector, vector
            )
            for document_id, vector in document_vectors.items()
        }

        selected = []
        remaining = set(document_vectors.keys())
        results = []

        while remaining and len(selected) < top_k:
            best_document = None
            best_score = float("-inf")
            best_redundancy = 0.0

            for document_id in remaining:
                if selected:
                    redundancy = max(
                        self.cosine_similarity(
                            document_vectors[document_id],
                            document_vectors[selected_id],
                        )
                        for selected_id in selected
                    )
                else:
                    redundancy = 0.0

                mmr_score = (
                    (1 - self.diversity)
                    * relevance[document_id]
                    - self.diversity * redundancy
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_document = document_id
                    best_redundancy = redundancy

            selected.append(best_document)
            remaining.remove(best_document)

            results.append(
                SelectionResult(
                    document_id=best_document,
                    relevance_score=round(
                        relevance[best_document], 4
                    ),
                    redundancy_score=round(
                        best_redundancy, 4
                    ),
                    mmr_score=round(best_score, 4),
                    selection_rank=len(selected),
                )
            )

        return results


def save_results(
    results: List[SelectionResult],
    filepath: str,
) -> None:
    """Export selected documents and their scores."""

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(
            [asdict(result) for result in results],
            file,
            indent=4,
        )

    print(f"MMR results saved to: {filepath}")


def main():
    # Simplified illustrative vectors.
    query_vector = [1.0, 0.5, 0.2]

    documents = {
        "D1": [0.95, 0.45, 0.20],
        "D2": [0.93, 0.48, 0.18],
        "D3": [0.20, 0.90, 0.40],
        "D4": [0.65, 0.30, 0.85],
        "D5": [0.10, 0.20, 0.95],
    }

    retriever = MMRRetriever(diversity=0.5)

    results = retriever.select(
        query_vector=query_vector,
        document_vectors=documents,
        top_k=3,
    )

    print("MMR DIVERSITY-AWARE RETRIEVAL")
    print("=" * 50)

    for result in results:
        print(f"\nRank: {result.selection_rank}")
        print(f"Document: {result.document_id}")
        print(f"Relevance: {result.relevance_score}")
        print(f"Redundancy: {result.redundancy_score}")
        print(f"MMR score: {result.mmr_score}")

    save_results(results, "retrieval_diversity.json")


if __name__ == "__main__":
    main()
