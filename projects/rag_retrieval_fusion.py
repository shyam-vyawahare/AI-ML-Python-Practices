"""
RAG Retrieval Fusion
--------------------
Combines rankings from multiple retrieval systems using
Reciprocal Rank Fusion (RRF).

Features:
- Multiple retriever support
- Weighted reciprocal rank fusion
- Configurable rank constant
- Duplicate document aggregation
- Final unified ranking
- JSON result export

No external dependencies required.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List
import json


@dataclass
class FusionResult:
    document_id: str
    fusion_score: float
    appearances: int
    retrievers: List[str]


class ReciprocalRankFusion:
    """Combine ranked document lists using weighted RRF."""

    def __init__(self, rank_constant: int = 60):
        if rank_constant < 0:
            raise ValueError("Rank constant cannot be negative.")

        self.rank_constant = rank_constant

    def fuse(
        self,
        rankings: Dict[str, List[str]],
        weights: Dict[str, float] = None,
    ) -> List[FusionResult]:
        """
        rankings:
            {
                "dense": ["D1", "D2", "D3"],
                "keyword": ["D2", "D4", "D1"]
            }

        weights:
            {
                "dense": 1.0,
                "keyword": 1.5
            }
        """

        if not rankings:
            return []

        weights = weights or {}

        unknown_weight_keys = set(weights) - set(rankings)

        if unknown_weight_keys:
            raise ValueError(
                f"Weights supplied for unknown retrievers: "
                f"{sorted(unknown_weight_keys)}"
            )

        for name, weight in weights.items():
            if weight < 0:
                raise ValueError(
                    f"Weight for {name} cannot be negative."
                )

        scores = {}
        appearances = {}
        contributing_retrievers = {}

        for retriever_name, documents in rankings.items():
            weight = weights.get(retriever_name, 1.0)

            if len(documents) != len(set(documents)):
                raise ValueError(
                    f"Duplicate document IDs in {retriever_name} ranking."
                )

            for rank, document_id in enumerate(documents, start=1):
                contribution = weight / (
                    self.rank_constant + rank
                )

                scores[document_id] = (
                    scores.get(document_id, 0.0) + contribution
                )

                appearances[document_id] = (
                    appearances.get(document_id, 0) + 1
                )

                contributing_retrievers.setdefault(
                    document_id, []
                ).append(retriever_name)

        results = [
            FusionResult(
                document_id=document_id,
                fusion_score=round(score, 6),
                appearances=appearances[document_id],
                retrievers=contributing_retrievers[document_id],
            )
            for document_id, score in scores.items()
        ]

        return sorted(
            results,
            key=lambda result: (
                -result.fusion_score,
                result.document_id,
            ),
        )


def save_results(
    results: List[FusionResult],
    filepath: str,
) -> None:
    """Save fused rankings to JSON."""

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(
            [asdict(result) for result in results],
            file,
            indent=4,
        )

    print(f"Fusion results saved to: {filepath}")


def main():
    retriever_rankings = {
        "dense_retriever": [
            "D1", "D3", "D2", "D5", "D4"
        ],
        "keyword_retriever": [
            "D2", "D4", "D1", "D6", "D3"
        ],
        "metadata_retriever": [
            "D4", "D2", "D7", "D1", "D3"
        ],
    }

    weights = {
        "dense_retriever": 1.0,
        "keyword_retriever": 1.2,
        "metadata_retriever": 0.8,
    }

    fusion = ReciprocalRankFusion(rank_constant=60)

    results = fusion.fuse(
        rankings=retriever_rankings,
        weights=weights,
    )

    print("RECIPROCAL RANK FUSION")
    print("=" * 65)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}: {result.document_id}")
        print(f"Fusion score: {result.fusion_score:.6f}")
        print(f"Appearances: {result.appearances}")
        print(f"Contributors: {result.retrievers}")

    save_results(results, "retrieval_fusion.json")


if __name__ == "__main__":
    main()
