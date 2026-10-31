"""
RAG Retrieval Ablation Study
----------------------------
Compare retrieval configurations by enabling or disabling
individual retrieval components.

Components:
- Dense retrieval
- Keyword retrieval
- Reciprocal rank fusion
- Reranking

Uses simulated retrieval rankings and ground-truth relevance labels.

No external dependencies required.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Set
import json


@dataclass
class AblationResult:
    configuration: str
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    average_precision: float


class RetrievalAblation:
    """Evaluate different retrieval configurations."""

    @staticmethod
    def precision_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int,
    ) -> float:
        if k <= 0:
            raise ValueError("K must be positive.")

        top_k = retrieved[:k]

        if not top_k:
            return 0.0

        return sum(doc in relevant for doc in top_k) / k

    @staticmethod
    def recall_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int,
    ) -> float:
        if k <= 0:
            raise ValueError("K must be positive.")

        if not relevant:
            return 0.0

        return len(set(retrieved[:k]) & relevant) / len(relevant)

    @staticmethod
    def reciprocal_rank(
        retrieved: List[str],
        relevant: Set[str],
    ) -> float:
        for rank, document in enumerate(retrieved, start=1):
            if document in relevant:
                return 1 / rank

        return 0.0

    @staticmethod
    def average_precision(
        retrieved: List[str],
        relevant: Set[str],
    ) -> float:
        if not relevant:
            return 0.0

        hits = 0
        score = 0.0
        seen = set()

        for rank, document in enumerate(retrieved, start=1):
            if document in seen:
                continue

            seen.add(document)

            if document in relevant:
                hits += 1
                score += hits / rank

        return score / len(relevant)

    def evaluate(
        self,
        configuration: str,
        retrieved: List[str],
        relevant: Set[str],
        k: int = 3,
    ) -> AblationResult:

        return AblationResult(
            configuration=configuration,
            precision_at_k=self.precision_at_k(
                retrieved, relevant, k
            ),
            recall_at_k=self.recall_at_k(
                retrieved, relevant, k
            ),
            reciprocal_rank=self.reciprocal_rank(
                retrieved, relevant
            ),
            average_precision=self.average_precision(
                retrieved, relevant
            ),
        )


def run_ablation(
    configurations: Dict[str, List[str]],
    relevant_documents: Set[str],
    k: int = 3,
) -> List[AblationResult]:

    evaluator = RetrievalAblation()
    results = []

    for name, ranking in configurations.items():
        result = evaluator.evaluate(
            configuration=name,
            retrieved=ranking,
            relevant=relevant_documents,
            k=k,
        )

        results.append(result)

    return results


def save_results(
    results: List[AblationResult],
    filepath: str,
) -> None:

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(
            [asdict(result) for result in results],
            file,
            indent=4,
        )

    print(f"\nAblation results saved to: {filepath}")


def main():
    # Simulated rankings for one evaluation query.
    relevant_documents = {"D1", "D2", "D3"}

    configurations = {
        "Dense Only": [
            "D4", "D1", "D5", "D2", "D3"
        ],
        "Dense + Keyword": [
            "D1", "D4", "D2", "D3", "D5"
        ],
        "Dense + Keyword + RRF": [
            "D2", "D1", "D3", "D4", "D5"
        ],
        "Full Pipeline + Reranking": [
            "D1", "D2", "D3", "D5", "D4"
        ],
    }

    results = run_ablation(
        configurations=configurations,
        relevant_documents=relevant_documents,
        k=3,
    )

    print("RAG RETRIEVAL ABLATION STUDY")
    print("=" * 65)

    print(
        f"{'Configuration':<35}"
        f"{'P@3':>8}"
        f"{'R@3':>8}"
        f"{'MRR':>8}"
        f"{'MAP':>8}"
    )

    print("-" * 65)

    for result in results:
        print(
            f"{result.configuration:<35}"
            f"{result.precision_at_k:>8.2f}"
            f"{result.recall_at_k:>8.2f}"
            f"{result.reciprocal_rank:>8.2f}"
            f"{result.average_precision:>8.2f}"
        )

    best_result = max(
        results,
        key=lambda item: item.average_precision,
    )

    print("\nBEST CONFIGURATION")
    print(best_result.configuration)
    print(f"MAP: {best_result.average_precision:.3f}")

    baseline = results[0]
    improvement = (
        best_result.average_precision
        - baseline.average_precision
    )

    print("\nIMPROVEMENT OVER BASELINE")
    print(f"MAP change: {improvement:+.3f}")

    save_results(results, "retrieval_ablation.json")


if __name__ == "__main__":
    main()
