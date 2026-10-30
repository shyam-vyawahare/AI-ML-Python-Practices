"""
RAG Retrieval Benchmark
-----------------------
Evaluate and compare retrieval strategies using standard
information retrieval metrics.

Metrics:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Average Precision (AP)
- Mean Average Precision (MAP)

No external dependencies required.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Set
import json


@dataclass
class QueryResult:
    query: str
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    average_precision: float


class RetrievalBenchmark:
    """Calculate retrieval evaluation metrics."""

    @staticmethod
    def precision_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int,
    ) -> float:
        if k <= 0:
            raise ValueError("K must be greater than zero.")

        top_k = retrieved[:k]

        if not top_k:
            return 0.0

        relevant_count = sum(
            document in relevant for document in top_k
        )

        return relevant_count / k

    @staticmethod
    def recall_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int,
    ) -> float:
        if k <= 0:
            raise ValueError("K must be greater than zero.")

        if not relevant:
            return 0.0

        top_k = retrieved[:k]

        relevant_count = sum(
            document in relevant for document in set(top_k)
        )

        return relevant_count / len(relevant)

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
        precision_sum = 0.0
        seen = set()

        for rank, document in enumerate(retrieved, start=1):
            # Ignore repeated document IDs.
            if document in seen:
                continue

            seen.add(document)

            if document in relevant:
                hits += 1
                precision_sum += hits / rank

        return precision_sum / len(relevant)

    def evaluate_query(
        self,
        query: str,
        retrieved: List[str],
        relevant: Set[str],
        k: int = 3,
    ) -> QueryResult:

        return QueryResult(
            query=query,
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

    def evaluate_strategy(
        self,
        strategy_name: str,
        test_queries: List[Dict],
        k: int = 3,
    ) -> Dict:

        results = []

        for item in test_queries:
            result = self.evaluate_query(
                query=item["query"],
                retrieved=item["retrieved"],
                relevant=set(item["relevant"]),
                k=k,
            )

            results.append(result)

        count = len(results)

        if count == 0:
            return {
                "strategy": strategy_name,
                "query_count": 0,
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "mrr": 0.0,
                "map": 0.0,
                "results": [],
            }

        return {
            "strategy": strategy_name,
            "query_count": count,
            "precision_at_k": round(
                sum(r.precision_at_k for r in results) / count, 3
            ),
            "recall_at_k": round(
                sum(r.recall_at_k for r in results) / count, 3
            ),
            "mrr": round(
                sum(r.reciprocal_rank for r in results) / count, 3
            ),
            "map": round(
                sum(r.average_precision for r in results) / count, 3
            ),
            "results": [asdict(r) for r in results],
        }


def save_benchmark(results: List[Dict], filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print(f"\nBenchmark saved to: {filepath}")


def main():
    benchmark = RetrievalBenchmark()

    # Each query has a manually prepared relevance ground truth.
    strategies = {
        "Vector Search": [
            {
                "query": "How does gradient descent work?",
                "retrieved": ["D2", "D1", "D4", "D3"],
                "relevant": ["D1", "D2", "D3"],
            },
            {
                "query": "What is a convolutional neural network?",
                "retrieved": ["D5", "D7", "D6", "D8"],
                "relevant": ["D5", "D6"],
            },
        ],
        "Hybrid Search": [
            {
                "query": "How does gradient descent work?",
                "retrieved": ["D1", "D3", "D2", "D4"],
                "relevant": ["D1", "D2", "D3"],
            },
            {
                "query": "What is a convolutional neural network?",
                "retrieved": ["D6", "D5", "D8", "D7"],
                "relevant": ["D5", "D6"],
            },
        ],
    }

    all_results = []

    print("RAG RETRIEVAL BENCHMARK")
    print("=" * 55)

    for strategy_name, test_queries in strategies.items():
        result = benchmark.evaluate_strategy(
            strategy_name=strategy_name,
            test_queries=test_queries,
            k=3,
        )

        all_results.append(result)

        print(f"\nStrategy: {strategy_name}")
        print(f"Precision@3: {result['precision_at_k']:.1%}")
        print(f"Recall@3: {result['recall_at_k']:.1%}")
        print(f"MRR: {result['mrr']:.3f}")
        print(f"MAP: {result['map']:.3f}")

    print("\nSTRATEGY COMPARISON")
    print("-" * 55)

    best_strategy = max(
        all_results,
        key=lambda item: item["map"]
    )

    print(f"Best MAP score: {best_strategy['strategy']}")
    print(f"MAP: {best_strategy['map']:.3f}")

    save_benchmark(all_results, "retrieval_benchmark.json")


if __name__ == "__main__":
    main()
