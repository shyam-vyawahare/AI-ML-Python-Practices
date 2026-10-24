
"""
RAG A/B Testing Framework

Compares two retrieval strategies on the same evaluation queries.

Metrics:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Average query latency

This is an offline evaluation framework, not a live randomized
experiment with real users.
"""

from dataclasses import dataclass
from collections import defaultdict
from time import perf_counter
import re

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class Document:
    doc_id: str
    text: str
    source: str


@dataclass
class EvaluationQuery:
    query: str
    relevant_doc_ids: set[str]


class RAGABTester:
    def __init__(self, documents, model_name=MODEL_NAME):
        self.documents = documents
        self.model = SentenceTransformer(model_name)

        self.doc_ids = [
            document.doc_id for document in documents
        ]

        self.doc_id_to_index = {
            doc_id: index
            for index, doc_id in enumerate(self.doc_ids)
        }

        if documents:
            self.embeddings = self.model.encode(
                [doc.text for doc in documents],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            self.embeddings = np.empty((0, 0))

    @staticmethod
    def tokenize(text):
        return set(
            re.findall(r"\b[a-z0-9]+\b", text.lower())
        )

    def semantic_search(self, query, top_k=3):
        """Variant A: Rank documents by semantic similarity."""

        if not self.documents:
            return []

        query_vector = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        scores = self.embeddings @ query_vector

        ranked = np.argsort(scores)[::-1][:top_k]

        return [
            self.documents[int(index)].doc_id
            for index in ranked
        ]

    def hybrid_search(
        self,
        query,
        top_k=3,
        semantic_weight=0.7,
        keyword_weight=0.3,
    ):
        """
        Variant B: Combine semantic similarity and lexical overlap.

        Both components are normalized to [0, 1] before fusion.
        """

        if not self.documents:
            return []

        if (
            semantic_weight < 0
            or keyword_weight < 0
            or not np.isclose(
                semantic_weight + keyword_weight,
                1.0,
            )
        ):
            raise ValueError(
                "Weights must be non-negative and sum to 1"
            )

        query_vector = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        semantic_scores = self.embeddings @ query_vector

        # Convert cosine similarity to [0, 1].
        semantic_scores = np.clip(
            (semantic_scores + 1.0) / 2.0,
            0.0,
            1.0,
        )

        query_tokens = self.tokenize(query)

        keyword_scores = []

        for document in self.documents:
            doc_tokens = self.tokenize(document.text)

            if not query_tokens:
                overlap = 0.0
            else:
                overlap = (
                    len(query_tokens & doc_tokens)
                    / len(query_tokens)
                )

            keyword_scores.append(overlap)

        keyword_scores = np.asarray(keyword_scores)

        combined_scores = (
            semantic_weight * semantic_scores
            + keyword_weight * keyword_scores
        )

        ranked = np.argsort(combined_scores)[::-1][:top_k]

        return [
            self.documents[int(index)].doc_id
            for index in ranked
        ]

    @staticmethod
    def precision_at_k(retrieved, relevant, k):
        if k <= 0:
            raise ValueError("k must be positive")

        selected = retrieved[:k]

        if not selected:
            return 0.0

        return sum(
            doc_id in relevant
            for doc_id in selected
        ) / k

    @staticmethod
    def recall_at_k(retrieved, relevant, k):
        if not relevant:
            return 0.0

        return sum(
            doc_id in relevant
            for doc_id in retrieved[:k]
        ) / len(relevant)

    @staticmethod
    def reciprocal_rank(retrieved, relevant):
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / rank

        return 0.0

    def evaluate_variant(
        self,
        queries,
        search_function,
        k=3,
    ):
        """Evaluate one strategy on the same query set."""

        precision_scores = []
        recall_scores = []
        reciprocal_ranks = []
        latencies_ms = []

        for item in queries:
            start = perf_counter()

            retrieved = search_function(
                item.query,
                top_k=k,
            )

            elapsed_ms = (
                perf_counter() - start
            ) * 1000

            latencies_ms.append(elapsed_ms)

            precision_scores.append(
                self.precision_at_k(
                    retrieved,
                    item.relevant_doc_ids,
                    k,
                )
            )

            recall_scores.append(
                self.recall_at_k(
                    retrieved,
                    item.relevant_doc_ids,
                    k,
                )
            )

            reciprocal_ranks.append(
                self.reciprocal_rank(
                    retrieved,
                    item.relevant_doc_ids,
                )
            )

        if not queries:
            return {
                "query_count": 0,
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "mrr": 0.0,
                "average_latency_ms": 0.0,
            }

        return {
            "query_count": len(queries),
            "precision_at_k": float(
                np.mean(precision_scores)
            ),
            "recall_at_k": float(
                np.mean(recall_scores)
            ),
            "mrr": float(
                np.mean(reciprocal_ranks)
            ),
            "average_latency_ms": float(
                np.mean(latencies_ms)
            ),
        }

    def run_experiment(self, queries, k=3):
        """Run both variants on an identical evaluation set."""

        variant_a = self.evaluate_variant(
            queries,
            self.semantic_search,
            k,
        )

        variant_b = self.evaluate_variant(
            queries,
            self.hybrid_search,
            k,
        )

        return {
            "semantic": variant_a,
            "hybrid": variant_b,
            "differences": {
                metric: (
                    variant_b[metric] - variant_a[metric]
                )
                for metric in [
                    "precision_at_k",
                    "recall_at_k",
                    "mrr",
                    "average_latency_ms",
                ]
            },
        }


def demo():
    documents = [
        Document(
            "doc-1",
            "An RCCB detects residual current and disconnects "
            "the circuit when its operating conditions are met.",
            "rccb_manual.txt",
        ),
        Document(
            "doc-2",
            "An RCBO combines residual-current and overcurrent "
            "protection in a single device.",
            "rcbo_manual.txt",
        ),
        Document(
            "doc-3",
            "A DC contactor switches the high-voltage DC circuit "
            "according to the charger's control sequence.",
            "contactor_manual.txt",
        ),
        Document(
            "doc-4",
            "A power module converts incoming AC power into "
            "regulated DC power for charging.",
            "power_module_manual.txt",
        ),
        Document(
            "doc-5",
            "An insulation monitoring device monitors insulation "
            "resistance between a DC circuit and earth.",
            "imd_manual.txt",
        ),
    ]

    evaluation_queries = [
        EvaluationQuery(
            "What does an RCCB detect?",
            {"doc-1"},
        ),
        EvaluationQuery(
            "Which device combines residual current "
            "and overcurrent protection?",
            {"doc-2"},
        ),
        EvaluationQuery(
            "How is the high-voltage DC circuit switched?",
            {"doc-3"},
        ),
        EvaluationQuery(
            "Which component converts AC power to DC?",
            {"doc-4"},
        ),
        EvaluationQuery(
            "What monitors DC insulation resistance?",
            {"doc-5"},
        ),
    ]

    tester = RAGABTester(documents)

    # Use k=3 so each strategy returns up to three results.
    results = tester.run_experiment(
        evaluation_queries,
        k=3,
    )

    print("=" * 65)
    print("RAG A/B TEST RESULTS")

    for variant, metrics in [
        ("Variant A — Semantic", results["semantic"]),
        ("Variant B — Hybrid", results["hybrid"]),
    ]:
        print(f"\n{variant}")

        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")

    print("\nDifference (Hybrid - Semantic)")

    for metric, difference in results["differences"].items():
        print(f"{metric}: {difference:+.4f}")

    print(
        "\nInterpretation: Higher precision, recall, and MRR "
        "are generally better. Lower latency is better."
    )


if __name__ == "__main__":
    demo()
