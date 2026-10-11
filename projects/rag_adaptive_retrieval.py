
"""
Adaptive Retrieval for RAG

Features:
- Query complexity estimation
- Adaptive top-K retrieval
- Similarity threshold filtering
- Insufficient-evidence detection
- Retrieval diagnostics
"""

import re
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class Document:
    doc_id: str
    text: str
    source: str


class AdaptiveRetriever:
    def __init__(
        self,
        documents,
        model_name=MODEL_NAME,
        min_k=1,
        max_k=5,
        min_similarity=0.30,
    ):
        if min_k < 1 or max_k < min_k:
            raise ValueError("Require 1 <= min_k <= max_k")

        self.documents = documents
        self.min_k = min_k
        self.max_k = max_k
        self.min_similarity = min_similarity

        self.model = SentenceTransformer(model_name)

        self.embeddings = self.model.encode(
            [doc.text for doc in documents],
            normalize_embeddings=True,
            show_progress_bar=False,
        ) if documents else np.empty((0, 0))

    @staticmethod
    def estimate_complexity(query):
        """
        Estimate complexity using simple heuristics.

        This is a baseline, not a true semantic complexity model.
        """

        words = query.split()
        lower_query = query.lower()

        multi_part_patterns = [
            r"\bcompare\b",
            r"\bdifference between\b",
            r"\band\b",
            r"\bwhy\b.*\bhow\b",
            r"\bsteps\b",
            r"\badvantages\b",
            r"\bdisadvantages\b",
        ]

        multi_part_count = sum(
            bool(re.search(pattern, lower_query))
            for pattern in multi_part_patterns
        )

        score = 1

        if len(words) > 10:
            score += 1

        if len(words) > 20:
            score += 1

        if multi_part_count >= 1:
            score += 1

        if multi_part_count >= 2:
            score += 1

        return min(score, 5)

    def choose_top_k(self, query):
        """Choose a retrieval depth from query complexity."""

        complexity = self.estimate_complexity(query)

        # Map complexity from 1–5 to min_k–max_k.
        top_k = round(
            self.min_k
            + (complexity - 1)
            * (self.max_k - self.min_k)
            / 4
        )

        return complexity, top_k

    def search(self, query):
        """Perform adaptive retrieval and return diagnostics."""

        if not self.documents:
            return {
                "query": query,
                "complexity": 0,
                "requested_k": 0,
                "results": [],
                "evidence_found": False,
                "message": "No documents are indexed.",
            }

        complexity, requested_k = self.choose_top_k(query)

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        scores = self.embeddings @ query_embedding

        ranked_indices = np.argsort(scores)[::-1]

        # Inspect enough candidates to fill the requested result count
        # after filtering out weak matches.
        results = []

        for index in ranked_indices:
            score = float(scores[index])

            if score < self.min_similarity:
                continue

            doc = self.documents[int(index)]

            results.append({
                "doc_id": doc.doc_id,
                "source": doc.source,
                "text": doc.text,
                "similarity": score,
            })

            if len(results) >= requested_k:
                break

        evidence_found = bool(results)

        return {
            "query": query,
            "complexity": complexity,
            "requested_k": requested_k,
            "results": results,
            "evidence_found": evidence_found,
            "message": (
                "Relevant evidence retrieved."
                if evidence_found
                else "Insufficient matching evidence. "
                     "Consider query rewriting or another source."
            ),
        }


def demo():
    documents = [
        Document(
            "doc-1",
            "Supervised learning trains models using labelled examples.",
            "ml_fundamentals.txt",
        ),
        Document(
            "doc-2",
            "Linear regression predicts continuous numerical values.",
            "regression.txt",
        ),
        Document(
            "doc-3",
            "Logistic regression estimates probabilities for classification.",
            "classification.txt",
        ),
        Document(
            "doc-4",
            "Decision trees make predictions using feature-based splits.",
            "decision_trees.txt",
        ),
        Document(
            "doc-5",
            "Random forests combine predictions from multiple decision trees.",
            "ensemble_learning.txt",
        ),
        Document(
            "doc-6",
            "Cross-validation estimates model performance across data splits.",
            "model_evaluation.txt",
        ),
    ]

    retriever = AdaptiveRetriever(
        documents,
        min_k=1,
        max_k=5,
        min_similarity=0.25,
    )

    queries = [
        "What is supervised learning?",
        "How does logistic regression work and when should it be used?",
        (
            "Compare linear regression and logistic regression, "
            "explain their differences, use cases, and how to evaluate "
            "their performance."
        ),
        "Explain quantum teleportation in detail.",
    ]

    for query in queries:
        output = retriever.search(query)

        print("\n" + "=" * 70)
        print(f"Query: {output['query']}")
        print(f"Complexity: {output['complexity']}/5")
        print(f"Requested top-K: {output['requested_k']}")
        print(f"Status: {output['message']}")

        for result in output["results"]:
            print(
                f"\n[{result['doc_id']}] "
                f"Similarity: {result['similarity']:.3f}"
            )
            print(f"Source: {result['source']}")
            print(f"Text: {result['text']}")


if __name__ == "__main__":
    demo()
