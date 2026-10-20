
"""
RAG Confidence Scoring

Estimates evidence quality using:
1. Retrieval relevance
2. Evidence coverage
3. Agreement between retrieved documents

IMPORTANT:
The resulting score is a heuristic, NOT a calibrated
probability that an answer is correct.
"""

from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class Document:
    doc_id: str
    text: str
    source: str


class RAGConfidenceScorer:
    def __init__(
        self,
        documents,
        model_name=MODEL_NAME,
        top_k=3,
        min_similarity=0.25,
    ):
        self.documents = documents
        self.top_k = top_k
        self.min_similarity = min_similarity

        self.model = SentenceTransformer(model_name)

        if documents:
            self.embeddings = self.model.encode(
                [doc.text for doc in documents],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            self.embeddings = np.empty((0, 0))

    def retrieve(self, query):
        """Retrieve relevant documents."""

        if not self.documents:
            return []

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        scores = self.embeddings @ query_embedding
        ranked_indices = np.argsort(scores)[::-1]

        results = []

        for index in ranked_indices:
            score = float(scores[index])

            if score < self.min_similarity:
                continue

            document = self.documents[int(index)]

            results.append({
                "doc_id": document.doc_id,
                "source": document.source,
                "text": document.text,
                "similarity": score,
            })

            if len(results) >= self.top_k:
                break

        return results

    @staticmethod
    def score_relevance(results):
        """Score the strength of retrieval matches."""

        if not results:
            return 0.0

        scores = [
            max(0.0, min(1.0, result["similarity"]))
            for result in results
        ]

        # Give the strongest match more influence.
        best_score = max(scores)
        mean_score = float(np.mean(scores))

        return 0.7 * best_score + 0.3 * mean_score

    @staticmethod
    def score_coverage(results, required_topics):
        """
        Estimate coverage using required topic keywords.

        This is lexical coverage, not semantic entailment.
        """

        if not required_topics:
            return 1.0 if results else 0.0

        evidence_text = " ".join(
            result["text"].lower()
            for result in results
        )

        covered = sum(
            1
            for topic in required_topics
            if topic.lower() in evidence_text
        )

        return covered / len(required_topics)

    @staticmethod
    def score_agreement(results):
        """
        Estimate consistency from pairwise embedding similarity.

        High similarity can indicate repeated evidence, but it
        does not prove that documents agree factually.
        """

        if not results:
            return 0.0

        if len(results) == 1:
            return 0.5

        texts = [
            result["text"]
            for result in results
        ]

        model = SentenceTransformer(MODEL_NAME)

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        similarities = embeddings @ embeddings.T

        upper_triangle = similarities[
            np.triu_indices(len(results), k=1)
        ]

        # Normalize cosine similarity from [-1, 1] to [0, 1].
        normalized = (upper_triangle + 1.0) / 2.0

        return float(np.mean(normalized))

    def evaluate(
        self,
        query,
        required_topics=None,
        relevance_weight=0.50,
        coverage_weight=0.30,
        agreement_weight=0.20,
        answer_threshold=0.65,
    ):
        """Retrieve evidence and calculate a heuristic score."""

        weights = [
            relevance_weight,
            coverage_weight,
            agreement_weight,
        ]

        if any(weight < 0 for weight in weights):
            raise ValueError("Weights must be non-negative")

        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Weights must sum to 1.0")

        if not 0.0 <= answer_threshold <= 1.0:
            raise ValueError(
                "answer_threshold must be between 0 and 1"
            )

        results = self.retrieve(query)

        relevance = self.score_relevance(results)

        coverage = self.score_coverage(
            results,
            required_topics or [],
        )

        agreement = self.score_agreement(results)

        confidence = (
            relevance_weight * relevance
            + coverage_weight * coverage
            + agreement_weight * agreement
        )

        # Require evidence as well as a sufficient score.
        if not results:
            decision = "abstain"
        elif confidence >= answer_threshold:
            decision = "answer_candidate"
        else:
            decision = "review_or_retrieve_more"

        return {
            "query": query,
            "results": results,
            "relevance": relevance,
            "coverage": coverage,
            "agreement": agreement,
            "confidence_score": confidence,
            "decision": decision,
        }


def demo():
    documents = [
        Document(
            "doc-1",
            (
                "An RCCB detects residual current and disconnects "
                "a circuit when its operating conditions are met. "
                "It does not normally provide overcurrent protection."
            ),
            "rccb_manual.txt",
        ),
        Document(
            "doc-2",
            (
                "An RCBO combines residual-current protection "
                "with overcurrent protection in one device."
            ),
            "rcbo_manual.txt",
        ),
        Document(
            "doc-3",
            (
                "Protective earthing provides a fault-current path "
                "to support operation of protective devices."
            ),
            "electrical_safety.txt",
        ),
    ]

    scorer = RAGConfidenceScorer(
        documents,
        top_k=3,
        min_similarity=0.20,
    )

    query = "How do RCCB and RCBO protection functions differ?"

    result = scorer.evaluate(
        query,
        required_topics=[
            "residual current",
            "overcurrent protection",
        ],
    )

    print("=" * 65)
    print(f"Query: {result['query']}")

    print("\nEvidence quality:")
    print(f"Relevance: {result['relevance']:.3f}")
    print(f"Coverage:  {result['coverage']:.3f}")
    print(f"Agreement: {result['agreement']:.3f}")

    print(
        "\nHeuristic confidence score: "
        f"{result['confidence_score']:.3f}"
    )

    print(f"Decision: {result['decision']}")

    print("\nRetrieved sources:")

    for item in result["results"]:
        print(
            f"- {item['source']} "
            f"(similarity={item['similarity']:.3f})"
        )
        print(f"  {item['text']}")


if __name__ == "__main__":
    demo()
