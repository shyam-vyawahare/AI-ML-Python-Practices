
"""
Self-Reflective RAG Pipeline

Demonstrates:
- Document retrieval
- Draft answer generation
- Evidence-based answer verification
- Answer revision
- Abstention when evidence is insufficient

This educational version uses a simple lexical verifier.
It does not replace an LLM-based factuality judge.
"""

from dataclasses import dataclass
import re

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class Document:
    doc_id: str
    text: str
    source: str


class SelfReflectiveRAG:
    def __init__(
        self,
        documents,
        top_k=3,
        similarity_threshold=0.30,
        support_threshold=0.50,
    ):
        self.documents = documents
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.support_threshold = support_threshold

        self.model = SentenceTransformer(MODEL_NAME)

        if documents:
            self.embeddings = self.model.encode(
                [doc.text for doc in documents],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            self.embeddings = np.empty((0, 0))

    # --------------------------------------------------------
    # Step 1: Retrieve evidence
    # --------------------------------------------------------

    def retrieve(self, query):
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

            if score < self.similarity_threshold:
                continue

            doc = self.documents[int(index)]

            results.append({
                "doc_id": doc.doc_id,
                "source": doc.source,
                "text": doc.text,
                "score": score,
            })

            if len(results) >= self.top_k:
                break

        return results

    # --------------------------------------------------------
    # Step 2: Generate a draft from evidence
    # --------------------------------------------------------

    @staticmethod
    def generate_draft(query, evidence):
        """
        A simple extractive demonstration.

        A production system can replace this method with an
        LLM call that answers using only retrieved evidence.
        """

        if not evidence:
            return (
                "I cannot answer confidently from the "
                "available documents."
            )

        query_terms = set(
            re.findall(r"\b[a-z0-9]+\b", query.lower())
        )

        stopwords = {
            "what", "which", "when", "where", "who",
            "why", "how", "does", "do", "is", "are",
            "the", "a", "an", "and", "or", "of",
            "to", "in", "for", "with", "explain",
            "tell", "me", "about",
        }

        query_terms -= stopwords

        candidates = []

        for item in evidence:
            sentences = re.split(
                r"(?<=[.!?])\s+",
                item["text"].strip(),
            )

            for sentence in sentences:
                if not sentence:
                    continue

                sentence_terms = set(
                    re.findall(
                        r"\b[a-z0-9]+\b",
                        sentence.lower(),
                    )
                )

                overlap = len(query_terms & sentence_terms)

                candidates.append(
                    (overlap, item["score"], sentence)
                )

        candidates.sort(
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )

        selected = [
            sentence
            for overlap, _, sentence in candidates[:3]
            if overlap > 0
        ]

        if not selected:
            return evidence[0]["text"]

        return " ".join(selected)

    # --------------------------------------------------------
    # Step 3: Reflect on answer support
    # --------------------------------------------------------

    @staticmethod
    def verify_answer(answer, evidence):
        """
        Estimate lexical support for the draft.

        This checks word overlap, not logical entailment.
        """

        if not answer or not evidence:
            return {
                "support_score": 0.0,
                "supported": False,
                "reason": "Missing answer or evidence.",
            }

        stopwords = {
            "the", "a", "an", "is", "are", "was", "were",
            "and", "or", "of", "to", "in", "for", "with",
            "it", "this", "that", "as", "by", "from",
        }

        answer_terms = set(
            re.findall(r"\b[a-z0-9]+\b", answer.lower())
        ) - stopwords

        evidence_text = " ".join(
            item["text"] for item in evidence
        )

        evidence_terms = set(
            re.findall(
                r"\b[a-z0-9]+\b",
                evidence_text.lower(),
            )
        ) - stopwords

        if not answer_terms:
            score = 0.0
        else:
            score = (
                len(answer_terms & evidence_terms)
                / len(answer_terms)
            )

        return {
            "support_score": score,
            "supported": score >= 0.50,
            "reason": (
                "Sufficient lexical overlap."
                if score >= 0.50
                else "Weak lexical overlap with evidence."
            ),
        }

    # --------------------------------------------------------
    # Step 4: Reflect, revise or abstain
    # --------------------------------------------------------

    def answer(self, query):
        evidence = self.retrieve(query)

        if not evidence:
            return {
                "query": query,
                "answer": (
                    "I don't have sufficient evidence "
                    "to answer this question."
                ),
                "decision": "abstain",
                "evidence": [],
                "verification": None,
            }

        draft = self.generate_draft(query, evidence)
        verification = self.verify_answer(draft, evidence)

        if verification["supported"]:
            final_answer = draft
            decision = "accept"

        else:
            # Revision uses only the strongest retrieved source.
            revised_evidence = evidence[:1]
            revised_draft = self.generate_draft(
                query,
                revised_evidence,
            )

            revised_verification = self.verify_answer(
                revised_draft,
                revised_evidence,
            )

            if revised_verification["supported"]:
                final_answer = revised_draft
                verification = revised_verification
                decision = "revise"

            else:
                final_answer = (
                    "I cannot verify a reliable answer "
                    "from the available evidence."
                )
                verification = revised_verification
                decision = "abstain"

        return {
            "query": query,
            "answer": final_answer,
            "decision": decision,
            "evidence": evidence,
            "verification": verification,
        }


def demo():
    documents = [
        Document(
            "doc-1",
            (
                "Supervised learning uses labelled training "
                "data. Classification predicts categories, "
                "while regression predicts continuous values."
            ),
            "machine_learning.txt",
        ),
        Document(
            "doc-2",
            (
                "Decision trees split data using feature "
                "conditions. Random forests combine predictions "
                "from multiple decision trees."
            ),
            "ensemble_learning.txt",
        ),
        Document(
            "doc-3",
            (
                "Precision measures the fraction of predicted "
                "positive examples that are actually positive. "
                "Recall measures the fraction of actual positive "
                "examples correctly identified."
            ),
            "evaluation_metrics.txt",
        ),
    ]

    rag = SelfReflectiveRAG(
        documents,
        top_k=2,
        similarity_threshold=0.25,
        support_threshold=0.50,
    )

    queries = [
        "What is supervised learning?",
        "What is the difference between precision and recall?",
        "Explain quantum computing.",
    ]

    for query in queries:
        result = rag.answer(query)

        print("\n" + "=" * 65)
        print(f"Query: {result['query']}")
        print(f"Decision: {result['decision'].upper()}")
        print(f"Answer: {result['answer']}")

        if result["verification"]:
            print(
                "Support score: "
                f"{result['verification']['support_score']:.2f}"
            )
            print(
                "Verification: "
                f"{result['verification']['reason']}"
            )

        print("Sources:")

        for item in result["evidence"]:
            print(f"- {item['source']} ({item['score']:.3f})")


if __name__ == "__main__":
    demo()
