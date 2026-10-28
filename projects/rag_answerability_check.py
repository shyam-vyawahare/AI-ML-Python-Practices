"""
RAG Answerability Check
-----------------------
Estimates whether retrieved documents contain enough lexical
evidence to attempt answering a user query.

Features:
- Tokenizes questions and retrieved documents.
- Scores query-term coverage.
- Supports configurable answerability thresholds.
- Recommends answering, retrieving more context, or abstaining.
- Generates a JSON evaluation report.

No external dependencies required.

Limitation:
Lexical coverage is a baseline heuristic. It does not understand
semantic relevance, contradictions, or whether evidence entails
a correct answer.
"""

from dataclasses import dataclass, asdict
from typing import List
import json
import re


STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was",
    "were", "be", "been", "being", "do", "does", "did", "what",
    "which", "who", "whom", "when", "where", "why", "how",
    "can", "could", "would", "should", "will", "may", "might",
    "to", "of", "in", "on", "at", "for", "by", "with", "from",
    "about", "it", "this", "that", "these", "those",
}


@dataclass
class AnswerabilityResult:
    query: str
    score: float
    covered_terms: List[str]
    missing_terms: List[str]
    decision: str
    explanation: str


class RAGAnswerabilityChecker:
    """Estimate whether retrieved context covers a query's key terms."""

    def __init__(
        self,
        answer_threshold: float = 0.6,
        retrieve_threshold: float = 0.3,
    ):
        if not 0 <= retrieve_threshold <= answer_threshold <= 1:
            raise ValueError(
                "Thresholds must satisfy "
                "0 <= retrieve_threshold <= answer_threshold <= 1."
            )

        self.answer_threshold = answer_threshold
        self.retrieve_threshold = retrieve_threshold

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Return normalized tokens, excluding common stop words."""

        tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

        return [
            token for token in tokens
            if token not in STOP_WORDS
        ]

    def check(
        self,
        query: str,
        retrieved_documents: List[str],
    ) -> AnswerabilityResult:
        """Evaluate query-term coverage in retrieved documents."""

        query_terms = set(self.tokenize(query))

        if not query_terms:
            raise ValueError(
                "The query must contain meaningful terms."
            )

        if not retrieved_documents:
            return AnswerabilityResult(
                query=query,
                score=0.0,
                covered_terms=[],
                missing_terms=sorted(query_terms),
                decision="retrieve_more",
                explanation="No retrieved documents were provided.",
            )

        context = " ".join(retrieved_documents)
        context_terms = set(self.tokenize(context))

        covered = query_terms & context_terms
        missing = query_terms - context_terms

        score = len(covered) / len(query_terms)

        if score >= self.answer_threshold:
            decision = "answer_with_evidence"
            explanation = (
                "Retrieved context covers enough query terms "
                "to attempt an evidence-based answer."
            )
        elif score >= self.retrieve_threshold:
            decision = "retrieve_more"
            explanation = (
                "Some query terms are covered, but additional "
                "retrieval or query reformulation may help."
            )
        else:
            decision = "abstain"
            explanation = (
                "Retrieved context has insufficient lexical "
                "overlap. Do not answer from this evidence alone."
            )

        return AnswerabilityResult(
            query=query,
            score=round(score, 3),
            covered_terms=sorted(covered),
            missing_terms=sorted(missing),
            decision=decision,
            explanation=explanation,
        )


def save_report(
    results: List[AnswerabilityResult],
    filepath: str,
) -> None:
    """Save answerability results to JSON."""

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(
            [asdict(result) for result in results],
            file,
            indent=4,
        )

    print(f"Answerability report saved to: {filepath}")


def main():
    checker = RAGAnswerabilityChecker(
        answer_threshold=0.6,
        retrieve_threshold=0.3,
    )

    test_cases = [
        {
            "query": "How does supervised learning use labeled data?",
            "documents": [
                (
                    "Supervised learning trains models using labeled "
                    "examples containing inputs and expected outputs."
                )
            ],
        },
        {
            "query": "How does a transformer use attention?",
            "documents": [
                "A database index can speed up data retrieval."
            ],
        },
        {
            "query": "What is cross validation for model evaluation?",
            "documents": [
                (
                    "Cross-validation evaluates a model using "
                    "multiple training and validation splits."
                )
            ],
        },
    ]

    results = []

    print("RAG ANSWERABILITY CHECK")
    print("=" * 45)

    for index, case in enumerate(test_cases, start=1):
        result = checker.check(
            query=case["query"],
            retrieved_documents=case["documents"],
        )

        results.append(result)

        print(f"\nTest {index}: {result.decision.upper()}")
        print(f"Query: {result.query}")
        print(f"Coverage score: {result.score:.1%}")
        print(f"Covered terms: {result.covered_terms}")
        print(f"Missing terms: {result.missing_terms}")
        print(f"Reason: {result.explanation}")

    save_report(results, "answerability_report.json")


if __name__ == "__main__":
    main()
