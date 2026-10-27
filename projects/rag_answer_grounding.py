"""
RAG Answer Grounding
--------------------
A lightweight baseline for checking whether answer sentences
have lexical support in retrieved context.

Features:
- Tokenizes answer sentences and source documents.
- Measures token overlap with retrieved context.
- Assigns a grounding score to each sentence.
- Flags sentences below a configurable threshold.
- Saves a JSON evaluation report.

No external dependencies required.

Limitation:
Lexical overlap cannot reliably detect contradictions, paraphrases,
or unsupported claims that reuse source vocabulary.
"""

from dataclasses import dataclass, asdict
from typing import List
import json
import re


STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "to", "of",
    "in", "on", "at", "for", "by", "with", "from", "is",
    "are", "was", "were", "be", "been", "being", "it", "this",
    "that", "these", "those", "as", "into", "than", "then",
    "can", "could", "should", "would", "will", "may", "might",
}


@dataclass
class GroundingResult:
    sentence: str
    score: float
    supported: bool
    matched_tokens: List[str]
    missing_tokens: List[str]


@dataclass
class GroundingReport:
    total_sentences: int
    supported_sentences: int
    unsupported_sentences: List[str]
    average_grounding_score: float
    results: List[GroundingResult]


class AnswerGroundingEvaluator:
    """Estimate lexical grounding against retrieved context."""

    def __init__(self, threshold: float = 0.5):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")

        self.threshold = threshold

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Convert text into normalized, meaningful tokens."""

        tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

        return [
            token for token in tokens
            if token not in STOP_WORDS
        ]

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Split text into approximate sentences."""

        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
            if sentence.strip()
        ]

    def evaluate_sentence(
        self,
        sentence: str,
        context_tokens: set,
    ) -> GroundingResult:
        """Score the overlap between one sentence and the context."""

        sentence_tokens = set(self.tokenize(sentence))

        if not sentence_tokens:
            return GroundingResult(
                sentence=sentence,
                score=0.0,
                supported=False,
                matched_tokens=[],
                missing_tokens=[],
            )

        matched = sentence_tokens & context_tokens
        missing = sentence_tokens - context_tokens

        score = len(matched) / len(sentence_tokens)

        return GroundingResult(
            sentence=sentence,
            score=round(score, 3),
            supported=score >= self.threshold,
            matched_tokens=sorted(matched),
            missing_tokens=sorted(missing),
        )

    def evaluate(
        self,
        answer: str,
        retrieved_contexts: List[str],
    ) -> GroundingReport:
        """Evaluate all answer sentences against retrieved contexts."""

        if not retrieved_contexts:
            raise ValueError(
                "At least one retrieved context is required."
            )

        context_text = " ".join(retrieved_contexts)
        context_tokens = set(self.tokenize(context_text))
        sentences = self.split_sentences(answer)

        results = [
            self.evaluate_sentence(sentence, context_tokens)
            for sentence in sentences
        ]

        supported_count = sum(
            result.supported for result in results
        )

        unsupported = [
            result.sentence
            for result in results
            if not result.supported
        ]

        average_score = (
            sum(result.score for result in results) / len(results)
            if results
            else 0.0
        )

        return GroundingReport(
            total_sentences=len(results),
            supported_sentences=supported_count,
            unsupported_sentences=unsupported,
            average_grounding_score=round(average_score, 3),
            results=results,
        )


def save_report(report: GroundingReport, filepath: str) -> None:
    """Save grounding evaluation results to JSON."""

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(asdict(report), file, indent=4)

    print(f"Grounding report saved to: {filepath}")


def main():
    retrieved_contexts = [
        (
            "Supervised learning uses labeled training examples "
            "to learn a mapping between inputs and outputs."
        ),
        (
            "Cross-validation evaluates model performance using "
            "multiple data splits."
        ),
    ]

    answer = (
        "Supervised learning uses labeled examples to learn patterns. "
        "Cross-validation evaluates performance across data splits. "
        "The model guarantees perfect accuracy in every situation."
    )

    evaluator = AnswerGroundingEvaluator(threshold=0.5)

    report = evaluator.evaluate(
        answer=answer,
        retrieved_contexts=retrieved_contexts,
    )

    print("RAG ANSWER GROUNDING REPORT")
    print("=" * 45)
    print(f"Total sentences: {report.total_sentences}")
    print(f"Supported sentences: {report.supported_sentences}")
    print(
        "Average lexical grounding score: "
        f"{report.average_grounding_score:.1%}"
    )

    for index, result in enumerate(report.results, start=1):
        status = "PASS" if result.supported else "REVIEW"

        print(f"\nSentence {index}: [{status}]")
        print(result.sentence)
        print(f"Score: {result.score:.1%}")
        print(f"Matched tokens: {result.matched_tokens}")
        print(f"Missing tokens: {result.missing_tokens}")

    print("\nSentences requiring review:")
    for sentence in report.unsupported_sentences:
        print(f"- {sentence}")

    save_report(report, "grounding_report.json")


if __name__ == "__main__":
    main()
