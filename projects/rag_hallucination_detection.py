"""
RAG Hallucination Detection Practice

A lightweight rule-based grounding checker.

The system:
1. Extracts claims from an answer.
2. Compares each claim against retrieved context.
3. Calculates a grounding score.
4. Flags potentially unsupported claims.

This is NOT a replacement for an LLM-based evaluator.
It is a practical foundation for understanding
answer faithfulness in RAG systems.
"""

import re
from dataclasses import dataclass
from typing import List


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ClaimResult:
    claim: str
    supported: bool
    matched_terms: List[str]
    support_score: float


@dataclass
class GroundingReport:
    answer: str
    claims: List[ClaimResult]
    grounding_score: float
    status: str


# ============================================================
# TEXT PROCESSING
# ============================================================

STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "with",
    "that",
    "this",
    "it",
    "as",
    "by",
    "from",
    "can",
    "be",
    "used",
    "uses",
    "using",
    "into",
    "than",
    "its",
}


def tokenize(text: str) -> List[str]:
    """
    Convert text into normalized tokens.
    """

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


def meaningful_tokens(text: str) -> List[str]:
    """
    Remove common stopwords.
    """

    return [
        token
        for token in tokenize(text)
        if token not in STOPWORDS
    ]


# ============================================================
# CLAIM EXTRACTION
# ============================================================

def extract_claims(answer: str) -> List[str]:
    """
    Split an answer into sentence-level claims.

    In a production system, an LLM or more advanced NLP
    model could perform structured claim extraction.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        answer.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# CLAIM SUPPORT
# ============================================================

def calculate_claim_support(
    claim: str,
    context: str
) -> ClaimResult:
    """
    Determine how strongly the context supports a claim.

    The score is based on the percentage of meaningful
    claim terms appearing in the context.
    """

    claim_terms = set(
        meaningful_tokens(claim)
    )

    context_terms = set(
        meaningful_tokens(context)
    )

    if not claim_terms:
        return ClaimResult(
            claim=claim,
            supported=False,
            matched_terms=[],
            support_score=0.0
        )

    matched_terms = sorted(
        claim_terms.intersection(
            context_terms
        )
    )

    score = (
        len(matched_terms)
        /
        len(claim_terms)
    )

    # A simple threshold.
    supported = score >= 0.5

    return ClaimResult(
        claim=claim,
        supported=supported,
        matched_terms=matched_terms,
        support_score=score
    )


# ============================================================
# GROUNDING EVALUATION
# ============================================================

def evaluate_grounding(
    answer: str,
    context: str
) -> GroundingReport:
    """
    Evaluate how well an answer is grounded in context.
    """

    claims = extract_claims(answer)

    claim_results = [
        calculate_claim_support(
            claim,
            context
        )
        for claim in claims
    ]

    if not claim_results:

        return GroundingReport(
            answer=answer,
            claims=[],
            grounding_score=0.0,
            status="NO CLAIMS"
        )

    grounding_score = sum(
        result.support_score
        for result in claim_results
    ) / len(claim_results)

    supported_count = sum(
        result.supported
        for result in claim_results
    )

    support_ratio = (
        supported_count
        /
        len(claim_results)
    )

    # --------------------------------------------------------
    # Status classification
    # --------------------------------------------------------

    if support_ratio >= 0.8:
        status = "GROUNDED"

    elif support_ratio >= 0.5:
        status = "PARTIALLY GROUNDED"

    else:
        status = "POTENTIAL HALLUCINATION"

    return GroundingReport(
        answer=answer,
        claims=claim_results,
        grounding_score=grounding_score,
        status=status
    )


# ============================================================
# DISPLAY REPORT
# ============================================================

def display_report(
    report: GroundingReport
):
    """
    Display a readable grounding report.
    """

    print("\n")
    print("=" * 70)
    print("RAG GROUNDING REPORT")
    print("=" * 70)

    print(
        f"\nAnswer:\n{report.answer}"
    )

    print(
        f"\nGrounding Score: "
        f"{report.grounding_score:.2f}"
    )

    print(
        f"Status: {report.status}"
    )

    print("\nClaim Analysis")
    print("-" * 70)

    for index, claim in enumerate(
        report.claims,
        start=1
    ):

        status = (
            "SUPPORTED"
            if claim.supported
            else "UNSUPPORTED"
        )

        print(
            f"\nClaim {index}:"
        )

        print(
            f"Text: {claim.claim}"
        )

        print(
            f"Status: {status}"
        )

        print(
            f"Support Score: "
            f"{claim.support_score:.2f}"
        )

        print(
            f"Matched Terms: "
            f"{', '.join(claim.matched_terms)}"
        )


# ============================================================
# CONTEXT BUILDER
# ============================================================

def build_context(
    chunks: List[str]
) -> str:
    """
    Combine retrieved chunks into a single context.
    """

    return "\n".join(
        chunks
    )


# ============================================================
# EXPERIMENT 1 — GROUNDED ANSWER
# ============================================================

context = build_context([
    """
    Retrieval augmented generation combines document
    retrieval with a language model to generate answers.
    """
    ,
    """
    The retrieval stage searches a knowledge base for
    relevant information before the language model
    generates the final response.
    """
])


grounded_answer = (
    "Retrieval augmented generation combines document "
    "retrieval with a language model. The retrieval stage "
    "searches a knowledge base for relevant information."
)


# ============================================================
# EXPERIMENT 2 — HALLUCINATED ANSWER
# ============================================================

hallucinated_answer = (
    "Retrieval augmented generation combines document "
    "retrieval with a language model. It was invented in "
    "2019 by a company called ExampleAI and always produces "
    "100% accurate answers."
)


# ============================================================
# EXPERIMENT 3 — PARTIALLY GROUNDED ANSWER
# ============================================================

partial_answer = (
    "Retrieval augmented generation combines document "
    "retrieval with a language model. The system searches "
    "a knowledge base before generating an answer. "
    "It can completely eliminate hallucinations."
)


# ============================================================
# CLAIM-LEVEL THRESHOLD EXPERIMENT
# ============================================================

def threshold_experiment(
    claim: str,
    context: str
):
    """
    Demonstrate how different support thresholds
    affect classification.
    """

    claim_terms = set(
        meaningful_tokens(claim)
    )

    context_terms = set(
        meaningful_tokens(context)
    )

    matched = claim_terms.intersection(
        context_terms
    )

    if not claim_terms:
        return

    score = (
        len(matched)
        /
        len(claim_terms)
    )

    print("\n")
    print("=" * 70)
    print("THRESHOLD EXPERIMENT")
    print("=" * 70)

    print(
        f"\nClaim:\n{claim}"
    )

    print(
        f"\nSupport Score: {score:.2f}"
    )

    for threshold in [
        0.3,
        0.5,
        0.7,
        0.9
    ]:

        supported = score >= threshold

        print(
            f"Threshold {threshold:.1f}: "
            f"{'SUPPORTED' if supported else 'UNSUPPORTED'}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("RAG HALLUCINATION DETECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Test 1
    # --------------------------------------------------------

    print("\n\nTEST 1 — GROUNDED ANSWER")

    report = evaluate_grounding(
        answer=grounded_answer,
        context=context
    )

    display_report(report)

    # --------------------------------------------------------
    # Test 2
    # --------------------------------------------------------

    print("\n\nTEST 2 — HALLUCINATED ANSWER")

    report = evaluate_grounding(
        answer=hallucinated_answer,
        context=context
    )

    display_report(report)

    # --------------------------------------------------------
    # Test 3
    # --------------------------------------------------------

    print("\n\nTEST 3 — PARTIALLY GROUNDED ANSWER")

    report = evaluate_grounding(
        answer=partial_answer,
        context=context
    )

    display_report(report)

    # --------------------------------------------------------
    # Threshold experiment
    # --------------------------------------------------------

    threshold_experiment(
        claim=(
            "The retrieval stage searches a knowledge base "
            "before generating an answer."
        ),
        context=context
    )
