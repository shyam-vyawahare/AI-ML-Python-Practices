"""
RAG Citation Validation
-----------------------
Validates citations in generated answers against retrieved documents.

Features:
- Extracts citation IDs from generated answers.
- Detects citations referencing unknown sources.
- Identifies uncited sentences.
- Reports unused retrieved sources.
- Calculates basic citation coverage.

No external dependencies required.

Note:
This is a baseline validator. Citation presence does not prove that
a cited source actually supports the associated claim.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List
import json
import re


@dataclass
class RetrievedSource:
    """A document retrieved by the RAG system."""

    source_id: str
    title: str
    content: str


@dataclass
class CitationReport:
    """Results of validating an answer's citations."""

    total_sentences: int
    cited_sentences: int
    uncited_sentences: List[str]
    used_citations: List[str]
    unknown_citations: List[str]
    unused_sources: List[str]
    citation_coverage: float
    is_valid: bool


class CitationValidator:
    """Validate answer citations against retrieved source IDs."""

    # Matches citations such as [S1], [S2], and [DOC_10].
    CITATION_PATTERN = re.compile(
        r"\[([A-Za-z][A-Za-z0-9_-]*)\]"
    )

    def __init__(self, sources: List[RetrievedSource]):
        self.sources: Dict[str, RetrievedSource] = {
            source.source_id: source
            for source in sources
        }

        if len(self.sources) != len(sources):
            raise ValueError("Source IDs must be unique.")

    @staticmethod
    def split_sentences(answer: str) -> List[str]:
        """Split text into approximate sentences."""

        answer = answer.strip()

        if not answer:
            return []

        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", answer)
            if sentence.strip()
        ]

    def extract_citations(self, text: str) -> List[str]:
        """Return unique citation IDs found in the text."""

        return list(dict.fromkeys(
            self.CITATION_PATTERN.findall(text)
        ))

    def validate(self, answer: str) -> CitationReport:
        """Validate citation references in a generated answer."""

        sentences = self.split_sentences(answer)

        all_citations = self.extract_citations(answer)

        known_citations = [
            citation
            for citation in all_citations
            if citation in self.sources
        ]

        unknown_citations = [
            citation
            for citation in all_citations
            if citation not in self.sources
        ]

        cited_sentences = []
        uncited_sentences = []

        for sentence in sentences:
            citations = self.extract_citations(sentence)

            if citations:
                cited_sentences.append(sentence)
            else:
                uncited_sentences.append(sentence)

        total_sentences = len(sentences)
        cited_count = len(cited_sentences)

        coverage = (
            cited_count / total_sentences
            if total_sentences
            else 0.0
        )

        used_sources = set(known_citations)

        unused_sources = [
            source_id
            for source_id in self.sources
            if source_id not in used_sources
        ]

        # Structural validation only; it does not verify factual support.
        is_valid = (
            not unknown_citations
            and not uncited_sentences
        )

        return CitationReport(
            total_sentences=total_sentences,
            cited_sentences=cited_count,
            uncited_sentences=uncited_sentences,
            used_citations=known_citations,
            unknown_citations=unknown_citations,
            unused_sources=unused_sources,
            citation_coverage=round(coverage, 3),
            is_valid=is_valid,
        )

    def get_source(self, source_id: str) -> RetrievedSource:
        """Retrieve a source by its ID."""

        if source_id not in self.sources:
            raise KeyError(f"Unknown source ID: {source_id}")

        return self.sources[source_id]


def save_report(report: CitationReport, filepath: str) -> None:
    """Save the validation report as JSON."""

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(asdict(report), file, indent=4)

    print(f"Report saved to {filepath}")


def main():
    sources = [
        RetrievedSource(
            source_id="S1",
            title="Machine Learning Basics",
            content=(
                "Supervised learning uses labeled examples "
                "to train predictive models."
            ),
        ),
        RetrievedSource(
            source_id="S2",
            title="Model Evaluation",
            content=(
                "Precision measures the proportion of predicted "
                "positive cases that are actually positive."
            ),
        ),
        RetrievedSource(
            source_id="S3",
            title="Cross Validation",
            content=(
                "Cross-validation estimates model performance "
                "across different data splits."
            ),
        ),
    ]

    answer = """
    Supervised learning uses labeled examples to train models [S1].
    Precision evaluates the correctness of positive predictions [S2].
    Cross-validation helps estimate model performance [S9].
    Machine learning can solve many practical problems.
    """

    validator = CitationValidator(sources)
    report = validator.validate(answer)

    print("RAG CITATION VALIDATION")
    print("=" * 40)
    print(f"Total sentences: {report.total_sentences}")
    print(f"Cited sentences: {report.cited_sentences}")
    print(f"Citation coverage: {report.citation_coverage:.1%}")
    print(f"Structurally valid: {report.is_valid}")

    print("\nUsed citations:")
    print(report.used_citations or "None")

    print("\nUnknown citations:")
    print(report.unknown_citations or "None")

    print("\nUncited sentences:")
    for sentence in report.uncited_sentences:
        print(f"- {sentence}")

    print("\nUnused sources:")
    print(report.unused_sources or "None")

    save_report(report, "citation_report.json")


if __name__ == "__main__":
    main()
