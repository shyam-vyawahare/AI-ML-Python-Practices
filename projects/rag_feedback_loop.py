
"""
RAG Feedback Loop

Features:
- Collect answer feedback
- Persist feedback to JSONL
- Track positive and negative ratings
- Identify queries needing investigation
- Compare retrieval outcomes by source

This example records feedback for analysis. It does not
automatically retrain a model or alter retrieval behavior.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter, defaultdict
import json
import uuid


FEEDBACK_FILE = Path("rag_feedback.jsonl")


@dataclass
class FeedbackRecord:
    feedback_id: str
    query: str
    answer: str
    rating: int
    comment: str
    retrieved_sources: list[str]
    retrieval_scores: list[float]
    created_at: str


class FeedbackStore:
    def __init__(self, file_path=FEEDBACK_FILE):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.records = []
        self.load()

    def load(self):
        """Load existing feedback records."""

        self.records = []

        if not self.file_path.exists():
            return

        with self.file_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    self.records.append(
                        FeedbackRecord(**data)
                    )
                except (json.JSONDecodeError, TypeError) as error:
                    print(
                        f"Skipping invalid feedback line "
                        f"{line_number}: {error}"
                    )

    def add_feedback(
        self,
        query,
        answer,
        rating,
        comment="",
        retrieved_sources=None,
        retrieval_scores=None,
    ):
        """
        Record a rating from 1 to 5.

        1 = very poor
        5 = excellent
        """

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if not answer.strip():
            raise ValueError("Answer cannot be empty")

        if not isinstance(rating, int) or isinstance(rating, bool):
            raise ValueError("Rating must be an integer from 1 to 5")

        if not 1 <= rating <= 5:
            raise ValueError("Rating must be from 1 to 5")

        sources = retrieved_sources or []
        scores = retrieval_scores or []

        if len(sources) != len(scores):
            raise ValueError(
                "Each retrieved source must have a corresponding score"
            )

        record = FeedbackRecord(
            feedback_id=str(uuid.uuid4()),
            query=query,
            answer=answer,
            rating=rating,
            comment=comment,
            retrieved_sources=sources,
            retrieval_scores=scores,
            created_at=datetime.now(
                timezone.utc
            ).isoformat(),
        )

        with self.file_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(asdict(record), ensure_ascii=False)
                + "\n"
            )

        self.records.append(record)

        return record

    def summary(self):
        """Summarize recorded ratings."""

        if not self.records:
            return {
                "total": 0,
                "average_rating": None,
                "positive": 0,
                "negative": 0,
                "rating_distribution": {},
            }

        ratings = [record.rating for record in self.records]

        return {
            "total": len(ratings),
            "average_rating": sum(ratings) / len(ratings),
            "positive": sum(rating >= 4 for rating in ratings),
            "negative": sum(rating <= 2 for rating in ratings),
            "rating_distribution": dict(
                sorted(Counter(ratings).items())
            ),
        }

    def low_rated_feedback(self, maximum_rating=2):
        """Find feedback that may need investigation."""

        return [
            record
            for record in self.records
            if record.rating <= maximum_rating
        ]

    def source_performance(self):
        """
        Group feedback by retrieved source.

        This is descriptive only: a source appearing in a low-rated
        answer is not necessarily the cause of the poor answer.
        """

        grouped = defaultdict(list)

        for record in self.records:
            for source in set(record.retrieved_sources):
                grouped[source].append(record.rating)

        return {
            source: {
                "feedback_count": len(ratings),
                "average_rating": sum(ratings) / len(ratings),
            }
            for source, ratings in sorted(grouped.items())
        }


def demo():
    # Use a separate file for repeatable demonstrations.
    store = FeedbackStore("demo_rag_feedback.jsonl")

    examples = [
        {
            "query": "What does an RCCB do?",
            "answer": (
                "An RCCB detects residual current and disconnects "
                "a circuit when its operating conditions are met."
            ),
            "rating": 5,
            "comment": "Clear and useful explanation.",
            "sources": ["electrical_safety.txt"],
            "scores": [0.87],
        },
        {
            "query": "Why is my DC charger showing an insulation fault?",
            "answer": (
                "Check the insulation monitoring system and follow "
                "the manufacturer's safe diagnostic procedure."
            ),
            "rating": 2,
            "comment": "Needs more specific troubleshooting evidence.",
            "sources": [
                "charger_manual.txt",
                "general_electrical_notes.txt",
            ],
            "scores": [0.62, 0.39],
        },
        {
            "query": "What does an RCBO protect against?",
            "answer": (
                "An RCBO combines residual-current and "
                "overcurrent protection."
            ),
            "rating": 4,
            "comment": "Correct, but could include an example.",
            "sources": ["electrical_safety.txt"],
            "scores": [0.82],
        },
    ]

    for item in examples:
        store.add_feedback(
            query=item["query"],
            answer=item["answer"],
            rating=item["rating"],
            comment=item["comment"],
            retrieved_sources=item["sources"],
            retrieval_scores=item["scores"],
        )

    report = store.summary()

    print("=" * 60)
    print("RAG FEEDBACK REPORT")
    print("=" * 60)

    print(f"Total feedback: {report['total']}")

    average = report["average_rating"]
    print(
        "Average rating: "
        f"{average:.2f}/5" if average is not None
        else "Average rating: N/A"
    )

    print(f"Positive ratings (4-5): {report['positive']}")
    print(f"Negative ratings (1-2): {report['negative']}")
    print(f"Rating distribution: {report['rating_distribution']}")

    print("\nQueries needing investigation:")

    low_rated = store.low_rated_feedback()

    if not low_rated:
        print("None")
    else:
        for record in low_rated:
            print(f"\nQuery: {record.query}")
            print(f"Rating: {record.rating}/5")
            print(f"Comment: {record.comment}")
            print(f"Sources: {record.retrieved_sources}")

    print("\nRatings grouped by source:")

    for source, metrics in store.source_performance().items():
        print(
            f"- {source}: "
            f"{metrics['average_rating']:.2f}/5 "
            f"across {metrics['feedback_count']} feedback records"
        )

    print(f"\nFeedback saved to: {store.file_path}")


if __name__ == "__main__":
    demo()
