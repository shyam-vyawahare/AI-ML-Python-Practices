"""
RAG Retrieval Timeout Management
--------------------------------
Demonstrates timeout-aware retrieval with concurrent execution.

Features:
- Thread-based retrieval execution
- Configurable timeouts
- Timeout exception handling
- Fallback retrieval
- Execution latency measurement
- JSON report export

No external dependencies required.

Note:
Python threads cannot forcibly terminate a running function.
Timed-out tasks may continue running in the background.
"""

from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from dataclasses import dataclass, asdict
from typing import List, Callable
import json
import time


@dataclass
class RetrievedDocument:
    document_id: str
    content: str
    score: float


@dataclass
class RetrievalAttempt:
    strategy: str
    status: str
    duration_seconds: float
    result_count: int
    error: str


@dataclass
class TimeoutReport:
    query: str
    successful_strategy: str
    total_duration_seconds: float
    final_status: str
    documents: List[dict]
    attempts: List[dict]


class TimeoutAwareRetriever:
    """Execute retrieval strategies with timeout protection."""

    def __init__(
        self,
        timeout_seconds: float = 2.0,
    ):
        if timeout_seconds <= 0:
            raise ValueError("Timeout must be positive.")

        self.timeout_seconds = timeout_seconds

    @staticmethod
    def primary_retriever(
        query: str,
    ) -> List[RetrievedDocument]:
        """Simulate a slow primary retrieval service."""

        time.sleep(3)

        return [
            RetrievedDocument(
                "D1",
                "Primary vector retrieval result.",
                0.92,
            )
        ]

    @staticmethod
    def fallback_retriever(
        query: str,
    ) -> List[RetrievedDocument]:
        """Simulate a faster fallback retrieval service."""

        time.sleep(0.2)

        return [
            RetrievedDocument(
                "D2",
                "Fallback keyword retrieval result.",
                0.84,
            ),
            RetrievedDocument(
                "D3",
                "Additional fallback result.",
                0.79,
            ),
        ]

    def execute_with_timeout(
        self,
        executor: ThreadPoolExecutor,
        strategy_name: str,
        retriever: Callable,
        query: str,
    ):
        """Execute one retriever with a timeout."""

        start_time = time.perf_counter()
        future = executor.submit(retriever, query)

        try:
            documents = future.result(
                timeout=self.timeout_seconds
            )

            status = (
                "success" if documents else "empty_results"
            )

            return (
                documents,
                RetrievalAttempt(
                    strategy=strategy_name,
                    status=status,
                    duration_seconds=round(
                        time.perf_counter() - start_time, 3
                    ),
                    result_count=len(documents),
                    error="",
                ),
            )

        except FutureTimeoutError:
            future.cancel()

            return (
                [],
                RetrievalAttempt(
                    strategy=strategy_name,
                    status="timeout",
                    duration_seconds=round(
                        time.perf_counter() - start_time, 3
                    ),
                    result_count=0,
                    error="Retrieval exceeded configured timeout.",
                ),
            )

        except Exception as error:
            return (
                [],
                RetrievalAttempt(
                    strategy=strategy_name,
                    status="failed",
                    duration_seconds=round(
                        time.perf_counter() - start_time, 3
                    ),
                    result_count=0,
                    error=str(error),
                ),
            )

    def retrieve(self, query: str) -> TimeoutReport:
        """Execute primary retrieval and fallback if necessary."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        overall_start = time.perf_counter()

        attempts = []
        documents = []
        successful_strategy = "none"

        # Two workers allow a timed-out task to remain active
        # while the fallback strategy is executed.
        with ThreadPoolExecutor(max_workers=2) as executor:

            strategies = [
                ("primary_vector", self.primary_retriever),
                ("keyword_fallback", self.fallback_retriever),
            ]

            for name, retriever in strategies:
                results, attempt = self.execute_with_timeout(
                    executor,
                    name,
                    retriever,
                    query,
                )

                attempts.append(attempt)

                if results:
                    documents = results
                    successful_strategy = name
                    break

        total_duration = time.perf_counter() - overall_start

        return TimeoutReport(
            query=query,
            successful_strategy=successful_strategy,
            total_duration_seconds=round(total_duration, 3),
            final_status=(
                "success" if documents else "failed"
            ),
            documents=[
                asdict(document)
                for document in documents
            ],
            attempts=[
                asdict(attempt)
                for attempt in attempts
            ],
        )


def save_report(
    report: TimeoutReport,
    filepath: str,
) -> None:
    """Export timeout and latency information."""

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(asdict(report), file, indent=4)

    print(f"\nTimeout report saved to: {filepath}")


def main():
    retriever = TimeoutAwareRetriever(
        timeout_seconds=2.0
    )

    query = "How does a neural network learn?"

    report = retriever.retrieve(query)

    print("RAG RETRIEVAL TIMEOUT MANAGEMENT")
    print("=" * 50)

    print(f"Query: {report.query}")
    print(f"Successful strategy: {report.successful_strategy}")
    print(f"Final status: {report.final_status}")

    print(
        f"Total duration: "
        f"{report.total_duration_seconds:.3f} seconds"
    )

    print("\nRETRIEVAL ATTEMPTS")

    for attempt in report.attempts:
        print(f"\nStrategy: {attempt['strategy']}")
        print(f"Status: {attempt['status']}")
        print(f"Duration: {attempt['duration_seconds']} seconds")
        print(f"Results: {attempt['result_count']}")

        if attempt["error"]:
            print(f"Error: {attempt['error']}")

    print("\nFINAL DOCUMENTS")

    for document in report.documents:
        print(
            f"{document['document_id']} "
            f"| Score: {document['score']}"
        )

    save_report(report, "retrieval_timeout_report.json")


if __name__ == "__main__":
    main()
