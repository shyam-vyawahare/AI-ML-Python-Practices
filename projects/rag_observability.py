
"""
RAG Observability

Tracks:
- Query and embedding latency
- Retrieval latency
- Total request latency
- Retrieved result count
- Similarity scores
- Errors
- Aggregate performance statistics

Uses JSONL for persistent trace logging.
"""

import json
import logging
import time
import uuid

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
TRACE_FILE = Path("rag_traces.jsonl")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("rag_observability")


class TraceLogger:
    """Persist one structured trace per request."""

    def __init__(self, file_path=TRACE_FILE):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def log(self, trace):
        with self.file_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(trace, ensure_ascii=False) + "\n"
            )


class ObservableRAG:
    def __init__(
        self,
        documents,
        model_name=MODEL_NAME,
        trace_file=TRACE_FILE,
    ):
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.trace_logger = TraceLogger(trace_file)
        self.traces = []

        # Measure initial indexing separately from query latency.
        indexing_start = time.perf_counter()

        if documents:
            self.embeddings = self.model.encode(
                [doc["text"] for doc in documents],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            self.embeddings = np.empty((0, 0))

        self.indexing_ms = (
            time.perf_counter() - indexing_start
        ) * 1000

        logger.info(
            "Index initialized: documents=%d indexing_ms=%.2f",
            len(documents),
            self.indexing_ms,
        )

    def search(self, query, top_k=3):
        """Retrieve documents while recording performance metrics."""

        trace_id = str(uuid.uuid4())
        request_start = time.perf_counter()

        trace = {
            "trace_id": trace_id,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "query": query,
            "status": "success",
            "document_count": len(self.documents),
            "top_k": top_k,
            "embedding_ms": 0.0,
            "retrieval_ms": 0.0,
            "total_ms": 0.0,
            "result_count": 0,
            "top_similarity": None,
            "mean_similarity": None,
            "error_type": None,
        }

        try:
            if top_k < 1:
                raise ValueError("top_k must be at least 1")

            if not query.strip():
                raise ValueError("Query cannot be empty")

            if not self.documents:
                trace["status"] = "empty_index"

                return {
                    "trace_id": trace_id,
                    "results": [],
                    "trace": trace,
                }

            embedding_start = time.perf_counter()

            query_embedding = self.model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]

            trace["embedding_ms"] = (
                time.perf_counter() - embedding_start
            ) * 1000

            retrieval_start = time.perf_counter()

            scores = self.embeddings @ query_embedding

            ranked_indices = np.argsort(scores)[::-1][:top_k]

            results = []

            for index in ranked_indices:
                document = self.documents[int(index)]

                results.append({
                    "doc_id": document["doc_id"],
                    "source": document["source"],
                    "text": document["text"],
                    "similarity": float(scores[index]),
                })

            trace["retrieval_ms"] = (
                time.perf_counter() - retrieval_start
            ) * 1000

            trace["result_count"] = len(results)

            if results:
                result_scores = [
                    item["similarity"]
                    for item in results
                ]

                trace["top_similarity"] = max(result_scores)
                trace["mean_similarity"] = float(
                    np.mean(result_scores)
                )

            return {
                "trace_id": trace_id,
                "results": results,
                "trace": trace,
            }

        except Exception as error:
            trace["status"] = "error"
            trace["error_type"] = type(error).__name__

            logger.exception(
                "RAG request failed: trace_id=%s",
                trace_id,
            )

            raise

        finally:
            trace["total_ms"] = (
                time.perf_counter() - request_start
            ) * 1000

            self.traces.append(trace)
            self.trace_logger.log(trace)

            logger.info(
                "trace_id=%s status=%s total_ms=%.2f "
                "results=%d",
                trace_id,
                trace["status"],
                trace["total_ms"],
                trace["result_count"],
            )

    def metrics_report(self):
        """Summarize requests recorded during this process."""

        if not self.traces:
            return {
                "total_requests": 0,
                "statuses": {},
            }

        successful = [
            trace
            for trace in self.traces
            if trace["status"] == "success"
        ]

        total_times = [
            trace["total_ms"]
            for trace in successful
        ]

        embedding_times = [
            trace["embedding_ms"]
            for trace in successful
        ]

        retrieval_times = [
            trace["retrieval_ms"]
            for trace in successful
        ]

        statuses = Counter(
            trace["status"]
            for trace in self.traces
        )

        def percentile(values, percentile_value):
            if not values:
                return None

            return float(
                np.percentile(values, percentile_value)
            )

        return {
            "total_requests": len(self.traces),
            "statuses": dict(statuses),
            "average_total_ms": (
                float(np.mean(total_times))
                if total_times else None
            ),
            "p95_total_ms": percentile(total_times, 95),
            "average_embedding_ms": (
                float(np.mean(embedding_times))
                if embedding_times else None
            ),
            "average_retrieval_ms": (
                float(np.mean(retrieval_times))
                if retrieval_times else None
            ),
            "empty_result_requests": sum(
                trace["result_count"] == 0
                for trace in successful
            ),
            "failed_requests": statuses.get("error", 0),
        }


def demo():
    documents = [
        {
            "doc_id": "doc-1",
            "source": "rccb_manual.txt",
            "text": (
                "An RCCB detects residual current and disconnects "
                "a circuit when its operating conditions are met."
            ),
        },
        {
            "doc_id": "doc-2",
            "source": "rcbo_manual.txt",
            "text": (
                "An RCBO combines residual-current protection "
                "with overcurrent protection."
            ),
        },
        {
            "doc_id": "doc-3",
            "source": "charger_manual.txt",
            "text": (
                "DC charger power modules convert incoming AC "
                "power into regulated DC power."
            ),
        },
    ]

    rag = ObservableRAG(
        documents,
        trace_file="demo_rag_traces.jsonl",
    )

    queries = [
        "What does an RCCB do?",
        "How does an RCBO protect a circuit?",
        "How does a DC charger convert power?",
    ]

    for query in queries:
        output = rag.search(query, top_k=2)

        print("\n" + "=" * 60)
        print(f"Query: {query}")
        print(f"Trace ID: {output['trace_id']}")

        for result in output["results"]:
            print(
                f"\n{result['source']} "
                f"(score={result['similarity']:.3f})"
            )
            print(result["text"])

    print("\n" + "=" * 60)
    print("OBSERVABILITY REPORT")
    print(
        json.dumps(
            rag.metrics_report(),
            indent=2,
        )
    )


if __name__ == "__main__":
    demo()
