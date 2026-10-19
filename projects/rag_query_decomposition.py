
"""
RAG Query Decomposition

Pipeline:
Complex Query
    -> Sub-queries
    -> Retrieve Evidence for Each
    -> Deduplicate Evidence
    -> Build Structured Context

This educational version uses rule-based decomposition.
"""

import re
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class Document:
    doc_id: str
    text: str
    source: str


class QueryDecomposer:
    """Split multi-part queries into focused sub-queries."""

    @staticmethod
    def decompose(query):
        query = query.strip()

        if not query:
            return []

        # Split on common conjunctions and comparison phrasing.
        parts = re.split(
            r"\s*(?:;\s*|\?\s+|,\s*(?:and|then)\s+"
            r"|\s+\b(?:and also|also explain|then explain)\b\s+)",
            query,
            flags=re.IGNORECASE,
        )

        sub_queries = []

        for part in parts:
            part = part.strip(" \t\n?.,")

            if len(part.split()) >= 2:
                sub_queries.append(part)

        # Keep the original question if splitting was ineffective.
        if len(sub_queries) <= 1:
            return [query]

        # Remove repeated sub-queries.
        unique_queries = []
        seen = set()

        for item in sub_queries:
            key = item.lower()

            if key not in seen:
                seen.add(key)
                unique_queries.append(item)

        return unique_queries


class DecomposedRAG:
    def __init__(
        self,
        documents,
        model_name=MODEL_NAME,
        top_k=2,
        min_similarity=0.20,
    ):
        self.documents = documents
        self.top_k = top_k
        self.min_similarity = min_similarity

        self.model = SentenceTransformer(model_name)
        self.decomposer = QueryDecomposer()

        if documents:
            self.embeddings = self.model.encode(
                [doc.text for doc in documents],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            self.embeddings = np.empty((0, 0))

    def retrieve(self, query):
        """Retrieve relevant documents for one query."""

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

    def answer_context(self, query):
        """Decompose, retrieve, and build a context package."""

        sub_queries = self.decomposer.decompose(query)

        retrieved_by_query = {}
        unique_evidence = {}
        coverage = []

        for sub_query in sub_queries:
            results = self.retrieve(sub_query)

            retrieved_by_query[sub_query] = results

            coverage.append({
                "sub_query": sub_query,
                "evidence_found": bool(results),
                "result_count": len(results),
            })

            for result in results:
                # Deduplicate repeated evidence across sub-queries.
                key = result["doc_id"]

                if (
                    key not in unique_evidence
                    or result["similarity"]
                    > unique_evidence[key]["similarity"]
                ):
                    unique_evidence[key] = result

        evidence = sorted(
            unique_evidence.values(),
            key=lambda item: item["similarity"],
            reverse=True,
        )

        context = "\n\n".join(
            (
                f"Source: {item['source']}\n"
                f"Evidence: {item['text']}"
            )
            for item in evidence
        )

        return {
            "original_query": query,
            "sub_queries": sub_queries,
            "retrieved_by_query": retrieved_by_query,
            "coverage": coverage,
            "evidence": evidence,
            "context": context,
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
                "EV charger electrical protection may include "
                "residual-current devices, overcurrent protection, "
                "surge protection, and protective earthing. The "
                "required arrangement depends on charger design "
                "and applicable electrical standards."
            ),
            "ev_charger_safety.txt",
        ),
        Document(
            "doc-4",
            (
                "A DC power module converts incoming AC power "
                "into regulated DC power for charging the vehicle."
            ),
            "power_module_manual.txt",
        ),
    ]

    rag = DecomposedRAG(
        documents,
        top_k=2,
        min_similarity=0.20,
    )

    query = (
        "Compare RCCB and RCBO, and also explain "
        "their protection functions and EV charger applications."
    )

    result = rag.answer_context(query)

    print("=" * 70)
    print("ORIGINAL QUERY")
    print(result["original_query"])

    print("\nDECOMPOSED QUERIES")

    for index, sub_query in enumerate(
        result["sub_queries"],
        start=1,
    ):
        print(f"{index}. {sub_query}")

    print("\nRETRIEVAL COVERAGE")

    for item in result["coverage"]:
        status = (
            "Evidence found"
            if item["evidence_found"]
            else "No evidence found"
        )

        print(
            f"- {item['sub_query']}: "
            f"{status} ({item['result_count']} results)"
        )

    print("\nCOMBINED EVIDENCE")

    if not result["evidence"]:
        print("No matching evidence was retrieved.")
    else:
        for item in result["evidence"]:
            print(
                f"\n[{item['doc_id']}] "
                f"Score: {item['similarity']:.3f}"
            )
            print(f"Source: {item['source']}")
            print(item["text"])

    print("\nCONTEXT FOR THE GENERATION MODEL")
    print(result["context"])


if __name__ == "__main__":
    demo()
