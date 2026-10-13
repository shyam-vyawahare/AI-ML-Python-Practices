
"""
RAG Query Routing

Routes questions to the most suitable knowledge source
before performing semantic retrieval.

This version uses rule-based routing. It can later be
extended with an LLM-based or classifier-based router.
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
    category: str


class QueryRouter:
    def __init__(self):
        self.routes = {
            "technical": {
                "keywords": {
                    "component", "rating", "voltage", "current",
                    "specification", "rccb", "contactor", "fuse",
                    "power module", "wiring", "circuit",
                },
                "description": "Technical documentation",
            },
            "troubleshooting": {
                "keywords": {
                    "fault", "error", "repair", "failure",
                    "troubleshoot", "maintenance", "issue",
                    "alarm", "diagnose", "not working",
                },
                "description": "Repair and troubleshooting manuals",
            },
            "general": {
                "keywords": set(),
                "description": "General knowledge documents",
            },
        }

    @staticmethod
    def normalize(query):
        return re.sub(
            r"\s+",
            " ",
            query.lower().strip(),
        )

    def route(self, query):
        """Choose a category using keyword matches."""

        normalized_query = self.normalize(query)

        scores = {}

        for category, config in self.routes.items():
            if category == "general":
                continue

            scores[category] = sum(
                1
                for keyword in config["keywords"]
                if re.search(
                    r"\b" + re.escape(keyword) + r"\b",
                    normalized_query,
                )
            )

        best_category = max(
            scores,
            key=scores.get,
            default="general",
        )

        # Avoid forcing a specialized route without evidence.
        if scores.get(best_category, 0) == 0:
            best_category = "general"

        return {
            "category": best_category,
            "description": self.routes[best_category]["description"],
            "route_scores": scores,
        }


class RoutedRetriever:
    def __init__(self, documents, model_name=MODEL_NAME):
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.router = QueryRouter()

        self.category_indices = {}

        for index, document in enumerate(documents):
            self.category_indices.setdefault(
                document.category,
                [],
            ).append(index)

        if documents:
            self.embeddings = self.model.encode(
                [document.text for document in documents],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            self.embeddings = np.empty((0, 0))

    def search(self, query, top_k=3):
        """Route the query, then search the selected source."""

        routing = self.router.route(query)
        category = routing["category"]

        # A general query can search all available documents.
        if category == "general":
            candidate_indices = list(range(len(self.documents)))
        else:
            candidate_indices = self.category_indices.get(
                category,
                [],
            )

            # Fall back to global retrieval if the selected
            # category has no indexed documents.
            if not candidate_indices:
                candidate_indices = list(range(len(self.documents)))

        if not candidate_indices:
            return {
                "query": query,
                "routing": routing,
                "results": [],
            }

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        candidate_embeddings = self.embeddings[candidate_indices]
        scores = candidate_embeddings @ query_embedding

        ranked_positions = np.argsort(scores)[::-1][:top_k]

        results = []

        for position in ranked_positions:
            document_index = candidate_indices[int(position)]
            document = self.documents[document_index]

            results.append({
                "doc_id": document.doc_id,
                "source": document.source,
                "category": document.category,
                "text": document.text,
                "similarity": float(scores[position]),
            })

        return {
            "query": query,
            "routing": routing,
            "results": results,
        }


def demo():
    documents = [
        Document(
            "tech-1",
            "An RCCB detects residual current and disconnects "
            "the circuit when its trip conditions are met.",
            "electrical_components.txt",
            "technical",
        ),
        Document(
            "tech-2",
            "A DC contactor switches the high-voltage DC circuit "
            "according to the charger's control sequence.",
            "dc_charger_manual.txt",
            "technical",
        ),
        Document(
            "repair-1",
            "If a charger reports an insulation fault, inspect "
            "the insulation monitoring system and check for "
            "leakage paths before restarting.",
            "troubleshooting_manual.txt",
            "troubleshooting",
        ),
        Document(
            "repair-2",
            "If the charger fails to start, inspect active alarms, "
            "communication status, emergency-stop state, and "
            "contactor feedback.",
            "maintenance_manual.txt",
            "troubleshooting",
        ),
        Document(
            "general-1",
            "Electric vehicles use electric motors for propulsion "
            "and rechargeable batteries for energy storage.",
            "ev_overview.txt",
            "general",
        ),
    ]

    retriever = RoutedRetriever(documents)

    queries = [
        "What does an RCCB do?",
        "How do I troubleshoot a charger that will not start?",
        "Explain electric vehicles.",
    ]

    for query in queries:
        output = retriever.search(query, top_k=2)

        print("\n" + "=" * 65)
        print(f"Query: {query}")
        print(
            f"Route: {output['routing']['category']}"
        )
        print(
            f"Source type: {output['routing']['description']}"
        )

        for result in output["results"]:
            print(
                f"\n[{result['doc_id']}] "
                f"Similarity: {result['similarity']:.3f}"
            )
            print(f"Source: {result['source']}")
            print(f"Text: {result['text']}")


if __name__ == "__main__":
    demo()
