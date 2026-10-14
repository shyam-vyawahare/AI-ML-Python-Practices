
"""
Hierarchical Parent-Document Retrieval for RAG

Stage 1: Retrieve relevant parent sections.
Stage 2: Retrieve the most relevant child chunks
         within the selected parents.
"""

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class ParentDocument:
    parent_id: str
    title: str
    text: str
    source: str


@dataclass
class ChildChunk:
    child_id: str
    parent_id: str
    text: str


class HierarchicalRetriever:
    def __init__(self, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)

        self.parents = {}
        self.children = []

        self.parent_embeddings = None
        self.child_embeddings = None

    @staticmethod
    def split_text(text, chunk_size=45, overlap=8):
        """Split text into overlapping word-based chunks."""

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "overlap must be >= 0 and smaller than chunk_size"
            )

        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))

            if end == len(words):
                break

            start = end - overlap

        return chunks

    def add_document(
        self,
        parent_id,
        title,
        text,
        source,
        chunk_size=45,
        overlap=8,
    ):
        """Store a parent section and its child chunks."""

        if parent_id in self.parents:
            raise ValueError(
                f"Duplicate parent ID: {parent_id}"
            )

        self.parents[parent_id] = ParentDocument(
            parent_id=parent_id,
            title=title,
            text=text,
            source=source,
        )

        for index, chunk_text in enumerate(
            self.split_text(text, chunk_size, overlap)
        ):
            self.children.append(
                ChildChunk(
                    child_id=f"{parent_id}-child-{index}",
                    parent_id=parent_id,
                    text=chunk_text,
                )
            )

        self._rebuild_index()

    def _rebuild_index(self):
        """Generate normalized embeddings for parents and children."""

        if not self.parents:
            self.parent_embeddings = None
            self.child_embeddings = None
            return

        parent_list = list(self.parents.values())

        parent_texts = [
            f"{parent.title}. {parent.text}"
            for parent in parent_list
        ]

        self.parent_embeddings = self.model.encode(
            parent_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        child_texts = [
            child.text for child in self.children
        ]

        if child_texts:
            self.child_embeddings = self.model.encode(
                child_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            self.child_embeddings = None

    def search(
        self,
        query,
        parent_top_k=2,
        child_top_k=3,
    ):
        """
        Retrieve parents first, then rank children
        only within the selected parent sections.
        """

        if (
            not self.parents
            or self.parent_embeddings is None
            or not self.children
            or self.child_embeddings is None
        ):
            return {
                "query": query,
                "parents": [],
                "chunks": [],
            }

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        # Stage 1: Rank parent sections.
        parent_scores = (
            self.parent_embeddings @ query_embedding
        )

        parent_order = np.argsort(parent_scores)[::-1]

        parent_list = list(self.parents.values())

        selected_parents = []

        for index in parent_order[:parent_top_k]:
            parent = parent_list[int(index)]

            selected_parents.append({
                "parent_id": parent.parent_id,
                "title": parent.title,
                "source": parent.source,
                "score": float(parent_scores[index]),
            })

        selected_parent_ids = {
            parent["parent_id"]
            for parent in selected_parents
        }

        # Stage 2: Rank children within selected parents.
        child_scores = (
            self.child_embeddings @ query_embedding
        )

        candidates = []

        for index, child in enumerate(self.children):
            if child.parent_id not in selected_parent_ids:
                continue

            candidates.append({
                "child_id": child.child_id,
                "parent_id": child.parent_id,
                "text": child.text,
                "score": float(child_scores[index]),
            })

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return {
            "query": query,
            "parents": selected_parents,
            "chunks": candidates[:child_top_k],
        }


def demo():
    retriever = HierarchicalRetriever()

    retriever.add_document(
        parent_id="section-1",
        title="EV Charger Electrical Protection",
        source="electrical_manual.txt",
        text=(
            "Electrical protection devices help protect charging "
            "equipment and electrical installations. An RCCB "
            "detects residual current and disconnects a circuit "
            "when its trip conditions are met. An RCBO combines "
            "residual-current protection with overcurrent "
            "protection. Surge protection devices help protect "
            "equipment from transient overvoltages."
        ),
        chunk_size=20,
        overlap=4,
    )

    retriever.add_document(
        parent_id="section-2",
        title="DC Charger Power Conversion",
        source="power_module_manual.txt",
        text=(
            "A DC charger power module converts incoming AC "
            "electrical power into regulated DC power. Multiple "
            "modules may operate in parallel to increase the "
            "available charging power. The controller manages "
            "module operation and monitors faults. Cooling "
            "systems help maintain safe operating temperatures."
        ),
        chunk_size=20,
        overlap=4,
    )

    retriever.add_document(
        parent_id="section-3",
        title="EV Charger Fault Diagnosis",
        source="maintenance_manual.txt",
        text=(
            "Troubleshooting begins by identifying the active "
            "fault code and checking charger status. Inspect "
            "connectors, communication links, emergency-stop "
            "state, and contactor feedback as appropriate. "
            "Follow the manufacturer's isolation and safety "
            "procedures before inspecting electrical hardware."
        ),
        chunk_size=20,
        overlap=4,
    )

    query = "How does an RCCB protect an EV charger?"

    result = retriever.search(
        query,
        parent_top_k=2,
        child_top_k=3,
    )

    print("=" * 65)
    print(f"Query: {result['query']}")

    print("\nStage 1: Selected parent sections")

    for parent in result["parents"]:
        print(
            f"\n{parent['title']} "
            f"(score={parent['score']:.3f})"
        )
        print(f"Source: {parent['source']}")

    print("\nStage 2: Most relevant child chunks")

    for chunk in result["chunks"]:
        print(
            f"\n{chunk['child_id']} "
            f"(score={chunk['score']:.3f})"
        )
        print(chunk["text"])


if __name__ == "__main__":
    demo()
