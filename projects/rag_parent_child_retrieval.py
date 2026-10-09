
"""
Parent-Child Retrieval for RAG

Small child chunks improve retrieval precision.
Their larger parent chunks provide richer context.
"""

from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class ParentDocument:
    parent_id: str
    text: str
    source: str


@dataclass
class ChildChunk:
    child_id: str
    parent_id: str
    text: str


class ParentChildRetriever:
    def __init__(self, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.parents = {}
        self.children = []
        self.embeddings = np.empty((0, 0))

    def add_document(
        self,
        parent_id,
        text,
        source,
        child_size=50,
        overlap=10,
    ):
        """Store a parent and split it into child chunks."""

        words = text.split()

        if child_size <= 0:
            raise ValueError("child_size must be positive")

        if overlap < 0 or overlap >= child_size:
            raise ValueError(
                "overlap must be >= 0 and smaller than child_size"
            )

        if not words:
            return

        parent = ParentDocument(
            parent_id=parent_id,
            text=text,
            source=source,
        )

        self.parents[parent_id] = parent

        start = 0
        child_index = 0

        while start < len(words):
            end = min(start + child_size, len(words))

            child_text = " ".join(words[start:end])

            self.children.append(
                ChildChunk(
                    child_id=f"{parent_id}-child-{child_index}",
                    parent_id=parent_id,
                    text=child_text,
                )
            )

            child_index += 1

            if end == len(words):
                break

            start = end - overlap

        self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild embeddings after documents are added."""

        if not self.children:
            self.embeddings = np.empty((0, 0))
            return

        texts = [child.text for child in self.children]

        self.embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def search(self, query, top_k=3):
        """
        Search child chunks, then return their parent documents.
        Deduplicate parents so one parent isn't returned repeatedly.
        """

        if not self.children:
            return []

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        scores = self.embeddings @ query_embedding
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        seen_parents = set()

        for index in ranked_indices:
            child = self.children[int(index)]

            if child.parent_id in seen_parents:
                continue

            parent = self.parents[child.parent_id]

            results.append({
                "parent_id": parent.parent_id,
                "source": parent.source,
                "parent_text": parent.text,
                "matched_child": child.text,
                "score": float(scores[index]),
            })

            seen_parents.add(child.parent_id)

            if len(results) >= top_k:
                break

        return results


def demo():
    retriever = ParentChildRetriever()

    retriever.add_document(
        parent_id="doc-1",
        source="machine_learning.txt",
        text=(
            "Machine learning enables computers to learn "
            "patterns from data. Supervised learning uses "
            "labelled examples for training. Classification "
            "predicts categories, while regression predicts "
            "continuous numerical values. Model performance "
            "can be evaluated using appropriate metrics."
        ),
        child_size=12,
        overlap=3,
    )

    retriever.add_document(
        parent_id="doc-2",
        source="deep_learning.txt",
        text=(
            "Deep learning uses neural networks with multiple "
            "layers. Convolutional neural networks are useful "
            "for image processing. Recurrent architectures "
            "process sequential information. Transformers use "
            "attention mechanisms for contextual representation."
        ),
        child_size=12,
        overlap=3,
    )

    query = "How does supervised learning train classification models?"

    print(f"\nQuery: {query}")
    print("\nRetrieved parent documents:")

    for result in retriever.search(query, top_k=2):
        print("\n" + "=" * 60)
        print(f"Parent: {result['parent_id']}")
        print(f"Source: {result['source']}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Matched child: {result['matched_child']}")
        print(f"Parent context: {result['parent_text']}")


if __name__ == "__main__":
    demo()
