
"""
Contextual Retrieval for RAG

Adds document and section context to chunks before
embedding, while preserving the original chunk text.
"""

from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source: str
    document_title: str
    section: str


@dataclass
class ContextualChunk:
    chunk_id: str
    original_text: str
    contextualized_text: str
    source: str
    document_title: str
    section: str


def add_context(chunk: DocumentChunk) -> ContextualChunk:
    """Attach document and section information to a chunk."""

    contextualized_text = (
        f"Document: {chunk.document_title}. "
        f"Section: {chunk.section}. "
        f"Content: {chunk.text}"
    )

    return ContextualChunk(
        chunk_id=chunk.chunk_id,
        original_text=chunk.text,
        contextualized_text=contextualized_text,
        source=chunk.source,
        document_title=chunk.document_title,
        section=chunk.section,
    )


class ContextualRetriever:
    def __init__(self, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None

    def index(self, chunks):
        """Contextualize and embed document chunks."""

        self.chunks = [add_context(chunk) for chunk in chunks]

        if not self.chunks:
            self.embeddings = np.empty((0, 0))
            return

        texts = [
            chunk.contextualized_text
            for chunk in self.chunks
        ]

        self.embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def search(self, query, top_k=3):
        """Retrieve chunks using contextualized embeddings."""

        if self.embeddings is None or not self.chunks:
            return []

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        scores = self.embeddings @ query_embedding

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:
            chunk = self.chunks[int(index)]

            results.append({
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "section": chunk.section,
                "score": float(scores[index]),
                # Return original text as answer context.
                "text": chunk.original_text,
                "contextualized_text": chunk.contextualized_text,
            })

        return results


def demo():
    documents = [
        DocumentChunk(
            chunk_id="chunk-1",
            text=(
                "It disconnects the supply when residual "
                "current exceeds its operating threshold."
            ),
            source="electrical_safety.txt",
            document_title="EV Charger Electrical Components",
            section="RCCB Protection",
        ),
        DocumentChunk(
            chunk_id="chunk-2",
            text=(
                "It monitors insulation resistance between "
                "the DC circuit and earth."
            ),
            source="dc_charger.txt",
            document_title="DC EV Charger Systems",
            section="Insulation Monitoring Device",
        ),
        DocumentChunk(
            chunk_id="chunk-3",
            text=(
                "It converts incoming AC power into "
                "regulated DC power for the charging system."
            ),
            source="power_modules.txt",
            document_title="DC EV Charger Systems",
            section="Power Module",
        ),
        DocumentChunk(
            chunk_id="chunk-4",
            text=(
                "It measures current indirectly by detecting "
                "the magnetic field around a conductor."
            ),
            source="electrical_sensors.txt",
            document_title="EV Charger Electrical Components",
            section="Current Transformer",
        ),
    ]

    retriever = ContextualRetriever()
    retriever.index(documents)

    queries = [
        "How does an RCCB protect against residual current?",
        "Which component monitors DC insulation to earth?",
        "Which unit converts AC electricity to DC?",
    ]

    for query in queries:
        print("\n" + "=" * 65)
        print(f"Query: {query}")

        for result in retriever.search(query, top_k=2):
            print(
                f"\nScore: {result['score']:.4f}"
                f"\nSource: {result['source']}"
                f"\nSection: {result['section']}"
                f"\nRetrieved text: {result['text']}"
            )


if __name__ == "__main__":
    demo()
