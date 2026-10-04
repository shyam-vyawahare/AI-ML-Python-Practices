"""
Persistent Vector Store Practice

A lightweight vector database implemented from scratch.

Features:
    - Document embedding
    - Persistent vector storage
    - Metadata storage
    - Add documents
    - Load existing store
    - Similarity search
    - Top-K retrieval

Storage:

    vector_store/
        vectors.npy
        metadata.json

This is an educational implementation.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Document:
    doc_id: str
    text: str
    source: str = ""


@dataclass
class SearchResult:
    document: Document
    score: float


# ============================================================
# PERSISTENT VECTOR STORE
# ============================================================

class PersistentVectorStore:

    def __init__(
        self,
        storage_path: str = "vector_store"
    ):

        self.storage_path = Path(
            storage_path
        )

        self.storage_path.mkdir(
            parents=True,
            exist_ok=True
        )

        self.vector_file = (
            self.storage_path /
            "vectors.npy"
        )

        self.metadata_file = (
            self.storage_path /
            "metadata.json"
        )

        self.documents: List[Document] = []

        self.vectors = np.empty(
            (0, 384),
            dtype=np.float32
        )

    # ========================================================
    # LOAD
    # ========================================================

    def load(self) -> None:
        """
        Load vectors and metadata from disk.
        """

        if (
            not self.vector_file.exists()
            or
            not self.metadata_file.exists()
        ):

            print(
                "No existing vector store found."
            )

            return

        self.vectors = np.load(
            self.vector_file
        )

        with open(
            self.metadata_file,
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(
                file
            )

        self.documents = [
            Document(**item)
            for item in metadata
        ]

        print(
            f"Loaded {len(self.documents)} "
            "documents from disk."
        )

    # ========================================================
    # SAVE
    # ========================================================

    def save(self) -> None:
        """
        Persist vectors and metadata to disk.
        """

        np.save(
            self.vector_file,
            self.vectors
        )

        metadata = [
            asdict(document)
            for document in self.documents
        ]

        with open(
            self.metadata_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"Saved {len(self.documents)} "
            "documents to disk."
        )

    # ========================================================
    # ADD DOCUMENTS
    # ========================================================

    def add_documents(
        self,
        documents: List[Document],
        model: SentenceTransformer
    ) -> None:
        """
        Embed and add documents to the store.
        """

        if not documents:
            return

        texts = [
            document.text
            for document in documents
        ]

        embeddings = model.encode(
            texts,
            convert_to_numpy=True
        )

        embeddings = embeddings.astype(
            np.float32
        )

        embeddings = self._normalize(
            embeddings
        )

        self.documents.extend(
            documents
        )

        if self.vectors.size == 0:

            self.vectors = embeddings

        else:

            self.vectors = np.vstack(
                [
                    self.vectors,
                    embeddings
                ]
            )

        print(
            f"Added {len(documents)} documents."
        )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize(
        vectors: np.ndarray
    ) -> np.ndarray:
        """
        Normalize vectors for cosine similarity.
        """

        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True
        )

        norms = np.maximum(
            norms,
            1e-12
        )

        return vectors / norms

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        model: SentenceTransformer,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Search the persistent vector store.
        """

        if not self.documents:

            return []

        query_vector = model.encode(
            query,
            convert_to_numpy=True
        )

        query_vector = self._normalize(
            query_vector.reshape(1, -1)
        )[0]

        scores = (
            self.vectors @ query_vector
        )

        ranked_indices = np.argsort(
            scores
        )[::-1]

        results = []

        for index in ranked_indices:

            score = float(
                scores[index]
            )

            if score < min_score:
                continue

            results.append(
                SearchResult(
                    document=self.documents[index],
                    score=score
                )
            )

            if len(results) >= top_k:
                break

        return results

    # ========================================================
    # DELETE
    # ========================================================

    def delete(
        self,
        doc_id: str
    ) -> bool:
        """
        Delete a document from the store.
        """

        index = None

        for position, document in enumerate(
            self.documents
        ):

            if document.doc_id == doc_id:

                index = position
                break

        if index is None:
            return False

        self.documents.pop(
            index
        )

        self.vectors = np.delete(
            self.vectors,
            index,
            axis=0
        )

        return True

    # ========================================================
    # CLEAR
    # ========================================================

    def clear(self) -> None:
        """
        Remove all stored documents and vectors.
        """

        self.documents.clear()

        dimension = (
            self.vectors.shape[1]
            if self.vectors.ndim == 2
            and self.vectors.shape[1] > 0
            else 384
        )

        self.vectors = np.empty(
            (0, dimension),
            dtype=np.float32
        )

    # ========================================================
    # SIZE
    # ========================================================

    def size(self) -> int:
        """
        Return number of stored documents.
        """

        return len(
            self.documents
        )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    results: List[SearchResult]
) -> None:

    print("\n")
    print("=" * 70)
    print("SEARCH RESULTS")
    print("=" * 70)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRank {rank}"
        )

        print(
            f"ID: {result.document.doc_id}"
        )

        print(
            f"Score: {result.score:.4f}"
        )

        print(
            f"Source: {result.document.source}"
        )

        print(
            f"Text: {result.document.text}"
        )


# ============================================================
# SAMPLE DOCUMENTS
# ============================================================

documents = [

    Document(
        doc_id="doc_01",
        text=(
            "Machine learning allows systems to learn "
            "patterns from data."
        ),
        source="ml_notes.txt"
    ),

    Document(
        doc_id="doc_02",
        text=(
            "Gradient descent is an optimization "
            "algorithm used to minimize a loss function."
        ),
        source="optimization_notes.txt"
    ),

    Document(
        doc_id="doc_03",
        text=(
            "Retrieval augmented generation retrieves "
            "external information before generating answers."
        ),
        source="rag_notes.txt"
    ),

    Document(
        doc_id="doc_04",
        text=(
            "Vector databases store embeddings and "
            "support similarity search."
        ),
        source="vector_db_notes.txt"
    ),

    Document(
        doc_id="doc_05",
        text=(
            "Neural networks consist of interconnected "
            "layers of artificial neurons."
        ),
        source="deep_learning_notes.txt"
    ),
]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("PERSISTENT VECTOR STORE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # --------------------------------------------------------
    # Create store
    # --------------------------------------------------------

    store = PersistentVectorStore(
        storage_path="vector_store"
    )

    # --------------------------------------------------------
    # Load existing data
    # --------------------------------------------------------

    store.load()

    # --------------------------------------------------------
    # Initialize store if empty
    # --------------------------------------------------------

    if store.size() == 0:

        store.add_documents(
            documents,
            model
        )

        store.save()

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    query = (
        "How can a system retrieve "
        "information using embeddings?"
    )

    results = store.search(
        query=query,
        model=model,
        top_k=3
    )

    display_results(
        results
    )

    # --------------------------------------------------------
    # Add a new document
    # --------------------------------------------------------

    new_document = Document(
        doc_id="doc_06",
        text=(
            "Semantic search uses embeddings to find "
            "documents with similar meaning."
        ),
        source="semantic_search_notes.txt"
    )

    # Avoid duplicate IDs.
    existing_ids = {
        document.doc_id
        for document in store.documents
    }

    if new_document.doc_id not in existing_ids:

        store.add_documents(
            [new_document],
            model
        )

        store.save()

    # --------------------------------------------------------
    # Search again
    # --------------------------------------------------------

    query = (
        "How does semantic search find similar documents?"
    )

    results = store.search(
        query=query,
        model=model,
        top_k=3
    )

    display_results(
        results
    )

    # --------------------------------------------------------
    # Store information
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("VECTOR STORE INFORMATION")
    print("=" * 70)

    print(
        f"\nDocuments: {store.size()}"
    )

    print(
        f"Vector Shape: {store.vectors.shape}"
    )

    print(
        f"Storage Directory: "
        f"{store.storage_path}"
    )
