"""
RAG Metadata Filtering Practice

Combines semantic similarity with structured metadata filters.

Supported metadata:
    - source
    - document_type
    - category
    - page
    - year

Pipeline:

Query
  ↓
Metadata Filtering
  ↓
Semantic Similarity
  ↓
Score Threshold
  ↓
Top-K Results
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Document:
    doc_id: str
    text: str
    source: str
    document_type: str
    category: str
    page: int
    year: int


@dataclass
class SearchResult:
    document: Document
    score: float


# ============================================================
# VECTOR UTILITIES
# ============================================================

def normalize_vectors(
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


# ============================================================
# VECTOR STORE
# ============================================================

class MetadataVectorStore:

    def __init__(
        self,
        model: SentenceTransformer
    ):
        self.model = model

        self.documents: List[
            Document
        ] = []

        self.embeddings = np.empty(
            (0, 384),
            dtype=np.float32
        )

    # ========================================================
    # ADD DOCUMENTS
    # ========================================================

    def add_documents(
        self,
        documents: List[Document]
    ) -> None:
        """
        Add documents and generate embeddings.
        """

        if not documents:
            return

        texts = [
            document.text
            for document in documents
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True
        )

        embeddings = normalize_vectors(
            embeddings
        ).astype(
            np.float32
        )

        self.documents.extend(
            documents
        )

        if self.embeddings.size == 0:

            self.embeddings = embeddings

        else:

            self.embeddings = np.vstack(
                [
                    self.embeddings,
                    embeddings
                ]
            )

    # ========================================================
    # METADATA MATCHING
    # ========================================================

    @staticmethod
    def matches_filter(
        document: Document,
        filters: Dict[str, Any]
    ) -> bool:
        """
        Check whether a document satisfies all filters.
        """

        for field, expected in filters.items():

            actual = getattr(
                document,
                field,
                None
            )

            # ------------------------------------------------
            # List filter
            # ------------------------------------------------

            if isinstance(
                expected,
                list
            ):

                if actual not in expected:
                    return False

            # ------------------------------------------------
            # Range filter
            # ------------------------------------------------

            elif isinstance(
                expected,
                dict
            ):

                if "min" in expected:

                    if actual < expected["min"]:
                        return False

                if "max" in expected:

                    if actual > expected["max"]:
                        return False

            # ------------------------------------------------
            # Exact match
            # ------------------------------------------------

            else:

                if actual != expected:
                    return False

        return True

    # ========================================================
    # CANDIDATE FILTERING
    # ========================================================

    def filter_documents(
        self,
        filters: Optional[
            Dict[str, Any]
        ] = None
    ) -> List[int]:
        """
        Return indexes of documents satisfying filters.
        """

        if not filters:
            return list(
                range(
                    len(self.documents)
                )
            )

        matching_indexes = []

        for index, document in enumerate(
            self.documents
        ):

            if self.matches_filter(
                document,
                filters
            ):

                matching_indexes.append(
                    index
                )

        return matching_indexes

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[
            Dict[str, Any]
        ] = None,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Perform metadata-filtered semantic search.
        """

        candidate_indexes = (
            self.filter_documents(
                filters
            )
        )

        if not candidate_indexes:
            return []

        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True
        )

        query_embedding = normalize_vectors(
            query_embedding.reshape(
                1,
                -1
            )
        )[0]

        candidate_vectors = (
            self.embeddings[
                candidate_indexes
            ]
        )

        scores = (
            candidate_vectors
            @
            query_embedding
        )

        ranked = np.argsort(
            scores
        )[::-1]

        results = []

        for position in ranked:

            score = float(
                scores[position]
            )

            if score < min_score:
                continue

            document_index = (
                candidate_indexes[position]
            )

            results.append(
                SearchResult(
                    document=self.documents[
                        document_index
                    ],
                    score=score
                )
            )

            if len(results) >= top_k:
                break

        return results


# ============================================================
# RESULT DISPLAY
# ============================================================

def display_results(
    results: List[SearchResult],
    title: str
) -> None:

    print("\n")
    print("=" * 75)
    print(title)
    print("=" * 75)

    if not results:

        print(
            "\nNo matching documents found."
        )

        return

    for rank, result in enumerate(
        results,
        start=1
    ):

        document = result.document

        print(
            f"\nRank {rank}"
        )

        print(
            f"ID: {document.doc_id}"
        )

        print(
            f"Score: {result.score:.4f}"
        )

        print(
            f"Source: {document.source}"
        )

        print(
            f"Type: {document.document_type}"
        )

        print(
            f"Category: {document.category}"
        )

        print(
            f"Page: {document.page}"
        )

        print(
            f"Year: {document.year}"
        )

        print(
            f"Text: {document.text}"
        )


# ============================================================
# SAMPLE KNOWLEDGE BASE
# ============================================================

documents = [

    Document(
        doc_id="doc_01",
        text=(
            "RAG retrieves relevant documents before "
            "a language model generates an answer."
        ),
        source="rag_guide.pdf",
        document_type="pdf",
        category="rag",
        page=5,
        year=2025
    ),

    Document(
        doc_id="doc_02",
        text=(
            "Vector databases store embeddings and "
            "support semantic similarity search."
        ),
        source="vector_database.pdf",
        document_type="pdf",
        category="databases",
        page=12,
        year=2024
    ),

    Document(
        doc_id="doc_03",
        text=(
            "RAG combines retrieval with language "
            "generation to provide grounded answers."
        ),
        source="rag_notes.md",
        document_type="markdown",
        category="rag",
        page=2,
        year=2026
    ),

    Document(
        doc_id="doc_04",
        text=(
            "Machine learning models learn patterns "
            "from historical training data."
        ),
        source="ml_course.pdf",
        document_type="pdf",
        category="machine-learning",
        page=18,
        year=2023
    ),

    Document(
        doc_id="doc_05",
        text=(
            "Semantic search uses vector embeddings "
            "to retrieve documents with similar meaning."
        ),
        source="search.md",
        document_type="markdown",
        category="retrieval",
        page=7,
        year=2026
    ),

    Document(
        doc_id="doc_06",
        text=(
            "Reranking improves retrieval by ordering "
            "candidate documents according to relevance."
        ),
        source="rag_guide.pdf",
        document_type="pdf",
        category="rag",
        page=15,
        year=2025
    ),

    Document(
        doc_id="doc_07",
        text=(
            "Neural networks contain multiple layers "
            "of interconnected computational units."
        ),
        source="deep_learning.pdf",
        document_type="pdf",
        category="deep-learning",
        page=22,
        year=2024
    ),
]


# ============================================================
# FILTER EXPERIMENTS
# ============================================================

def run_experiments(
    store: MetadataVectorStore
):

    query = (
        "How does RAG retrieve information "
        "for generating answers?"
    )

    # --------------------------------------------------------
    # Experiment 1 — No filter
    # --------------------------------------------------------

    results = store.search(
        query=query,
        top_k=5
    )

    display_results(
        results,
        "1. NO METADATA FILTER"
    )

    # --------------------------------------------------------
    # Experiment 2 — Category filter
    # --------------------------------------------------------

    results = store.search(
        query=query,
        top_k=5,
        filters={
            "category": "rag"
        }
    )

    display_results(
        results,
        "2. CATEGORY = RAG"
    )

    # --------------------------------------------------------
    # Experiment 3 — Document type
    # --------------------------------------------------------

    results = store.search(
        query=query,
        top_k=5,
        filters={
            "document_type": "pdf"
        }
    )

    display_results(
        results,
        "3. DOCUMENT TYPE = PDF"
    )

    # --------------------------------------------------------
    # Experiment 4 — Source
    # --------------------------------------------------------

    results = store.search(
        query=query,
        top_k=5,
        filters={
            "source": "rag_guide.pdf"
        }
    )

    display_results(
        results,
        "4. SOURCE = rag_guide.pdf"
    )

    # --------------------------------------------------------
    # Experiment 5 — Year range
    # --------------------------------------------------------

    results = store.search(
        query=query,
        top_k=5,
        filters={
            "year": {
                "min": 2025
            }
        }
    )

    display_results(
        results,
        "5. YEAR >= 2025"
    )

    # --------------------------------------------------------
    # Experiment 6 — Combined filters
    # --------------------------------------------------------

    results = store.search(
        query=query,
        top_k=5,
        filters={
            "category": "rag",
            "document_type": "pdf",
            "year": {
                "min": 2025
            }
        }
    )

    display_results(
        results,
        "6. COMBINED FILTERS"
    )

    # --------------------------------------------------------
    # Experiment 7 — Minimum similarity
    # --------------------------------------------------------

    results = store.search(
        query=query,
        top_k=5,
        min_score=0.45
    )

    display_results(
        results,
        "7. SIMILARITY THRESHOLD >= 0.45"
    )

    # --------------------------------------------------------
    # Experiment 8 — Impossible filter
    # --------------------------------------------------------

    results = store.search(
        query=query,
        top_k=5,
        filters={
            "category": "quantum-computing"
        }
    )

    display_results(
        results,
        "8. NO MATCHING METADATA"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 75)
    print("RAG METADATA FILTERING")
    print("=" * 75)

    print(
        "\nLoading embedding model..."
    )

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # --------------------------------------------------------
    # Create store
    # --------------------------------------------------------

    store = MetadataVectorStore(
        model
    )

    # --------------------------------------------------------
    # Add documents
    # --------------------------------------------------------

    store.add_documents(
        documents
    )

    print(
        f"\nIndexed documents: "
        f"{len(store.documents)}"
    )

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    run_experiments(
        store
    )
