"""
RAG Document Deduplication

Detects:
1. Exact duplicate chunks
2. Near-duplicate chunks using cosine similarity

Pipeline:
Chunks → Exact Deduplication → Similarity Check → Clean Chunks
"""

from dataclasses import dataclass
import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

SIMILARITY_THRESHOLD = 0.90


# ============================================================
# Data Model
# ============================================================

@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source: str
    page: int | None = None


# ============================================================
# Exact Duplicate Detection
# ============================================================

def normalize_text(text):
    """Normalize text for duplicate comparison."""

    return " ".join(text.lower().split())


def text_hash(text):
    """Generate a stable hash for text."""

    normalized = normalize_text(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def remove_exact_duplicates(chunks):
    """
    Remove chunks containing exactly the same text.

    Returns:
        unique_chunks
        duplicate_chunks
    """

    seen_hashes = set()

    unique_chunks = []
    duplicate_chunks = []

    for chunk in chunks:

        current_hash = text_hash(chunk.text)

        if current_hash in seen_hashes:

            duplicate_chunks.append(chunk)

        else:

            seen_hashes.add(current_hash)
            unique_chunks.append(chunk)

    return unique_chunks, duplicate_chunks


# ============================================================
# Embedding Generation
# ============================================================

def generate_embeddings(chunks, model):
    """Generate normalized embeddings."""

    texts = [
        chunk.text
        for chunk in chunks
    ]

    if not texts:
        return np.empty((0, 384))

    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )


# ============================================================
# Cosine Similarity
# ============================================================

def cosine_similarity_matrix(embeddings):
    """
    Calculate pairwise cosine similarity.

    Embeddings are already normalized.
    Therefore:

        cosine similarity = dot product
    """

    return embeddings @ embeddings.T


# ============================================================
# Near Duplicate Detection
# ============================================================

def remove_near_duplicates(
    chunks,
    embeddings,
    threshold=SIMILARITY_THRESHOLD
):
    """
    Remove chunks that are highly similar to
    an earlier chunk.
    """

    if len(chunks) <= 1:
        return chunks, []

    similarities = cosine_similarity_matrix(
        embeddings
    )

    keep_indices = []
    duplicate_indices = []

    for i in range(len(chunks)):

        is_duplicate = False

        for kept_index in keep_indices:

            similarity = similarities[
                i,
                kept_index
            ]

            if similarity >= threshold:

                is_duplicate = True

                duplicate_indices.append(i)

                print(
                    f"Near duplicate detected: "
                    f"{chunks[i].chunk_id} "
                    f"≈ "
                    f"{chunks[kept_index].chunk_id} "
                    f"({similarity:.3f})"
                )

                break

        if not is_duplicate:
            keep_indices.append(i)

    unique_chunks = [
        chunks[i]
        for i in keep_indices
    ]

    duplicate_chunks = [
        chunks[i]
        for i in duplicate_indices
    ]

    return unique_chunks, duplicate_chunks


# ============================================================
# Complete Deduplication Pipeline
# ============================================================

def deduplicate_chunks(
    chunks,
    similarity_threshold=SIMILARITY_THRESHOLD
):
    """
    Run exact + semantic deduplication.
    """

    print("=" * 60)
    print("RAG DOCUMENT DEDUPLICATION")
    print("=" * 60)

    print(
        f"\nOriginal chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # Step 1: Exact duplicates
    # --------------------------------------------------------

    unique_chunks, exact_duplicates = (
        remove_exact_duplicates(chunks)
    )

    print(
        f"After exact deduplication: "
        f"{len(unique_chunks)}"
    )

    print(
        f"Exact duplicates removed: "
        f"{len(exact_duplicates)}"
    )

    if not unique_chunks:
        return [], exact_duplicates

    # --------------------------------------------------------
    # Step 2: Semantic duplicates
    # --------------------------------------------------------

    model = SentenceTransformer(
        MODEL_NAME
    )

    embeddings = generate_embeddings(
        unique_chunks,
        model
    )

    final_chunks, near_duplicates = (
        remove_near_duplicates(
            unique_chunks,
            embeddings,
            similarity_threshold
        )
    )

    print(
        f"\nAfter semantic deduplication: "
        f"{len(final_chunks)}"
    )

    print(
        f"Near duplicates removed: "
        f"{len(near_duplicates)}"
    )

    print(
        f"\nFinal chunks: "
        f"{len(final_chunks)}"
    )

    return (
        final_chunks,
        exact_duplicates + near_duplicates
    )


# ============================================================
# Statistics
# ============================================================

def show_statistics(
    original,
    final,
    removed
):
    """Display deduplication statistics."""

    original_count = len(original)
    final_count = len(final)
    removed_count = len(removed)

    reduction = (
        removed_count / original_count * 100
        if original_count
        else 0
    )

    print("\n" + "=" * 60)
    print("DEDUPLICATION STATISTICS")
    print("=" * 60)

    print(f"Original chunks : {original_count}")
    print(f"Removed chunks  : {removed_count}")
    print(f"Final chunks    : {final_count}")
    print(f"Reduction       : {reduction:.2f}%")


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":

    chunks = [

        DocumentChunk(
            chunk_id="chunk-1",
            text=(
                "Machine learning is a branch of "
                "artificial intelligence that learns "
                "patterns from data."
            ),
            source="ml.txt",
            page=1
        ),

        # Exact duplicate
        DocumentChunk(
            chunk_id="chunk-2",
            text=(
                "Machine learning is a branch of "
                "artificial intelligence that learns "
                "patterns from data."
            ),
            source="ml.txt",
            page=2
        ),

        # Near duplicate
        DocumentChunk(
            chunk_id="chunk-3",
            text=(
                "Machine learning is a field of "
                "artificial intelligence that learns "
                "patterns from available data."
            ),
            source="ml.pdf",
            page=5
        ),

        DocumentChunk(
            chunk_id="chunk-4",
            text=(
                "Supervised learning uses labelled "
                "datasets to train machine learning models."
            ),
            source="ml.txt",
            page=3
        ),

        DocumentChunk(
            chunk_id="chunk-5",
            text=(
                "Unsupervised learning discovers hidden "
                "patterns in unlabeled datasets."
            ),
            source="ml.txt",
            page=4
        ),
    ]

    final_chunks, removed_chunks = (
        deduplicate_chunks(
            chunks,
            similarity_threshold=0.85
        )
    )

    show_statistics(
        original=chunks,
        final=final_chunks,
        removed=removed_chunks
    )

    print("\nFinal chunks:")

    for chunk in final_chunks:

        print(
            f"\n[{chunk.chunk_id}] "
            f"{chunk.text}"
        )
