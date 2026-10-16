
"""
RAG Document Versioning

Features:
- Content-hash change detection
- Automatic document version numbers
- Replacement of old chunks
- Version history metadata
- Semantic search over current document versions

This is an in-memory educational implementation.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    version: int
    chunk_index: int
    text: str
    source: str
    content_hash: str
    updated_at: str


class VersionedRAGStore:
    def __init__(self, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)

        # Contains only the current version of each document.
        self.documents = {}

        # Contains chunks from current document versions.
        self.chunks = []

        self.embeddings = np.empty((0, 0))

        # Historical metadata is retained separately.
        self.version_history = {}

    @staticmethod
    def hash_content(text):
        normalized = " ".join(text.split())

        return hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def split_text(text, chunk_size=80, overlap=15):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size"
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

    def _rebuild_index(self):
        if not self.chunks:
            self.embeddings = np.empty((0, 0))
            return

        self.embeddings = self.model.encode(
            [chunk.text for chunk in self.chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def ingest(
        self,
        document_id,
        source,
        text,
        chunk_size=80,
        overlap=15,
    ):
        """Add a document or update its current version."""

        if not document_id.strip():
            raise ValueError("document_id cannot be empty")

        if not text.strip():
            raise ValueError("Document text cannot be empty")

        content_hash = self.hash_content(text)

        existing = self.documents.get(document_id)

        # Skip unchanged content.
        if (
            existing is not None
            and existing["content_hash"] == content_hash
        ):
            return {
                "status": "unchanged",
                "document_id": document_id,
                "version": existing["version"],
                "chunks_created": 0,
            }

        version = (
            existing["version"] + 1
            if existing is not None
            else 1
        )

        timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        # Remove chunks belonging to the previous version.
        self.chunks = [
            chunk
            for chunk in self.chunks
            if chunk.document_id != document_id
        ]

        text_chunks = self.split_text(
            text,
            chunk_size,
            overlap,
        )

        new_chunks = []

        for index, chunk_text in enumerate(text_chunks):
            new_chunks.append(
                DocumentChunk(
                    chunk_id=f"{document_id}-v{version}-{index}",
                    document_id=document_id,
                    version=version,
                    chunk_index=index,
                    text=chunk_text,
                    source=source,
                    content_hash=content_hash,
                    updated_at=timestamp,
                )
            )

        self.chunks.extend(new_chunks)

        self.documents[document_id] = {
            "document_id": document_id,
            "source": source,
            "version": version,
            "content_hash": content_hash,
            "updated_at": timestamp,
            "chunk_count": len(new_chunks),
        }

        self.version_history.setdefault(
            document_id,
            [],
        ).append({
            "version": version,
            "source": source,
            "content_hash": content_hash,
            "updated_at": timestamp,
            "chunk_count": len(new_chunks),
        })

        self._rebuild_index()

        return {
            "status": "created" if version == 1 else "updated",
            "document_id": document_id,
            "version": version,
            "chunks_created": len(new_chunks),
        }

    def delete_document(self, document_id):
        """Remove a document from the current search index."""

        if document_id not in self.documents:
            return False

        del self.documents[document_id]

        self.chunks = [
            chunk
            for chunk in self.chunks
            if chunk.document_id != document_id
        ]

        self._rebuild_index()

        # History remains available for auditing.
        return True

    def search(self, query, top_k=3):
        """Search only chunks from current document versions."""

        if not self.chunks:
            return []

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        scores = self.embeddings @ query_embedding
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in ranked_indices:
            chunk = self.chunks[int(index)]

            results.append({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "version": chunk.version,
                "source": chunk.source,
                "text": chunk.text,
                "similarity": float(scores[index]),
            })

        return results

    def get_version_history(self, document_id):
        """Return recorded version metadata."""

        return self.version_history.get(
            document_id,
            [],
        )

    def save_metadata(self, output_path="versioned_rag_metadata.json"):
        """Export current metadata and version history."""

        payload = {
            "documents": self.documents,
            "version_history": self.version_history,
            "chunks": [
                asdict(chunk)
                for chunk in self.chunks
            ],
        }

        Path(output_path).write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        print(f"Metadata saved to {output_path}")


def demo():
    store = VersionedRAGStore()

    original_text = (
        "The EV charger maintenance manual explains that "
        "technicians must inspect active fault codes, verify "
        "communication status, and follow approved electrical "
        "isolation procedures before servicing equipment."
    )

    updated_text = (
        "The EV charger maintenance manual requires technicians "
        "to inspect active fault codes, verify communication "
        "status, and follow approved electrical isolation "
        "procedures before servicing equipment. The revised "
        "edition also requires recording the repair outcome "
        "and completing the maintenance checklist."
    )

    print("\n1. Ingest original document")

    print(
        store.ingest(
            document_id="charger-manual",
            source="maintenance_manual.txt",
            text=original_text,
            chunk_size=20,
            overlap=4,
        )
    )

    print("\n2. Ingest identical content")

    print(
        store.ingest(
            document_id="charger-manual",
            source="maintenance_manual.txt",
            text=original_text,
            chunk_size=20,
            overlap=4,
        )
    )

    print("\n3. Ingest updated document")

    print(
        store.ingest(
            document_id="charger-manual",
            source="maintenance_manual_v2.txt",
            text=updated_text,
            chunk_size=20,
            overlap=4,
        )
    )

    print("\n4. Current version metadata")

    print(
        json.dumps(
            store.documents["charger-manual"],
            indent=2,
        )
    )

    print("\n5. Version history")

    for version in store.get_version_history(
        "charger-manual"
    ):
        print(version)

    print("\n6. Search current index")

    for result in store.search(
        "What should technicians do during maintenance?",
        top_k=3,
    ):
        print(
            f"\nVersion: {result['version']}"
            f"\nSource: {result['source']}"
            f"\nText: {result['text']}"
        )

    store.save_metadata()


if __name__ == "__main__":
    demo()
