"""
RAG Document Ingestion Pipeline

Pipeline:
File → Text Extraction → Chunking → Metadata → Embeddings

Supported:
- .txt
- .md
- .pdf
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import re
import json

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# ============================================================
# Data Model
# ============================================================

@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source: str
    document_type: str
    chunk_index: int
    page: int | None = None


# ============================================================
# Text Extraction
# ============================================================

def clean_text(text):
    """Clean unnecessary whitespace from extracted text."""

    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_txt(path):
    """Extract text from a TXT file."""

    return path.read_text(
        encoding="utf-8",
        errors="ignore"
    )


def extract_markdown(path):
    """Extract readable text from a Markdown file."""

    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    # Remove code fences
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)

    # Remove markdown links but keep link text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove heading markers
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)

    return text


def extract_pdf(path):
    """
    Extract text from a PDF.

    Returns:
        list[tuple[int, str]]
        Each tuple contains (page_number, page_text).
    """

    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            "PDF support requires pypdf. "
            "Install it using: pip install pypdf"
        )

    reader = PdfReader(str(path))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append((page_number, text))

    return pages


# ============================================================
# Chunking
# ============================================================

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping word-based chunks.
    """

    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):

        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


# ============================================================
# File Ingestion
# ============================================================

def ingest_file(
    file_path,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    """
    Ingest a single document.

    Returns:
        list[DocumentChunk]
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    document_type = extension.replace(".", "")

    chunks = []

    # --------------------------------------------------------
    # TXT / Markdown
    # --------------------------------------------------------

    if extension in {".txt", ".md"}:

        if extension == ".txt":
            text = extract_txt(path)
        else:
            text = extract_markdown(path)

        text = clean_text(text)

        text_chunks = chunk_text(
            text,
            chunk_size,
            overlap
        )

        for index, chunk in enumerate(text_chunks):

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{path.stem}-{index}",
                    text=chunk,
                    source=path.name,
                    document_type=document_type,
                    chunk_index=index
                )
            )

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    elif extension == ".pdf":

        pages = extract_pdf(path)

        global_chunk_index = 0

        for page_number, page_text in pages:

            page_text = clean_text(page_text)

            page_chunks = chunk_text(
                page_text,
                chunk_size,
                overlap
            )

            for chunk in page_chunks:

                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{path.stem}-{global_chunk_index}",
                        text=chunk,
                        source=path.name,
                        document_type="pdf",
                        chunk_index=global_chunk_index,
                        page=page_number
                    )
                )

                global_chunk_index += 1

    return chunks


# ============================================================
# Directory Ingestion
# ============================================================

def ingest_directory(
    directory,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    """
    Ingest all supported documents from a directory.
    """

    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    all_chunks = []

    for file_path in sorted(directory.iterdir()):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:

            file_chunks = ingest_file(
                file_path,
                chunk_size,
                overlap
            )

            all_chunks.extend(file_chunks)

            print(
                f"✓ {file_path.name}: "
                f"{len(file_chunks)} chunks"
            )

        except Exception as error:

            print(
                f"✗ {file_path.name}: {error}"
            )

    return all_chunks


# ============================================================
# Embedding Generation
# ============================================================

def generate_embeddings(
    chunks,
    model_name=EMBEDDING_MODEL
):
    """
    Generate embeddings for document chunks.
    """

    if not chunks:
        return np.empty((0, 384))

    model = SentenceTransformer(model_name)

    texts = [
        chunk.text
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return np.asarray(embeddings)


# ============================================================
# Save Ingestion Data
# ============================================================

def save_ingestion(
    chunks,
    embeddings,
    output_directory="ingestion_output"
):
    """
    Save chunks, metadata and embeddings.
    """

    output = Path(output_directory)

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save embeddings
    np.save(
        output / "embeddings.npy",
        embeddings
    )

    # Save metadata
    metadata = [
        asdict(chunk)
        for chunk in chunks
    ]

    with open(
        output / "metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\nSaved:")
    print(f"- {output / 'embeddings.npy'}")
    print(f"- {output / 'metadata.json'}")


# ============================================================
# Pipeline
# ============================================================

def run_ingestion(input_directory):

    print("=" * 60)
    print("RAG DOCUMENT INGESTION")
    print("=" * 60)

    # 1. Extract + chunk + metadata
    chunks = ingest_directory(input_directory)

    print(
        f"\nTotal chunks created: {len(chunks)}"
    )

    if not chunks:
        print("No supported documents found.")
        return

    # 2. Generate embeddings
    print("\nGenerating embeddings...")

    embeddings = generate_embeddings(chunks)

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # 3. Save everything
    save_ingestion(
        chunks,
        embeddings
    )

    # 4. Preview
    print("\nSample chunks:")

    for chunk in chunks[:3]:

        print("\n" + "-" * 50)
        print(f"ID:       {chunk.chunk_id}")
        print(f"Source:   {chunk.source}")
        print(f"Type:     {chunk.document_type}")
        print(f"Page:     {chunk.page}")
        print(f"Chunk:    {chunk.chunk_index}")
        print(f"Text:     {chunk.text[:200]}...")


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":

    # Create demo directory
    demo_directory = Path("sample_documents")

    demo_directory.mkdir(
        exist_ok=True
    )

    # Create sample TXT document
    sample_file = (
        demo_directory / "machine_learning.txt"
    )

    if not sample_file.exists():

        sample_file.write_text(
            """
            Machine learning is a branch of artificial
            intelligence that enables computers to learn
            patterns from data.

            Supervised learning uses labelled datasets.
            Unsupervised learning discovers hidden patterns.
            Reinforcement learning learns through rewards
            and penalties.

            Common machine learning algorithms include
            linear regression, logistic regression,
            decision trees, random forests and neural networks.
            """,
            encoding="utf-8"
        )

    # Run pipeline
    run_ingestion(
        demo_directory
    )
