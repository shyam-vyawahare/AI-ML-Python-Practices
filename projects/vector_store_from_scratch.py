"""
Text Chunking for AI / RAG

Practice:
- Fixed-size chunking
- Overlapping chunks
- Sentence-aware chunking
- Chunk metadata
- Token estimation
- Chunk size experiments
- Preparing text for embedding/retrieval

No external dependencies required.
"""

import re
from dataclasses import dataclass


# ---------------------------------------------------------
# 1. Sample Document
# ---------------------------------------------------------

document = """
Artificial intelligence is transforming the way software
applications are built. Machine learning allows systems to
learn patterns from data and make predictions.

Deep learning uses neural networks with multiple layers.
These networks can process images, text, audio, and other
types of complex information.

Natural language processing focuses on understanding and
generating human language. Modern NLP systems commonly use
transformer-based architectures and large language models.

Retrieval augmented generation combines information
retrieval with language generation. A RAG system first
retrieves relevant information from a knowledge base and
then provides that information to a language model.

Good document chunking is important for retrieval systems.
Chunks that are too small may lose context, while chunks
that are too large may contain too much unrelated
information.

Chunk overlap can preserve context between neighboring
chunks. This is particularly useful when an important
sentence or concept appears near a chunk boundary.
"""


# ---------------------------------------------------------
# 2. Basic Text Cleaning
# ---------------------------------------------------------

def clean_text(text):
    """
    Normalize whitespace and remove unnecessary spaces.
    """

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


cleaned_document = clean_text(document)

print("Cleaned Document:")
print(cleaned_document)


# ---------------------------------------------------------
# 3. Fixed-Size Chunking
# ---------------------------------------------------------

def fixed_size_chunks(
    text,
    chunk_size=200,
):
    """
    Split text into fixed character-size chunks.
    """

    chunks = []

    for start in range(
        0,
        len(text),
        chunk_size,
    ):
        chunk = text[
            start:start + chunk_size
        ]

        chunks.append(chunk)

    return chunks


chunks = fixed_size_chunks(
    cleaned_document,
    chunk_size=200,
)

print("\nFixed-Size Chunks:")

for index, chunk in enumerate(
    chunks,
    start=1,
):
    print(
        f"\nChunk {index}:"
    )
    print(chunk)


# ---------------------------------------------------------
# 4. Overlapping Chunking
# ---------------------------------------------------------

def overlapping_chunks(
    text,
    chunk_size=200,
    overlap=50,
):
    """
    Split text into overlapping chunks.

    Example:

    Chunk 1: 0   -> 200
    Chunk 2: 150 -> 350
    Chunk 3: 300 -> 500
    """

    if overlap >= chunk_size:
        raise ValueError(
            "Overlap must be smaller than chunk size."
        )

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(
            text[start:end]
        )

        start += chunk_size - overlap

    return chunks


overlap_chunks = overlapping_chunks(
    cleaned_document,
    chunk_size=200,
    overlap=50,
)

print("\nOverlapping Chunks:")

for index, chunk in enumerate(
    overlap_chunks,
    start=1,
):
    print(
        f"\nChunk {index}:"
    )
    print(chunk)


# ---------------------------------------------------------
# 5. Sentence Splitting
# ---------------------------------------------------------

def split_sentences(text):
    """
    Basic sentence splitter.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


sentences = split_sentences(
    cleaned_document
)

print("\nSentences:")

for index, sentence in enumerate(
    sentences,
    start=1,
):
    print(
        f"{index}. {sentence}"
    )


# ---------------------------------------------------------
# 6. Sentence-Aware Chunking
# ---------------------------------------------------------

def sentence_chunks(
    text,
    max_characters=250,
):
    """
    Build chunks using complete sentences while
    respecting a maximum character size.
    """

    sentences = split_sentences(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        if not current_chunk:

            current_chunk = sentence

        elif (
            len(current_chunk)
            + 1
            + len(sentence)
            <= max_characters
        ):

            current_chunk += (
                " " + sentence
            )

        else:

            chunks.append(
                current_chunk
            )

            current_chunk = sentence

    if current_chunk:
        chunks.append(
            current_chunk
        )

    return chunks


sentence_based_chunks = sentence_chunks(
    cleaned_document,
    max_characters=250,
)

print("\nSentence-Aware Chunks:")

for index, chunk in enumerate(
    sentence_based_chunks,
    start=1,
):
    print(
        f"\nChunk {index}:"
    )
    print(chunk)


# ---------------------------------------------------------
# 7. Chunk Metadata
# ---------------------------------------------------------

@dataclass
class TextChunk:
    chunk_id: int
    text: str
    start: int
    end: int
    character_count: int
    estimated_tokens: int


def estimate_tokens(text):
    """
    Rough token estimate.

    This is NOT a tokenizer.
    A simple approximation is used for practice.
    """

    words = text.split()

    return max(
        1,
        int(len(words) * 1.3),
    )


def create_chunks_with_metadata(
    text,
    chunk_size=200,
    overlap=50,
):
    """
    Create chunks with useful metadata.
    """

    if overlap >= chunk_size:
        raise ValueError(
            "Overlap must be smaller than chunk size."
        )

    chunks = []

    start = 0
    chunk_id = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text),
        )

        chunk_text = text[start:end]

        chunk = TextChunk(
            chunk_id=chunk_id,
            text=chunk_text,
            start=start,
            end=end,
            character_count=len(
                chunk_text
            ),
            estimated_tokens=estimate_tokens(
                chunk_text
            ),
        )

        chunks.append(chunk)

        chunk_id += 1

        start += chunk_size - overlap

    return chunks


metadata_chunks = create_chunks_with_metadata(
    cleaned_document,
    chunk_size=200,
    overlap=50,
)

print("\nChunks with Metadata:")

for chunk in metadata_chunks:

    print(
        f"\nID: {chunk.chunk_id}"
    )

    print(
        f"Position: {chunk.start}"
        f" -> {chunk.end}"
    )

    print(
        f"Characters: "
        f"{chunk.character_count}"
    )

    print(
        f"Estimated Tokens: "
        f"{chunk.estimated_tokens}"
    )

    print(
        f"Text: {chunk.text}"
    )


# ---------------------------------------------------------
# 8. Chunk Size Experiment
# ---------------------------------------------------------

chunk_sizes = [
    100,
    200,
    300,
    500,
]

print("\nChunk Size Experiment:")

for size in chunk_sizes:

    chunks = fixed_size_chunks(
        cleaned_document,
        chunk_size=size,
    )

    print(
        f"Chunk size {size}: "
        f"{len(chunks)} chunks"
    )


# ---------------------------------------------------------
# 9. Overlap Experiment
# ---------------------------------------------------------

overlap_values = [
    0,
    25,
    50,
    75,
]

print("\nOverlap Experiment:")

for overlap in overlap_values:

    chunks = overlapping_chunks(
        cleaned_document,
        chunk_size=200,
        overlap=overlap,
    )

    print(
        f"Overlap {overlap}: "
        f"{len(chunks)} chunks"
    )


# ---------------------------------------------------------
# 10. Find a Keyword Across Chunks
# ---------------------------------------------------------

keyword = "retrieval"

print(
    f"\nSearching chunks for: '{keyword}'"
)

for chunk in metadata_chunks:

    if keyword.lower() in chunk.text.lower():

        print(
            f"\nFound in Chunk "
            f"{chunk.chunk_id}"
        )

        print(chunk.text)


# ---------------------------------------------------------
# 11. Context Preservation Example
# ---------------------------------------------------------

context_text = (
    "A RAG system retrieves relevant information "
    "from a knowledge base. The retrieved context "
    "is then provided to a language model. "
    "The language model uses this context to "
    "generate a grounded response."
)

context_chunks = overlapping_chunks(
    context_text,
    chunk_size=100,
    overlap=30,
)

print("\nContext Preservation Example:")

for index, chunk in enumerate(
    context_chunks,
    start=1,
):

    print(
        f"\nChunk {index}:"
    )

    print(chunk)


# ---------------------------------------------------------
# 12. Simulated RAG Preparation
# ---------------------------------------------------------

rag_chunks = create_chunks_with_metadata(
    cleaned_document,
    chunk_size=300,
    overlap=75,
)

print("\nSimulated RAG Preparation:")

for chunk in rag_chunks:

    print(
        f"Chunk ID: {chunk.chunk_id} | "
        f"Tokens ≈ {chunk.estimated_tokens}"
    )


# ---------------------------------------------------------
# 13. Important Chunking Rules
# ---------------------------------------------------------

print(
    """
\nImportant Chunking Rules:

1. Very small chunks can lose context.
2. Very large chunks can contain unrelated information.
3. Overlap helps preserve boundary context.
4. Sentence-aware chunking can improve readability.
5. Chunk size should match the retrieval task.
6. Store metadata with every chunk.
7. Keep document/page/source information when available.
8. Token counts should ideally use the target model's tokenizer.
"""
)


# ---------------------------------------------------------
# 14. Complete RAG Preprocessing Flow
# ---------------------------------------------------------

print(
    """
\nRAG Preprocessing:

Raw Document
      ↓
Text Extraction
      ↓
Cleaning
      ↓
Chunking
      ↓
Chunk Metadata
      ↓
Embedding Generation
      ↓
Vector Database
      ↓
Semantic Retrieval
      ↓
LLM
"""
)
