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
    chunk
