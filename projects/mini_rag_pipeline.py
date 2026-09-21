"""
Mini RAG Retrieval Pipeline

Practice:
- Document preparation
- Text chunking
- Sentence embeddings
- Vector retrieval
- Cosine similarity
- Top-K retrieval
- Similarity thresholding
- Context assembly
- Source metadata

This is a retrieval-only RAG pipeline.
The final LLM generation step is intentionally omitted
so the retrieval process can be understood clearly.

Requirement:
    pip install sentence-transformers numpy
"""

import re
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


# =========================================================
# 1. Sample Knowledge Base
# =========================================================

documents = [
    {
        "document_id": "ai_001",
        "title": "Introduction to Artificial Intelligence",
        "source": "ai_intro.txt",
        "text": """
        Artificial intelligence is a field of computer science
        focused on building systems that can perform tasks that
        normally require human intelligence. These tasks include
        reasoning, learning, perception, language understanding,
        and decision making.
        """,
    },
    {
        "document_id": "ml_001",
        "title": "Machine Learning Fundamentals",
        "source": "machine_learning.txt",
        "text": """
        Machine learning is a branch of artificial intelligence
        where systems learn patterns from data. A machine learning
        model can use training examples to learn relationships
        between input features and target outputs. Models can be
        evaluated using data that was not used during training.
        """,
    },
    {
        "document_id": "rag_001",
        "title": "Retrieval Augmented Generation",
        "source": "rag.txt",
        "text": """
        Retrieval augmented generation, commonly called RAG,
        combines information retrieval with language generation.
        A RAG system retrieves relevant information from a
        knowledge base and provides that information as context
        to a language model before generating an answer.
        """,
    },
    {
        "document_id": "embedding_001",
        "title": "Text Embeddings",
        "source": "embeddings.txt",
        "text": """
        Text embeddings represent text as numerical vectors.
        Semantically related pieces of text can have similar
        vector representations. Embeddings are commonly used
        for semantic search, document retrieval, clustering,
        recommendation systems, and RAG applications.
        """,
    },
    {
        "document_id": "vector_001",
        "title": "Vector Search",
        "source": "vector_search.txt",
        "text": """
        Vector search compares a query vector against stored
        document vectors. Similarity metrics such as cosine
        similarity can be used to rank documents according to
        their relevance to a query. Vector databases optimize
        this process for large collections of embeddings.
        """,
    },
]


# =========================================================
# 2. Text Cleaning
# =========================================================

def clean_text(text):
    """
    Normalize whitespace in a document.
    """

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# =========================================================
# 3. Sentence Splitting
# =========================================================

def split_sentences(text):
    """
    Split text into sentences.
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


# =========================================================
# 4. Chunk Data Structure
# =========================================================

@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    source: str
    text: str


# =========================================================
# 5. Create Chunks
# =========================================================

def create_chunks(
    documents,
    max_characters=350,
):
    """
    Convert documents into smaller chunks.

    Sentences are kept together whenever possible.
    """

    chunks = []

    for document in documents:

        cleaned_text = clean_text(
            document["text"]
        )

        sentences = split_sentences(
            cleaned_text
        )

        current_chunk = ""
        chunk_number = 0

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
                    Chunk(
                        chunk_id=(
                            f"{document['document_id']}_"
                            f"chunk_{chunk_number}"
                        ),
                        document_id=document[
                            "document_id"
                        ],
                        title=document["title"],
                        source=document["source"],
                        text=current_chunk,
                    )
                )

                chunk_number += 1

                current_chunk = sentence

        if current_chunk:

            chunks.append(
                Chunk(
                    chunk_id=(
                        f"{document['document_id']}_"
                        f"chunk_{chunk_number}"
                    ),
                    document_id=document[
                        "document_id"
                    ],
                    title=document["title"],
                    source=document["source"],
                    text=current_chunk,
                )
            )

    return chunks


# =========================================================
# 6. Create Knowledge Base Chunks
# =========================================================

chunks = create_chunks(
    documents,
    max_characters=350,
)

print("Total Chunks:")
print(len(chunks))


print("\nKnowledge Base Chunks:")

for chunk in chunks:

    print(
        f"\n[{chunk.chunk_id}] "
        f"{chunk.title}"
    )

    print(
        f"Source: {chunk.source}"
    )

    print(
        f"Text: {chunk.text}"
    )


# =========================================================
# 7. Load Embedding Model
# =========================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# =========================================================
# 8. Generate Chunk Embeddings
# =========================================================

chunk_texts = [
    chunk.text
    for chunk in chunks
]

chunk_embeddings = embedding_model.encode(
    chunk_texts,
    convert_to_numpy=True,
)

print("\nEmbedding Shape:")
print(chunk_embeddings.shape)


# =========================================================
# 9. Normalize Embeddings
# =========================================================

def normalize_vectors(vectors):
    """
    Normalize vectors to unit length.
    """

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(
        norms,
        1e-12,
    )

    return vectors / norms


normalized_embeddings = normalize_vectors(
    chunk_embeddings
)


# =========================================================
# 10. Retrieve Relevant Chunks
# =========================================================

def retrieve(
    query,
    top_k=3,
    threshold=0.0,
):
    """
    Retrieve the most relevant chunks for a query.
    """

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
    )

    query_embedding = normalize_vectors(
        query_embedding
    )[0]

    scores = (
        normalized_embeddings
        @ query_embedding
    )

    ranked_indices = np.argsort(
        scores
    )[::-1]

    results = []

    for index in ranked_indices:

        score = float(
            scores[index]
        )

        if score < threshold:
            continue

        results.append(
            {
                "chunk": chunks[index],
                "score": score,
            }
        )

        if len(results) >= top_k:
            break

    return results


# =========================================================
# 11. Test Retrieval
# =========================================================

query = (
    "How does RAG use retrieved information?"
)

results = retrieve(
    query,
    top_k=3,
)

print("\n" + "=" * 60)
print("RETRIEVAL")
print("=" * 60)

print("\nQuery:")
print(query)

print("\nRetrieved Chunks:")

for rank, result in enumerate(
    results,
    start=1,
):

    chunk = result["chunk"]

    print(
        f"\nRank {rank}"
    )

    print(
        f"Score: {result['score']:.4f}"
    )

    print(
        f"Source: {chunk.source}"
    )

    print(
        f"Text: {chunk.text}"
    )


# =========================================================
# 12. Context Assembly
# =========================================================

def build_context(results):
    """
    Combine retrieved chunks into a context block
    that could later be sent to an LLM.
    """

    context_parts = []

    for rank, result in enumerate(
        results,
        start=1,
    ):

        chunk = result["chunk"]

        context_parts.append(
            f"[Source {rank}: {chunk.source}]\n"
            f"{chunk.text}"
        )

    return "\n\n".join(
        context_parts
    )


context = build_context(
    results
)

print("\n" + "=" * 60)
print("ASSEMBLED CONTEXT")
print("=" * 60)

print(context)


# =========================================================
# 13. Build LLM Prompt
# =========================================================

def build_prompt(
    query,
    context,
):
    """
    Create a grounded prompt for a future LLM.
    """

    return f"""
You are an AI assistant answering questions
using only the provided context.

Context:
{context}

Question:
{query}

Instructions:
- Answer using the provided context.
- Do not invent unsupported information.
- If the context does not contain the answer,
  say that the information is not available.
"""


prompt = build_prompt(
    query,
    context,
)

print("\n" + "=" * 60)
print("LLM PROMPT")
print("=" * 60)

print(prompt)


# =========================================================
# 14. Retrieval with Similarity Threshold
# =========================================================

query = "What is quantum computing?"

threshold_results = retrieve(
    query,
    top_k=3,
    threshold=0.45,
)

print("\n" + "=" * 60)
print("THRESHOLD RETRIEVAL")
print("=" * 60)

print("\nQuery:")
print(query)

if threshold_results:

    for result in threshold_results:

        print(
            f"\nScore: "
            f"{result['score']:.4f}"
        )

        print(
            result["chunk"].text
        )

else:

    print(
        "No sufficiently relevant context found."
    )


# =========================================================
# 15. Retrieval Function for Applications
# =========================================================

def retrieve_context(
    query,
    top_k=3,
    threshold=0.0,
):
    """
    Complete retrieval helper.

    Returns:
        context
        sources
    """

    results = retrieve(
        query,
        top_k=top_k,
        threshold=threshold,
    )

    if not results:
        return "", []

    context = build_context(
        results
    )

    sources = [
        {
            "source": result["chunk"].source,
            "score": result["score"],
        }
        for result in results
    ]

    return context, sources


# =========================================================
# 16. Application-Style Query
# =========================================================

query = (
    "Why are embeddings useful for RAG?"
)

context, sources = retrieve_context(
    query,
    top_k=3,
)

print("\n" + "=" * 60)
print("APPLICATION QUERY")
print("=" * 60)

print("\nQuestion:")
print(query)

print("\nContext:")
print(context)

print("\nSources:")

for source in sources:

    print(
        f"- {source['source']} "
        f"(score={source['score']:.4f})"
    )


# =========================================================
# 17. Complete RAG Architecture
# =========================================================

print(
    """
\nComplete RAG Architecture:

                DOCUMENTS
                    │
                    ▼
               Text Cleaning
                    │
                    ▼
                 Chunking
                    │
                    ▼
              Embeddings
                    │
                    ▼
              Vector Store
                    │
                    │
User Question ──────┤
                    ▼
            Query Embedding
                    │
                    ▼
           Similarity Search
                    │
                    ▼
               Top-K Chunks
                    │
                    ▼
            Context Assembly
                    │
                    ▼
               LLM Prompt
                    │
                    ▼
               LLM Response
"""
)


# =========================================================
# 18. Important RAG Concepts
# =========================================================

print("\nImportant Concepts:")

print(
    "1. Chunking determines the units retrieved."
)

print(
    "2. Embeddings convert chunks into searchable vectors."
)

print(
    "3. Retrieval selects relevant context."
)

print(
    "4. Top-K controls how much context is retrieved."
)

print(
    "5. Similarity thresholds can reject weak matches."
)

print(
    "6. Metadata allows sources to be shown with answers."
)

print(
    "7. The retrieved context can be passed to an LLM."
)

print(
    "8. RAG quality depends heavily on retrieval quality."
)
