"""
LLM RAG Pipeline

Practice:
- Document chunking
- Text embeddings
- Vector retrieval
- Context assembly
- Prompt construction
- LLM generation
- Source attribution
- Basic RAG evaluation

Requirements:
    pip install sentence-transformers numpy openai

Environment:
    Set your LLM API key before running.

Example:
    export LLM_API_KEY="your-api-key"
"""

import os
import re
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# =========================================================
# 1. Configuration
# =========================================================

API_KEY = os.getenv("LLM_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "LLM_API_KEY environment variable is not set."
    )


# OpenAI-compatible client.
#
# For another provider, the base_url and model can be
# changed without changing the RAG retrieval architecture.

client = OpenAI(
    api_key=API_KEY,
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gpt-4o-mini",
)


# =========================================================
# 2. Knowledge Base
# =========================================================

documents = [
    {
        "document_id": "ai_001",
        "title": "Artificial Intelligence",
        "source": "ai.txt",
        "text": """
        Artificial intelligence is a field of computer science
        focused on creating systems capable of performing tasks
        that normally require human intelligence. These tasks
        include reasoning, learning, perception, language
        understanding, and decision making.
        """,
    },
    {
        "document_id": "ml_001",
        "title": "Machine Learning",
        "source": "machine_learning.txt",
        "text": """
        Machine learning is a branch of artificial intelligence
        where systems learn patterns from data. Models are trained
        using examples and can then make predictions on previously
        unseen data.
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
        to a language model.
        """,
    },
    {
        "document_id": "embedding_001",
        "title": "Text Embeddings",
        "source": "embeddings.txt",
        "text": """
        Text embeddings represent text as numerical vectors.
        Semantically related text can have similar vector
        representations. Embeddings are useful for semantic
        search, recommendation systems, clustering, and
        retrieval augmented generation.
        """,
    },
    {
        "document_id": "vector_001",
        "title": "Vector Search",
        "source": "vector_search.txt",
        "text": """
        Vector search compares a query embedding against
        stored document embeddings. Similarity metrics such
        as cosine similarity can rank documents according
        to their relevance to a query.
        """,
    },
]


# =========================================================
# 3. Chunk Structure
# =========================================================

@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    source: str
    text: str


# =========================================================
# 4. Text Cleaning
# =========================================================

def clean_text(text):
    """
    Normalize whitespace.
    """

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


# =========================================================
# 5. Sentence Splitting
# =========================================================

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


# =========================================================
# 6. Create Chunks
# =========================================================

def create_chunks(
    documents,
    max_characters=400,
):
    """
    Convert documents into sentence-aware chunks.
    """

    chunks = []

    for document in documents:

        text = clean_text(
            document["text"]
        )

        sentences = split_sentences(
            text
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
                            f"{document['document_id']}"
                            f"_chunk_{chunk_number}"
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
                        f"{document['document_id']}"
                        f"_chunk_{chunk_number}"
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
# 7. Prepare Chunks
# =========================================================

chunks = create_chunks(
    documents
)

print(
    f"Created {len(chunks)} chunks."
)


# =========================================================
# 8. Load Embedding Model
# =========================================================

print(
    "Loading embedding model..."
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print(
    "Embedding model loaded."
)


# =========================================================
# 9. Generate Chunk Embeddings
# =========================================================

chunk_texts = [
    chunk.text
    for chunk in chunks
]

chunk_embeddings = embedding_model.encode(
    chunk_texts,
    convert_to_numpy=True,
)


# =========================================================
# 10. Normalize Vectors
# =========================================================

def normalize_vectors(vectors):
    """
    Normalize vectors for cosine similarity.
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


normalized_chunk_embeddings = (
    normalize_vectors(
        chunk_embeddings
    )
)


# =========================================================
# 11. Retrieve Relevant Chunks
# =========================================================

def retrieve(
    query,
    top_k=3,
):
    """
    Retrieve the most relevant chunks.
    """

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
    )

    query_embedding = normalize_vectors(
        query_embedding
    )[0]

    scores = (
        normalized_chunk_embeddings
        @ query_embedding
    )

    ranked_indices = np.argsort(
        scores
    )[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        results.append(
            {
                "chunk": chunks[index],
                "score": float(
                    scores[index]
                ),
            }
        )

    return results


# =========================================================
# 12. Build Context
# =========================================================

def build_context(results):
    """
    Build context for the LLM from retrieved chunks.
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


# =========================================================
# 13. Build RAG Prompt
# =========================================================

def build_prompt(
    question,
    context,
):
    """
    Create a grounded RAG prompt.
    """

    return f"""
You are a helpful AI assistant.

Answer the user's question using ONLY
the provided context.

If the context does not contain enough
information to answer the question,
clearly say that the information is
not available in the provided documents.

Do not invent facts.

Context:
----------------
{context}
----------------

Question:
{question}

Answer:
"""


# =========================================================
# 14. Generate LLM Answer
# =========================================================

def generate_answer(
    question,
    context,
):
    """
    Send the RAG prompt to the LLM.
    """

    prompt = build_prompt(
        question,
        context,
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer questions using "
                    "only the supplied context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content


# =========================================================
# 15. Complete RAG Function
# =========================================================

def ask_rag(
    question,
    top_k=3,
):
    """
    Complete RAG workflow.

    1. Retrieve
    2. Build context
    3. Generate answer
    4. Return sources
    """

    results = retrieve(
        question,
        top_k=top_k,
    )

    context = build_context(
        results
    )

    answer = generate_answer(
        question,
        context,
    )

    sources = [
        {
            "source": result["chunk"].source,
            "score": result["score"],
        }
        for result in results
    ]

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }


# =========================================================
# 16. Ask a Question
# =========================================================

question = (
    "What is retrieval augmented generation?"
)

result = ask_rag(
    question,
    top_k=3,
)

print("\n" + "=" * 60)
print("RAG QUESTION")
print("=" * 60)

print(
    f"\nQuestion:\n{result['question']}"
)

print(
    f"\nAnswer:\n{result['answer']}"
)


# =========================================================
# 17. Display Sources
# =========================================================

print("\nSources:")

for source in result["sources"]:

    print(
        f"- {source['source']} "
        f"(similarity={source['score']:.4f})"
    )


# =========================================================
# 18. Test Multiple Questions
# =========================================================

questions = [
    "What are text embeddings used for?",
    "How does vector search work?",
    "What is machine learning?",
]


print("\n" + "=" * 60)
print("MULTIPLE RAG QUESTIONS")
print("=" * 60)

for question in questions:

    result = ask_rag(
        question,
        top_k=2,
    )

    print(
        f"\nQ: {result['question']}"
    )

    print(
        f"A: {result['answer']}"
    )

    print("\nSources:")

    for source in result["sources"]:

        print(
            f"  - {source['source']} "
            f"({source['score']:.4f})"
        )


# =========================================================
# 19. Test an Unanswerable Question
# =========================================================

question = (
    "Who invented the first practical airplane?"
)

result = ask_rag(
    question,
    top_k=2,
)

print("\n" + "=" * 60)
print("UNANSWERABLE QUESTION TEST")
print("=" * 60)

print(
    f"\nQuestion:\n{result['question']}"
)

print(
    f"\nAnswer:\n{result['answer']}"
)


# =========================================================
# 20. Complete RAG Architecture
# =========================================================

print(
    """
\nComplete RAG Pipeline:

Documents
    ↓
Text Cleaning
    ↓
Chunking
    ↓
Embedding Model
    ↓
Vector Store
    ↓
        User Question
              ↓
        Query Embedding
              ↓
        Similarity Search
              ↓
           Top-K Chunks
              ↓
        Context Assembly
              ↓
          RAG Prompt
              ↓
             LLM
              ↓
        Grounded Answer
              ↓
          Source List
"""
)


# =========================================================
# 21. Important RAG Principles
# =========================================================

print("\nImportant RAG Principles:")

print(
    "1. Retrieval determines what information reaches the LLM."
)

print(
    "2. Better retrieval generally provides better context."
)

print(
    "3. The prompt should instruct the LLM to stay grounded."
)

print(
    "4. The system should handle questions outside its knowledge base."
)

print(
    "5. Sources make retrieved information traceable."
)

print(
    "6. Temperature=0 is useful for deterministic factual RAG answers."
)

print(
    "7. Production systems should use a persistent vector database."
)
