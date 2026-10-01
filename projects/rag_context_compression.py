"""
RAG Context Compression Practice

Reduces retrieved context before sending it to an LLM.

Pipeline:

Query
  ↓
Retrieved Chunks
  ↓
Sentence Extraction
  ↓
Relevance Scoring
  ↓
Sentence Ranking
  ↓
Top Relevant Sentences
  ↓
Compressed Context

This implementation uses lightweight lexical relevance
instead of an external LLM or reranker.
"""

import re
from dataclasses import dataclass
from typing import List


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class SentenceScore:
    sentence: str
    score: float
    matched_terms: List[str]


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "with",
    "that",
    "this",
    "it",
    "as",
    "by",
    "from",
    "can",
    "be",
    "used",
    "uses",
    "using",
    "into",
    "its",
    "how",
    "what",
    "why",
    "when",
    "where",
}


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text: str) -> List[str]:
    """
    Convert text into normalized tokens.
    """

    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


def meaningful_tokens(text: str) -> List[str]:
    """
    Remove common stopwords.
    """

    return [
        token
        for token in tokenize(text)
        if token not in STOPWORDS
    ]


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(
    text: str
) -> List[str]:
    """
    Split text into individual sentences.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# QUERY RELEVANCE
# ============================================================

def calculate_relevance(
    query: str,
    sentence: str
) -> SentenceScore:
    """
    Calculate lexical relevance between a query
    and a sentence.

    Score:

        matched query terms
        -------------------
        total query terms
    """

    query_terms = set(
        meaningful_tokens(query)
    )

    sentence_terms = set(
        meaningful_tokens(sentence)
    )

    if not query_terms:
        return SentenceScore(
            sentence=sentence,
            score=0.0,
            matched_terms=[]
        )

    matched_terms = sorted(
        query_terms.intersection(
            sentence_terms
        )
    )

    score = (
        len(matched_terms)
        /
        len(query_terms)
    )

    return SentenceScore(
        sentence=sentence,
        score=score,
        matched_terms=matched_terms
    )


# ============================================================
# SCORE ALL SENTENCES
# ============================================================

def score_sentences(
    query: str,
    chunks: List[str]
) -> List[SentenceScore]:
    """
    Score every sentence across all retrieved chunks.
    """

    scored_sentences = []

    for chunk in chunks:

        sentences = split_sentences(
            chunk
        )

        for sentence in sentences:

            result = calculate_relevance(
                query,
                sentence
            )

            scored_sentences.append(
                result
            )

    return scored_sentences


# ============================================================
# RANK SENTENCES
# ============================================================

def rank_sentences(
    scored_sentences: List[SentenceScore]
) -> List[SentenceScore]:
    """
    Rank sentences by relevance.
    """

    return sorted(
        scored_sentences,
        key=lambda item: item.score,
        reverse=True
    )


# ============================================================
# COMPRESS CONTEXT
# ============================================================

def compress_context(
    query: str,
    chunks: List[str],
    top_n: int = 5,
    min_score: float = 0.2
) -> List[SentenceScore]:
    """
    Select the most relevant sentences.

    Parameters:
        top_n:
            Maximum number of sentences.

        min_score:
            Minimum relevance score.
    """

    scored = score_sentences(
        query,
        chunks
    )

    ranked = rank_sentences(
        scored
    )

    selected = [
        sentence
        for sentence in ranked
        if sentence.score >= min_score
    ]

    return selected[:top_n]


# ============================================================
# BUILD FINAL CONTEXT
# ============================================================

def build_compressed_context(
    selected_sentences: List[SentenceScore]
) -> str:
    """
    Combine selected sentences into final context.
    """

    return " ".join(
        item.sentence
        for item in selected_sentences
    )


# ============================================================
# CONTEXT STATISTICS
# ============================================================

def context_statistics(
    text: str
):
    """
    Calculate basic context statistics.
    """

    characters = len(text)

    words = len(
        tokenize(text)
    )

    sentences = len(
        split_sentences(text)
    )

    return {
        "characters": characters,
        "words": words,
        "sentences": sentences,
    }


# ============================================================
# COMPRESSION RATIO
# ============================================================

def compression_ratio(
    original: str,
    compressed: str
) -> float:
    """
    Calculate percentage of original text retained.
    """

    original_words = len(
        tokenize(original)
    )

    compressed_words = len(
        tokenize(compressed)
    )

    if original_words == 0:
        return 0.0

    return (
        compressed_words
        /
        original_words
    )


# ============================================================
# DISPLAY
# ============================================================

def display_sentence_scores(
    results: List[SentenceScore]
):
    """
    Display sentence relevance scores.
    """

    print("\n")
    print("=" * 75)
    print("SENTENCE RELEVANCE")
    print("=" * 75)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRank {rank}"
        )

        print(
            f"Score: {result.score:.3f}"
        )

        print(
            f"Matched Terms: "
            f"{', '.join(result.matched_terms)}"
        )

        print(
            f"Sentence: "
            f"{result.sentence}"
        )


# ============================================================
# DISPLAY STATISTICS
# ============================================================

def display_statistics(
    original: str,
    compressed: str
):
    """
    Display before/after context statistics.
    """

    original_stats = context_statistics(
        original
    )

    compressed_stats = context_statistics(
        compressed
    )

    ratio = compression_ratio(
        original,
        compressed
    )

    print("\n")
    print("=" * 75)
    print("CONTEXT COMPRESSION STATISTICS")
    print("=" * 75)

    print("\nOriginal Context")

    for key, value in original_stats.items():

        print(
            f"{key}: {value}"
        )

    print("\nCompressed Context")

    for key, value in compressed_stats.items():

        print(
            f"{key}: {value}"
        )

    print(
        f"\nContext Retained: "
        f"{ratio * 100:.2f}%"
    )

    print(
        f"Context Removed: "
        f"{(1 - ratio) * 100:.2f}%"
    )


# ============================================================
# EXPERIMENT
# ============================================================

retrieved_chunks = [

    """
    Retrieval augmented generation combines document
    retrieval with a large language model. The retrieval
    stage searches a knowledge base for relevant information.
    This approach is useful for question answering systems.
    """,

    """
    Vector databases store numerical representations
    called embeddings. These embeddings allow systems to
    search for semantically similar documents.
    Vector databases are commonly used in RAG systems.
    """,

    """
    Query rewriting transforms a user's original question
    into a clearer retrieval query. This can improve the
    quality of retrieved documents.
    """,

    """
    Reranking reorders retrieved documents based on their
    relevance to the user's query. It can improve the final
    context given to the language model.
    """,

    """
    Large language models can generate natural language
    responses from instructions and contextual information.
    Model size and training data can affect response quality.
    """
]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    query = (
        "How does RAG retrieve relevant information "
        "for generating answers?"
    )

    print("\n")
    print("=" * 75)
    print("RAG CONTEXT COMPRESSION")
    print("=" * 75)

    print(
        f"\nQuery:\n{query}"
    )

    # --------------------------------------------------------
    # Original context
    # --------------------------------------------------------

    original_context = "\n".join(
        retrieved_chunks
    )

    print("\n")
    print("=" * 75)
    print("ORIGINAL CONTEXT")
    print("=" * 75)

    print(
        original_context
    )

    # --------------------------------------------------------
    # Score sentences
    # --------------------------------------------------------

    scored_sentences = score_sentences(
        query,
        retrieved_chunks
    )

    ranked_sentences = rank_sentences(
        scored_sentences
    )

    display_sentence_scores(
        ranked_sentences
    )

    # --------------------------------------------------------
    # Compress
    # --------------------------------------------------------

    selected = compress_context(
        query=query,
        chunks=retrieved_chunks,
        top_n=5,
        min_score=0.2
    )

    compressed_context = (
        build_compressed_context(
            selected
        )
    )

    print("\n")
    print("=" * 75)
    print("COMPRESSED CONTEXT")
    print("=" * 75)

    print(
        compressed_context
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    display_statistics(
        original_context,
        compressed_context
    )

    # --------------------------------------------------------
    # Selected sentences
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("SELECTED SENTENCES")
    print("=" * 75)

    for index, result in enumerate(
        selected,
        start=1
    ):

        print(
            f"{index}. "
            f"{result.sentence}"
        )
