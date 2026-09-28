"""
RAG Query Rewriting Practice

Improves user queries before they are sent to
the retrieval system.

Techniques:
    1. Conversation-aware rewriting
    2. Query normalization
    3. Abbreviation expansion
    4. Query expansion
    5. Multi-query generation

This implementation uses deterministic rules so that
the retrieval concepts can be understood without
requiring an LLM API.
"""

import re
from dataclasses import dataclass
from typing import List, Optional


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ConversationContext:
    previous_user_query: Optional[str] = None
    previous_answer: Optional[str] = None
    topic: Optional[str] = None


@dataclass
class RewrittenQuery:
    original_query: str
    rewritten_query: str
    expanded_queries: List[str]


# ============================================================
# QUERY NORMALIZATION
# ============================================================

def normalize_query(query: str) -> str:
    """
    Normalize whitespace and punctuation.
    """

    query = query.strip()

    query = re.sub(
        r"\s+",
        " ",
        query
    )

    query = re.sub(
        r"[?]+$",
        "",
        query
    )

    return query


# ============================================================
# ABBREVIATION EXPANSION
# ============================================================

ABBREVIATIONS = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning",
    "nn": "neural network",
    "cnn": "convolutional neural network",
    "nlp": "natural language processing",
    "llm": "large language model",
    "rag": "retrieval augmented generation",
    "api": "application programming interface",
    "db": "database",
    "tf-idf": "term frequency inverse document frequency",
}


def expand_abbreviations(
    query: str
) -> str:
    """
    Expand common AI/ML abbreviations.
    """

    result = query

    for abbreviation, expansion in ABBREVIATIONS.items():

        pattern = rf"\b{re.escape(abbreviation)}\b"

        result = re.sub(
            pattern,
            expansion,
            result,
            flags=re.IGNORECASE
        )

    return result


# ============================================================
# CONVERSATIONAL QUERY DETECTION
# ============================================================

FOLLOW_UP_PATTERNS = [
    r"^how does it work",
    r"^how does it",
    r"^what is it",
    r"^why does it",
    r"^why is it",
    r"^what about it",
    r"^tell me more",
    r"^explain it",
    r"^explain this",
    r"^how about",
    r"^what about",
    r"^and what",
    r"^what are its",
    r"^what is its",
]


def is_follow_up_query(
    query: str
) -> bool:
    """
    Detect whether a query likely depends on
    previous conversation context.
    """

    normalized = normalize_query(
        query
    ).lower()

    return any(
        re.search(
            pattern,
            normalized
        )
        for pattern in FOLLOW_UP_PATTERNS
    )


# ============================================================
# PRONOUN DETECTION
# ============================================================

def contains_reference_words(
    query: str
) -> bool:
    """
    Detect vague references that may require
    conversational context.
    """

    reference_words = {
        "it",
        "this",
        "that",
        "they",
        "them",
        "these",
        "those",
        "its",
    }

    tokens = set(
        re.findall(
            r"\b[a-zA-Z]+\b",
            query.lower()
        )
    )

    return bool(
        tokens.intersection(
            reference_words
        )
    )


# ============================================================
# TOPIC EXTRACTION
# ============================================================

def extract_topic(
    context: ConversationContext
) -> Optional[str]:
    """
    Get the best available topic from conversation context.
    """

    if context.topic:
        return context.topic

    if context.previous_user_query:
        return context.previous_user_query

    return None


# ============================================================
# CONTEXT-AWARE REWRITING
# ============================================================

def rewrite_with_context(
    query: str,
    context: ConversationContext
) -> str:
    """
    Rewrite a vague follow-up query using
    the known conversation topic.
    """

    query = normalize_query(
        query
    )

    topic = extract_topic(
        context
    )

    if not topic:
        return query

    if not (
        is_follow_up_query(query)
        or contains_reference_words(query)
    ):
        return query

    topic = normalize_query(
        topic
    )

    # --------------------------------------------------------
    # Common conversational patterns
    # --------------------------------------------------------

    lower_query = query.lower()

    if lower_query.startswith(
        "how does it work"
    ):
        return f"How does {topic} work?"

    if lower_query.startswith(
        "what is it"
    ):
        return f"What is {topic}?"

    if lower_query.startswith(
        "why does it"
    ):
        return f"Why does {topic} work?"

    if lower_query.startswith(
        "tell me more"
    ):
        return f"More information about {topic}"

    if lower_query.startswith(
        "explain it"
    ):
        return f"Explain {topic}"

    if lower_query.startswith(
        "what are its"
    ):
        return query.replace(
            "its",
            topic
        )

    # Generic fallback
    return f"{topic}: {query}"


# ============================================================
# QUERY EXPANSION
# ============================================================

def generate_query_variations(
    query: str
) -> List[str]:
    """
    Generate multiple retrieval-friendly versions
    of a query.
    """

    normalized = normalize_query(
        query
    )

    expanded = expand_abbreviations(
        normalized
    )

    variations = [
        normalized,
        expanded,
        f"Explain {expanded}",
        f"{expanded} detailed explanation",
    ]

    # Remove duplicates while preserving order.
    unique_variations = []

    for variation in variations:

        if variation not in unique_variations:
            unique_variations.append(
                variation
            )

    return unique_variations


# ============================================================
# COMPLETE QUERY REWRITER
# ============================================================

def rewrite_query(
    query: str,
    context: Optional[ConversationContext] = None
) -> RewrittenQuery:
    """
    Complete query rewriting pipeline.
    """

    original = normalize_query(
        query
    )

    rewritten = original

    # --------------------------------------------------------
    # Conversation-aware rewriting
    # --------------------------------------------------------

    if context:

        rewritten = rewrite_with_context(
            rewritten,
            context
        )

    # --------------------------------------------------------
    # Abbreviation expansion
    # --------------------------------------------------------

    rewritten = expand_abbreviations(
        rewritten
    )

    # --------------------------------------------------------
    # Query expansion
    # --------------------------------------------------------

    expanded_queries = generate_query_variations(
        rewritten
    )

    return RewrittenQuery(
        original_query=original,
        rewritten_query=rewritten,
        expanded_queries=expanded_queries
    )


# ============================================================
# MULTI-QUERY RETRIEVAL PREPARATION
# ============================================================

def build_multi_query(
    rewritten_query: RewrittenQuery
) -> List[str]:
    """
    Prepare unique queries for multi-query retrieval.
    """

    queries = []

    for query in rewritten_query.expanded_queries:

        normalized = normalize_query(
            query
        )

        if normalized not in queries:
            queries.append(
                normalized
            )

    return queries


# ============================================================
# DISPLAY
# ============================================================

def display_rewrite(
    result: RewrittenQuery
):
    """
    Display query rewriting results.
    """

    print("\n")
    print("=" * 70)
    print("QUERY REWRITING RESULT")
    print("=" * 70)

    print(
        f"\nOriginal Query:"
        f"\n{result.original_query}"
    )

    print(
        f"\nRewritten Query:"
        f"\n{result.rewritten_query}"
    )

    print(
        "\nExpanded Queries:"
    )

    for index, query in enumerate(
        result.expanded_queries,
        start=1
    ):

        print(
            f"{index}. {query}"
        )


# ============================================================
# EXPERIMENT 1
# ============================================================

def experiment_follow_up():

    context = ConversationContext(
        previous_user_query=(
            "What is gradient descent?"
        ),

        previous_answer=(
            "Gradient descent is an optimization "
            "algorithm used to minimize a loss function."
        ),

        topic="gradient descent"
    )

    query = "How does it work?"

    result = rewrite_query(
        query,
        context
    )

    display_rewrite(
        result
    )


# ============================================================
# EXPERIMENT 2
# ============================================================

def experiment_abbreviations():

    query = (
        "How is RAG used with an LLM?"
    )

    result = rewrite_query(
        query
    )

    display_rewrite(
        result
    )


# ============================================================
# EXPERIMENT 3
# ============================================================

def experiment_specific_query():

    query = (
        "What are the benefits of vector databases?"
    )

    result = rewrite_query(
        query
    )

    display_rewrite(
        result
    )


# ============================================================
# EXPERIMENT 4
# ============================================================

def experiment_multi_query():

    context = ConversationContext(
        topic="retrieval augmented generation"
    )

    query = "What are its benefits?"

    result = rewrite_query(
        query,
        context
    )

    multi_queries = build_multi_query(
        result
    )

    print("\n")
    print("=" * 70)
    print("MULTI-QUERY RETRIEVAL")
    print("=" * 70)

    print(
        f"\nOriginal:"
        f"\n{query}"
    )

    print(
        "\nQueries prepared for retrieval:"
    )

    for index, search_query in enumerate(
        multi_queries,
        start=1
    ):

        print(
            f"{index}. {search_query}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("RAG QUERY REWRITING PRACTICE")
    print("=" * 70)

    experiment_follow_up()

    experiment_abbreviations()

    experiment_specific_query()

    experiment_multi_query()
