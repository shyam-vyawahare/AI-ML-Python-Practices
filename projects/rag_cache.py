"""
RAG Cache Practice

Implements a lightweight in-memory cache for RAG systems.

Features:
    - Query normalization
    - Cache hits / misses
    - TTL expiration
    - Retrieval-result caching
    - Answer caching
    - Cache invalidation
    - Cache statistics

This implementation uses only Python's standard library.
"""

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


# ============================================================
# CACHE ENTRY
# ============================================================

@dataclass
class CacheEntry:
    value: Any
    created_at: float
    expires_at: float


# ============================================================
# CACHE
# ============================================================

class TTLCache:
    """
    Simple in-memory cache with TTL support.
    """

    def __init__(
        self,
        default_ttl: int = 300
    ):
        self.default_ttl = default_ttl

        self._cache: Dict[
            str,
            CacheEntry
        ] = {}

        self.hits = 0
        self.misses = 0

    # --------------------------------------------------------
    # SET
    # --------------------------------------------------------

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """
        Store a value in the cache.
        """

        if ttl is None:
            ttl = self.default_ttl

        current_time = time.time()

        self._cache[key] = CacheEntry(
            value=value,
            created_at=current_time,
            expires_at=current_time + ttl
        )

    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

    def get(
        self,
        key: str
    ) -> Optional[Any]:
        """
        Retrieve a value from the cache.

        Returns None on cache miss or expiration.
        """

        entry = self._cache.get(
            key
        )

        if entry is None:

            self.misses += 1

            return None

        current_time = time.time()

        # ----------------------------------------------------
        # Expiration
        # ----------------------------------------------------

        if current_time >= entry.expires_at:

            del self._cache[key]

            self.misses += 1

            return None

        self.hits += 1

        return entry.value

    # --------------------------------------------------------
    # DELETE
    # --------------------------------------------------------

    def delete(
        self,
        key: str
    ) -> bool:
        """
        Delete a specific cache entry.
        """

        if key in self._cache:

            del self._cache[key]

            return True

        return False

    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------

    def clear(self) -> None:
        """
        Clear the entire cache.
        """

        self._cache.clear()

    # --------------------------------------------------------
    # SIZE
    # --------------------------------------------------------

    def size(self) -> int:
        """
        Return number of cached entries.
        """

        return len(
            self._cache
        )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(self) -> Dict[str, Any]:
        """
        Return cache statistics.
        """

        total_requests = (
            self.hits +
            self.misses
        )

        hit_rate = (
            self.hits / total_requests
            if total_requests
            else 0.0
        )

        return {
            "entries": self.size(),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }


# ============================================================
# QUERY NORMALIZATION
# ============================================================

def normalize_query(
    query: str
) -> str:
    """
    Normalize a user query so semantically identical
    formatting variations can share the same cache key.
    """

    query = query.strip().lower()

    # Normalize whitespace.
    query = " ".join(
        query.split()
    )

    # Remove trailing question marks.
    query = query.rstrip("?")

    return query


# ============================================================
# CACHE KEY
# ============================================================

def create_cache_key(
    prefix: str,
    query: str
) -> str:
    """
    Create a deterministic cache key.

    Hashing keeps keys compact and avoids storing the
    complete query directly as the key.
    """

    normalized_query = normalize_query(
        query
    )

    raw_key = (
        f"{prefix}:{normalized_query}"
    )

    return hashlib.sha256(
        raw_key.encode("utf-8")
    ).hexdigest()


# ============================================================
# RAG CACHE
# ============================================================

class RAGCache:
    """
    Separate caches for retrieval and generated answers.
    """

    def __init__(
        self,
        retrieval_ttl: int = 300,
        answer_ttl: int = 600
    ):

        self.retrieval_cache = TTLCache(
            default_ttl=retrieval_ttl
        )

        self.answer_cache = TTLCache(
            default_ttl=answer_ttl
        )

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    def get_retrieval(
        self,
        query: str
    ):

        key = create_cache_key(
            "retrieval",
            query
        )

        return self.retrieval_cache.get(
            key
        )

    def set_retrieval(
        self,
        query: str,
        results,
        ttl: Optional[int] = None
    ):

        key = create_cache_key(
            "retrieval",
            query
        )

        self.retrieval_cache.set(
            key,
            results,
            ttl
        )

    # --------------------------------------------------------
    # ANSWERS
    # --------------------------------------------------------

    def get_answer(
        self,
        query: str
    ):

        key = create_cache_key(
            "answer",
            query
        )

        return self.answer_cache.get(
            key
        )

    def set_answer(
        self,
        query: str,
        answer: str,
        ttl: Optional[int] = None
    ):

        key = create_cache_key(
            "answer",
            query
        )

        self.answer_cache.set(
            key,
            answer,
            ttl
        )

    # --------------------------------------------------------
    # INVALIDATION
    # --------------------------------------------------------

    def invalidate_query(
        self,
        query: str
    ) -> None:
        """
        Remove both retrieval and answer cache entries.
        """

        retrieval_key = create_cache_key(
            "retrieval",
            query
        )

        answer_key = create_cache_key(
            "answer",
            query
        )

        self.retrieval_cache.delete(
            retrieval_key
        )

        self.answer_cache.delete(
            answer_key
        )

    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------

    def clear(self) -> None:
        """
        Clear both caches.
        """

        self.retrieval_cache.clear()

        self.answer_cache.clear()

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    def statistics(self):
        """
        Return statistics for both cache layers.
        """

        return {
            "retrieval": (
                self.retrieval_cache.statistics()
            ),

            "answer": (
                self.answer_cache.statistics()
            ),
        }


# ============================================================
# SIMULATED RETRIEVAL
# ============================================================

def retrieve_documents(
    query: str
):
    """
    Simulate an expensive retrieval operation.
    """

    print(
        f"\n[RETRIEVAL] Searching for: {query}"
    )

    # Simulated result.
    return [
        "chunk_01",
        "chunk_03",
        "chunk_07"
    ]


# ============================================================
# SIMULATED LLM
# ============================================================

def generate_answer(
    query: str,
    context
) -> str:
    """
    Simulate an expensive LLM generation call.
    """

    print(
        f"[LLM] Generating answer for: {query}"
    )

    return (
        f"Answer generated for '{query}' "
        f"using {len(context)} retrieved chunks."
    )


# ============================================================
# CACHED RAG PIPELINE
# ============================================================

def ask_rag(
    query: str,
    cache: RAGCache
) -> str:
    """
    Run a RAG pipeline with caching.
    """

    print("\n" + "-" * 60)
    print(
        f"Query: {query}"
    )

    # --------------------------------------------------------
    # Answer cache
    # --------------------------------------------------------

    cached_answer = cache.get_answer(
        query
    )

    if cached_answer is not None:

        print(
            "[CACHE HIT] Returning cached answer."
        )

        return cached_answer

    print(
        "[CACHE MISS] Answer not found."
    )

    # --------------------------------------------------------
    # Retrieval cache
    # --------------------------------------------------------

    cached_retrieval = cache.get_retrieval(
        query
    )

    if cached_retrieval is not None:

        print(
            "[CACHE HIT] Using cached retrieval."
        )

        retrieval_results = cached_retrieval

    else:

        print(
            "[CACHE MISS] Retrieval not found."
        )

        retrieval_results = retrieve_documents(
            query
        )

        cache.set_retrieval(
            query,
            retrieval_results
        )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    answer = generate_answer(
        query,
        retrieval_results
    )

    # --------------------------------------------------------
    # Store answer
    # --------------------------------------------------------

    cache.set_answer(
        query,
        answer
    )

    return answer


# ============================================================
# CACHE STATISTICS
# ============================================================

def print_statistics(
    cache: RAGCache
):
    """
    Display cache statistics.
    """

    statistics = cache.statistics()

    print("\n")
    print("=" * 60)
    print("CACHE STATISTICS")
    print("=" * 60)

    for cache_type, data in statistics.items():

        print(
            f"\n{cache_type.upper()} CACHE"
        )

        print(
            f"Entries: "
            f"{data['entries']}"
        )

        print(
            f"Hits: "
            f"{data['hits']}"
        )

        print(
            f"Misses: "
            f"{data['misses']}"
        )

        print(
            f"Hit Rate: "
            f"{data['hit_rate']:.2%}"
        )


# ============================================================
# TTL EXPERIMENT
# ============================================================

def ttl_experiment():
    """
    Demonstrate cache expiration.
    """

    print("\n")
    print("=" * 60)
    print("TTL EXPERIMENT")
    print("=" * 60)

    cache = TTLCache(
        default_ttl=1
    )

    cache.set(
        "temporary",
        "cached value"
    )

    print(
        "\nImmediately after storing:"
    )

    print(
        cache.get("temporary")
    )

    print(
        "\nWaiting for expiration..."
    )

    time.sleep(1.2)

    print(
        "After TTL expiration:"
    )

    print(
        cache.get("temporary")
    )


# ============================================================
# NORMALIZATION EXPERIMENT
# ============================================================

def normalization_experiment(
    cache: RAGCache
):
    """
    Demonstrate that formatting variations can
    share the same cache entry.
    """

    print("\n")
    print("=" * 60)
    print("QUERY NORMALIZATION EXPERIMENT")
    print("=" * 60)

    query_1 = (
        "What is Retrieval Augmented Generation?"
    )

    query_2 = (
        "  what is retrieval augmented generation  "
    )

    cache.set_answer(
        query_1,
        "RAG combines retrieval with generation."
    )

    result = cache.get_answer(
        query_2
    )

    print(
        f"\nQuery 1: {query_1}"
    )

    print(
        f"Query 2: {query_2}"
    )

    print(
        f"\nCached result:"
    )

    print(
        result
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("RAG CACHING PRACTICE")
    print("=" * 60)

    cache = RAGCache(
        retrieval_ttl=300,
        answer_ttl=600
    )

    query = (
        "How does retrieval augmented generation work?"
    )

    # --------------------------------------------------------
    # First request
    # --------------------------------------------------------

    answer = ask_rag(
        query,
        cache
    )

    print(
        f"\nAnswer: {answer}"
    )

    # --------------------------------------------------------
    # Second request
    # --------------------------------------------------------

    answer = ask_rag(
        query,
        cache
    )

    print(
        f"\nAnswer: {answer}"
    )

    # --------------------------------------------------------
    # Similar formatting
    # --------------------------------------------------------

    answer = ask_rag(
        "  How does retrieval augmented generation work?  ",
        cache
    )

    print(
        f"\nAnswer: {answer}"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print_statistics(
        cache
    )

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    normalization_experiment(
        cache
    )

    # --------------------------------------------------------
    # TTL
    # --------------------------------------------------------

    ttl_experiment()
