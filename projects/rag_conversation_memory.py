"""
RAG Conversation Memory Practice

Implements short-term conversational memory for a RAG system.

Features:
    - Message storage
    - User/assistant turns
    - Configurable memory window
    - Context-aware follow-up rewriting
    - Prompt construction
    - Memory trimming
    - Conversation reset

This implementation uses only Python's standard library.
"""

from dataclasses import dataclass
from typing import List, Optional


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Message:
    role: str
    content: str


# ============================================================
# CONVERSATION MEMORY
# ============================================================

class ConversationMemory:
    """
    Stores recent conversation messages.

    A turn consists of:
        user message
        assistant response
    """

    def __init__(
        self,
        max_turns: int = 5
    ):
        self.max_turns = max_turns
        self.messages: List[Message] = []

    # --------------------------------------------------------
    # ADD MESSAGE
    # --------------------------------------------------------

    def add_message(
        self,
        role: str,
        content: str
    ) -> None:
        """
        Add a message to conversation memory.
        """

        if role not in {
            "user",
            "assistant",
            "system"
        }:
            raise ValueError(
                "Invalid role."
            )

        self.messages.append(
            Message(
                role=role,
                content=content
            )
        )

        self._trim()

    # --------------------------------------------------------
    # ADD TURN
    # --------------------------------------------------------

    def add_turn(
        self,
        user_message: str,
        assistant_message: str
    ) -> None:
        """
        Add a complete conversation turn.
        """

        self.messages.append(
            Message(
                role="user",
                content=user_message
            )
        )

        self.messages.append(
            Message(
                role="assistant",
                content=assistant_message
            )
        )

        self._trim()

    # --------------------------------------------------------
    # TRIM MEMORY
    # --------------------------------------------------------

    def _trim(self) -> None:
        """
        Keep only the configured number of turns.

        Two messages represent one normal turn.
        """

        max_messages = (
            self.max_turns * 2
        )

        if len(self.messages) > max_messages:

            self.messages = self.messages[
                -max_messages:
            ]

    # --------------------------------------------------------
    # GET MESSAGES
    # --------------------------------------------------------

    def get_messages(
        self
    ) -> List[Message]:
        """
        Return stored messages.
        """

        return list(
            self.messages
        )

    # --------------------------------------------------------
    # GET LAST USER QUERY
    # --------------------------------------------------------

    def last_user_query(
        self
    ) -> Optional[str]:
        """
        Return the most recent user message.
        """

        for message in reversed(
            self.messages
        ):

            if message.role == "user":
                return message.content

        return None

    # --------------------------------------------------------
    # GET RECENT CONTEXT
    # --------------------------------------------------------

    def get_recent_context(
        self,
        turns: Optional[int] = None
    ) -> List[Message]:
        """
        Return the most recent conversation turns.
        """

        if turns is None:
            turns = self.max_turns

        messages = turns * 2

        return self.messages[
            -messages:
        ]

    # --------------------------------------------------------
    # FORMAT HISTORY
    # --------------------------------------------------------

    def format_history(
        self,
        turns: Optional[int] = None
    ) -> str:
        """
        Convert conversation history into text.
        """

        messages = self.get_recent_context(
            turns
        )

        if not messages:
            return "(No previous conversation)"

        formatted = []

        for message in messages:

            role = message.role.upper()

            formatted.append(
                f"{role}: {message.content}"
            )

        return "\n".join(
            formatted
        )

    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------

    def clear(self) -> None:
        """
        Clear conversation memory.
        """

        self.messages.clear()

    # --------------------------------------------------------
    # SIZE
    # --------------------------------------------------------

    def size(self) -> int:
        """
        Return number of stored messages.
        """

        return len(
            self.messages
        )


# ============================================================
# FOLLOW-UP DETECTION
# ============================================================

FOLLOW_UP_PHRASES = [
    "how does it work",
    "how does it",
    "what about it",
    "why does it",
    "tell me more",
    "explain it",
    "explain this",
    "what are its benefits",
    "what are its advantages",
    "how about",
]


def is_follow_up(
    query: str
) -> bool:
    """
    Detect whether a query probably depends on
    previous conversation context.
    """

    normalized = query.lower().strip()

    return any(
        phrase in normalized
        for phrase in FOLLOW_UP_PHRASES
    )


# ============================================================
# TOPIC EXTRACTION
# ============================================================

def extract_topic(
    memory: ConversationMemory
) -> Optional[str]:
    """
    Extract a simple topic from the latest
    user question.

    This is intentionally heuristic.
    """

    query = memory.last_user_query()

    if not query:
        return None

    query = query.strip()

    prefixes = [
        "what is ",
        "what are ",
        "explain ",
        "tell me about ",
        "define ",
    ]

    lower_query = query.lower()

    for prefix in prefixes:

        if lower_query.startswith(prefix):

            topic = query[
                len(prefix):
            ]

            return topic.strip(
                "?. "
            )

    return query.strip(
        "?. "
    )


# ============================================================
# QUERY REWRITING
# ============================================================

def rewrite_follow_up(
    query: str,
    memory: ConversationMemory
) -> str:
    """
    Rewrite a conversational follow-up into a
    retrieval-friendly query.
    """

    query = query.strip()

    if not is_follow_up(query):
        return query

    topic = extract_topic(
        memory
    )

    if not topic:
        return query

    lower_query = query.lower()

    if (
        "how does it work"
        in lower_query
    ):
        return (
            f"How does {topic} work?"
        )

    if (
        "what are its benefits"
        in lower_query
    ):
        return (
            f"What are the benefits of {topic}?"
        )

    if (
        "what are its advantages"
        in lower_query
    ):
        return (
            f"What are the advantages of {topic}?"
        )

    if (
        "tell me more"
        in lower_query
    ):
        return (
            f"Detailed information about {topic}"
        )

    if (
        "explain it"
        in lower_query
    ):
        return (
            f"Explain {topic}"
        )

    return (
        f"{topic}: {query}"
    )


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_rag_prompt(
    query: str,
    retrieved_context: str,
    memory: ConversationMemory
) -> str:
    """
    Build a prompt containing:

        - conversation history
        - retrieved knowledge
        - current question
    """

    history = memory.format_history()

    return f"""
You are a helpful RAG assistant.

Use the retrieved context as the primary
source of factual information.

If the context does not contain enough information,
say that the information is not available.

Do not invent unsupported facts.

CONVERSATION HISTORY:
{history}

RETRIEVED CONTEXT:
{retrieved_context}

CURRENT QUESTION:
{query}

ANSWER:
""".strip()


# ============================================================
# SIMULATED RETRIEVER
# ============================================================

def retrieve_context(
    query: str
) -> str:
    """
    Simulate retrieval.

    A real application would call the actual
    RAG retrieval pipeline here.
    """

    knowledge_base = {

        "machine learning": (
            "Machine learning allows systems to learn "
            "patterns from data."
        ),

        "gradient descent": (
            "Gradient descent is an optimization "
            "algorithm used to minimize a loss function."
        ),

        "retrieval augmented generation": (
            "RAG retrieves relevant information from "
            "external knowledge before generating an answer."
        ),
    }

    query_lower = query.lower()

    for topic, context in knowledge_base.items():

        if topic in query_lower:

            return context

    return (
        "No relevant information was found "
        "in the knowledge base."
    )


# ============================================================
# SIMULATED ANSWER GENERATOR
# ============================================================

def generate_answer(
    query: str,
    context: str
) -> str:
    """
    Simulate an LLM response.

    This function represents the point where
    an actual LLM API would be called.
    """

    return (
        f"Based on the retrieved information: "
        f"{context}"
    )


# ============================================================
# CONVERSATIONAL RAG
# ============================================================

def ask_rag(
    query: str,
    memory: ConversationMemory
) -> str:
    """
    Execute a conversational RAG request.
    """

    # --------------------------------------------------------
    # Rewrite query
    # --------------------------------------------------------

    rewritten_query = rewrite_follow_up(
        query,
        memory
    )

    print(
        f"\nOriginal query:"
        f"\n{query}"
    )

    print(
        f"\nRetrieval query:"
        f"\n{rewritten_query}"
    )

    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    context = retrieve_context(
        rewritten_query
    )

    # --------------------------------------------------------
    # Build prompt
    # --------------------------------------------------------

    prompt = build_rag_prompt(
        query=query,
        retrieved_context=context,
        memory=memory
    )

    print(
        "\nGenerated RAG prompt:"
    )

    print(
        "-" * 60
    )

    print(
        prompt
    )

    print(
        "-" * 60
    )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    answer = generate_answer(
        query=query,
        context=context
    )

    # --------------------------------------------------------
    # Store conversation
    # --------------------------------------------------------

    memory.add_turn(
        user_message=query,
        assistant_message=answer
    )

    return answer


# ============================================================
# MEMORY INSPECTION
# ============================================================

def display_memory(
    memory: ConversationMemory
):
    """
    Display current conversation memory.
    """

    print("\n")
    print("=" * 60)
    print("CONVERSATION MEMORY")
    print("=" * 60)

    for index, message in enumerate(
        memory.get_messages(),
        start=1
    ):

        print(
            f"\n{index}. "
            f"{message.role.upper()}"
        )

        print(
            message.content
        )


# ============================================================
# MEMORY WINDOW EXPERIMENT
# ============================================================

def memory_window_experiment():
    """
    Demonstrate how old conversation turns are removed.
    """

    print("\n")
    print("=" * 60)
    print("MEMORY WINDOW EXPERIMENT")
    print("=" * 60)

    memory = ConversationMemory(
        max_turns=2
    )

    conversations = [
        (
            "What is machine learning?",
            "Machine learning learns patterns from data."
        ),
        (
            "What is deep learning?",
            "Deep learning uses multi-layer neural networks."
        ),
        (
            "What is RAG?",
            "RAG retrieves external information for generation."
        ),
    ]

    for user, assistant in conversations:

        memory.add_turn(
            user,
            assistant
        )

    display_memory(
        memory
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("RAG CONVERSATION MEMORY")
    print("=" * 60)

    memory = ConversationMemory(
        max_turns=3
    )

    # --------------------------------------------------------
    # First question
    # --------------------------------------------------------

    answer = ask_rag(
        "What is machine learning?",
        memory
    )

    print(
        f"\nAssistant:\n{answer}"
    )

    # --------------------------------------------------------
    # Follow-up question
    # --------------------------------------------------------

    answer = ask_rag(
        "How does it work?",
        memory
    )

    print(
        f"\nAssistant:\n{answer}"
    )

    # --------------------------------------------------------
    # Another follow-up
    # --------------------------------------------------------

    answer = ask_rag(
        "What are its benefits?",
        memory
    )

    print(
        f"\nAssistant:\n{answer}"
    )

    # --------------------------------------------------------
    # Display memory
    # --------------------------------------------------------

    display_memory(
        memory
    )

    # --------------------------------------------------------
    # Memory window experiment
    # --------------------------------------------------------

    memory_window_experiment()
