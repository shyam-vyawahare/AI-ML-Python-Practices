
"""
Access-Controlled RAG Retrieval

Features:
- User roles
- Document-level access permissions
- Department-based access
- Authorization before retrieval
- Semantic search over authorized documents

Educational example: identities and policies are defined
locally. Production systems must authenticate users securely.
"""

from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class User:
    user_id: str
    role: str
    department: str


@dataclass
class Document:
    doc_id: str
    text: str
    source: str
    allowed_roles: set[str] = field(default_factory=set)
    allowed_departments: set[str] = field(default_factory=set)


class AccessControlledRAG:
    def __init__(self, documents, model_name=MODEL_NAME):
        self.documents = list(documents)
        self.model = SentenceTransformer(model_name)

        if self.documents:
            self.embeddings = self.model.encode(
                [doc.text for doc in self.documents],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            self.embeddings = np.empty((0, 0))

    @staticmethod
    def is_authorized(user, document):
        """
        Apply role AND department restrictions.

        Empty role or department sets mean unrestricted
        for that particular dimension.
        """

        role_allowed = (
            not document.allowed_roles
            or user.role in document.allowed_roles
        )

        department_allowed = (
            not document.allowed_departments
            or user.department in document.allowed_departments
        )

        return role_allowed and department_allowed

    def retrieve(self, user, query, top_k=3):
        """
        Filter documents by authorization BEFORE ranking.

        The user object must come from a trusted authentication
        layer, not directly from client-submitted role values.
        """

        if not user.user_id:
            raise PermissionError("Unauthenticated user")

        allowed_indices = [
            index
            for index, document in enumerate(self.documents)
            if self.is_authorized(user, document)
        ]

        if not allowed_indices:
            return {
                "query": query,
                "user_id": user.user_id,
                "results": [],
                "message": "No accessible documents found.",
            }

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        # Score only documents authorized for this user.
        authorized_embeddings = self.embeddings[allowed_indices]
        scores = authorized_embeddings @ query_embedding

        ranked_positions = np.argsort(scores)[::-1][:top_k]

        results = []

        for position in ranked_positions:
            original_index = allowed_indices[int(position)]
            document = self.documents[original_index]

            results.append({
                "doc_id": document.doc_id,
                "source": document.source,
                "text": document.text,
                "similarity": float(scores[position]),
            })

        return {
            "query": query,
            "user_id": user.user_id,
            "results": results,
            "message": "Authorized retrieval completed.",
        }


def demo():
    documents = [
        Document(
            doc_id="doc-1",
            source="maintenance_manual.txt",
            text=(
                "Technicians should inspect fault codes, "
                "communication status, and contactor feedback "
                "during charger troubleshooting."
            ),
            allowed_roles={"technician", "admin"},
            allowed_departments={"engineering"},
        ),
        Document(
            doc_id="doc-2",
            source="finance_report.txt",
            text=(
                "The quarterly financial report summarizes "
                "department expenditure and operational costs."
            ),
            allowed_roles={"finance", "admin"},
            allowed_departments={"finance"},
        ),
        Document(
            doc_id="doc-3",
            source="company_handbook.txt",
            text=(
                "Employees must follow workplace policies, "
                "security requirements, and reporting procedures."
            ),
            # Empty permissions mean unrestricted in this demo.
        ),
    ]

    rag = AccessControlledRAG(documents)

    users = [
        User(
            user_id="user-101",
            role="technician",
            department="engineering",
        ),
        User(
            user_id="user-102",
            role="finance",
            department="finance",
        ),
        User(
            user_id="user-103",
            role="admin",
            department="engineering",
        ),
    ]

    query = "Show me operational and maintenance information."

    for user in users:
        response = rag.retrieve(
            user,
            query,
            top_k=3,
        )

        print("\n" + "=" * 60)
        print(f"User: {user.user_id}")
        print(f"Role: {user.role}")
        print(f"Department: {user.department}")
        print(response["message"])

        for result in response["results"]:
            print(
                f"\n[{result['doc_id']}] "
                f"{result['source']}"
            )
            print(f"Similarity: {result['similarity']:.3f}")
            print(result["text"])


if __name__ == "__main__":
    demo()
