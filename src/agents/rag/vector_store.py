"""ChromaDB vector store for RAG implementation."""

from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from pydantic import BaseModel, Field

from agents.core.config import get_settings
from agents.core.llm_client import OllamaClient


class SearchResult(BaseModel):
    """Result from vector similarity search."""

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = Field(description="Similarity score (lower is more similar)")
    document_id: str


class ChromaVectorStore:
    """Vector store using ChromaDB for document storage and retrieval."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_dir: Optional[Path] = None,
        llm_client: Optional[OllamaClient] = None,
    ):
        """Initialize the ChromaDB vector store.

        Args:
            collection_name: Name of the collection. Defaults to settings.
            persist_dir: Directory to persist data. Defaults to settings.
            llm_client: LLM client for embeddings. Creates new if None.
        """
        settings = get_settings()
        self._collection_name = collection_name or settings.chroma.collection_name
        self._persist_dir = persist_dir or settings.chroma.persist_dir
        self._llm_client = llm_client or OllamaClient()

        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection_name(self) -> str:
        """Get the collection name."""
        return self._collection_name

    @property
    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self._collection.count()

    def add_documents(
        self,
        documents: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """Add documents to the vector store.

        Args:
            documents: List of document texts.
            metadatas: Optional metadata for each document.
            ids: Optional IDs for each document. Auto-generated if None.

        Returns:
            List of document IDs.
        """
        if not documents:
            return []

        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in documents]

        if metadatas is None:
            metadatas = [{} for _ in documents]

        embeddings = self._llm_client.embed_texts(documents)

        self._collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        return ids

    def add_document(
        self,
        document: str,
        metadata: Optional[dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """Add a single document to the vector store.

        Args:
            document: Document text.
            metadata: Optional metadata.
            doc_id: Optional document ID.

        Returns:
            Document ID.
        """
        ids = self.add_documents(
            documents=[document],
            metadatas=[metadata] if metadata else None,
            ids=[doc_id] if doc_id else None,
        )
        return ids[0]

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Search for similar documents.

        Args:
            query: Search query text.
            n_results: Maximum number of results to return.
            where: Optional metadata filter.

        Returns:
            List of SearchResult objects ordered by similarity.
        """
        query_embedding = self._llm_client.embed_text(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                search_results.append(
                    SearchResult(
                        content=doc,
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        score=results["distances"][0][i] if results["distances"] else 0.0,
                        document_id=results["ids"][0][i],
                    )
                )

        return search_results

    def delete_document(self, doc_id: str) -> None:
        """Delete a document by ID.

        Args:
            doc_id: Document ID to delete.
        """
        self._collection.delete(ids=[doc_id])

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self._client.delete_collection(self._collection_name)

    def reset(self) -> None:
        """Reset the collection (delete all documents)."""
        self.delete_collection()
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_document(self, doc_id: str) -> Optional[SearchResult]:
        """Get a document by ID.

        Args:
            doc_id: Document ID.

        Returns:
            SearchResult if found, None otherwise.
        """
        result = self._collection.get(
            ids=[doc_id],
            include=["documents", "metadatas"],
        )

        if result["documents"] and result["documents"][0]:
            return SearchResult(
                content=result["documents"][0],
                metadata=result["metadatas"][0] if result["metadatas"] else {},
                score=0.0,
                document_id=doc_id,
            )
        return None
