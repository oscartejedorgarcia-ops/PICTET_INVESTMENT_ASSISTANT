"""Thin wrapper around ChromaDB for document storage and similarity search."""

from __future__ import annotations

import logging
from typing import Any

import os

import chromadb
from chromadb.config import Settings as ChromaSettings

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from app.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "macro_documents"


class VectorStore:
    """Manages a single ChromaDB collection used for macro-economic PDFs."""

    def __init__(self) -> None:
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready (%d documents).",
            COLLECTION_NAME,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Upsert documents (idempotent thanks to deterministic ids)."""
        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def query(self, query_text: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Return the top-*n_results* most similar chunks to *query_text*."""
        results = self._collection.query(query_texts=[query_text], n_results=n_results)
        docs: list[dict[str, Any]] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            docs.append({"text": doc, "metadata": meta, "distance": dist})
        return docs

    @property
    def count(self) -> int:
        return self._collection.count()


# Module-level singleton – instantiated on first import.
vector_store = VectorStore()
