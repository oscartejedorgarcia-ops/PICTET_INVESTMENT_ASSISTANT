"""
ChromaDB vector store for the ingestion pipeline.

Features
--------
- Uses pre-computed sentence-transformer embeddings (or falls back to
  ChromaDB's built-in default embedding function).
- Separate collections for text, table, and figure chunks (allows
  block-type-specific retrieval).
- Deduplication via content_hash — upserting prevents duplicates on
  re-ingestion.
- Hybrid retrieval helper that queries multiple collections and merges
  results by score.
"""

from __future__ import annotations

import logging
from typing import Any

import os

import chromadb
from chromadb.config import Settings as ChromaSettings

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from app.ingestion.config import ingest_settings
from app.ingestion.embeddings import embed_texts, embed_query
from app.ingestion.schemas import (
    BlockType,
    DocumentChunk,
    FigureChunk,
    TableChunk,
    chunk_to_metadata_dict,
    chunk_to_text,
)

logger = logging.getLogger(__name__)

_COLLECTION_NAMES = {
    BlockType.TEXT: "ingest_text",
    BlockType.TABLE: "ingest_tables",
    BlockType.FIGURE: "ingest_figures",
    BlockType.PAGE_SUMMARY: "ingest_text",  # summaries go into text collection
}


class IngestVectorStore:
    """Multi-collection ChromaDB store for ingested document chunks."""

    def __init__(self, collection_name: str = None) -> None:
        ingest_settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(ingest_settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collections: dict[str, Any] = {}

        if collection_name:
            # Initialize only the specified collection
            self._collections[collection_name] = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            # Initialize all collections
            for name in set(_COLLECTION_NAMES.values()):
                self._collections[name] = self._client.get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"},
                )

        self._log_stats()

    def _log_stats(self) -> None:
        for name, col in self._collections.items():
            logger.info("ChromaDB collection '%s': %d documents.", name, col.count())

    # ── Write ─────────────────────────────────────────────────────────
    def upsert_chunks(
        self,
        chunks: list[DocumentChunk | TableChunk | FigureChunk],
    ) -> int:
        """Upsert chunks into the appropriate collections. Returns count."""
        if not chunks:
            return 0

        # Group by collection
        groups: dict[str, list] = {}
        for c in chunks:
            col_name = _COLLECTION_NAMES.get(c.metadata.block_type, "ingest_text")
            groups.setdefault(col_name, []).append(c)

        total = 0
        for col_name, group in groups.items():
            ids = [c.metadata.content_hash for c in group]
            texts = [chunk_to_text(c) for c in group]
            metadatas = [chunk_to_metadata_dict(c) for c in group]

            # Deduplicate by ID within the batch (ChromaDB rejects dupes)
            seen: set[str] = set()
            deduped: list[tuple[int, str]] = []
            for idx, cid in enumerate(ids):
                if cid not in seen:
                    seen.add(cid)
                    deduped.append((idx, cid))

            if len(deduped) < len(ids):
                logger.debug(
                    "Dropped %d duplicate chunk(s) in collection '%s'.",
                    len(ids) - len(deduped), col_name,
                )
                ids = [cid for _, cid in deduped]
                texts = [texts[i] for i, _ in deduped]
                metadatas = [metadatas[i] for i, _ in deduped]

            # Try pre-computed embeddings
            embeddings = embed_texts(texts)

            col = self._collections[col_name]
            if embeddings is not None:
                col.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
            else:
                col.upsert(ids=ids, documents=texts, metadatas=metadatas)

            total += len(ids)

        return total

    def add_documents(
        self, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]] | None = None
    ) -> None:
        """Upsert documents into the specified collection."""
        if len(self._collections) != 1:
            raise ValueError("add_documents requires exactly one initialized collection.")

        # Get the single collection
        collection = next(iter(self._collections.values()))
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    # ── Read ──────────────────────────────────────────────────────────
    def query(
        self,
        query_text: str,
        n_results: int = 10,
        block_types: list[BlockType] | None = None,
    ) -> list[dict[str, Any]]:
        """Query across collections, merge, and sort by distance."""
        target_collections: list[str] = []
        if block_types:
            for bt in block_types:
                cn = _COLLECTION_NAMES.get(bt, "ingest_text")
                if cn not in target_collections:
                    target_collections.append(cn)
        else:
            target_collections = list(self._collections.keys())

        query_emb = embed_query(query_text)

        all_results: list[dict[str, Any]] = []
        for col_name in target_collections:
            col = self._collections[col_name]
            if col.count() == 0:
                continue
            try:
                if query_emb is not None:
                    res = col.query(
                        query_embeddings=[query_emb],
                        n_results=min(n_results, col.count()),
                    )
                else:
                    res = col.query(
                        query_texts=[query_text],
                        n_results=min(n_results, col.count()),
                    )
                for doc, meta, dist in zip(
                    res["documents"][0],
                    res["metadatas"][0],
                    res["distances"][0],
                ):
                    all_results.append({
                        "text": doc,
                        "metadata": meta,
                        "distance": dist,
                        "collection": col_name,
                    })
            except Exception as exc:
                logger.warning("Query failed on collection '%s': %s", col_name, exc)

        # Sort by cosine distance (lower = more similar)
        all_results.sort(key=lambda r: r["distance"])
        return all_results[:n_results]

    @property
    def total_count(self) -> int:
        return sum(c.count() for c in self._collections.values())

    def collection_counts(self) -> dict[str, int]:
        return {name: col.count() for name, col in self._collections.items()}
