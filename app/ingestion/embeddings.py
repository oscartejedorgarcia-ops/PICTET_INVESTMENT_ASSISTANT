"""
Embedding generation using sentence-transformers (Apache-2.0).

Default model: ``all-MiniLM-L6-v2`` — fast, 384-dim, good general-purpose.
Upgrade path: ``BAAI/bge-base-en-v1.5`` for higher accuracy on finance text,
or ``nomic-ai/nomic-embed-text-v1.5`` (Apache-2.0) with Matryoshka support.

This module handles batched encoding and is used both at ingestion time
(to pre-compute embeddings for ChromaDB) and at query time.
"""

from __future__ import annotations

import logging
import warnings

from app.ingestion.config import ingest_settings

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    """Lazy-load the sentence-transformer model."""
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model '%s' …", ingest_settings.embedding_model)
        _is_mps = _mps_available()
        _model = SentenceTransformer(
            ingest_settings.embedding_model,
            device="mps" if _is_mps else "cpu",
        )
        # Ensure the tokenizer truncates inputs to the model's max length
        # (e.g. 512 tokens for all-MiniLM-L6-v2) to avoid indexing errors.
        _model.max_seq_length = _model.max_seq_length or 512
        _model.tokenizer.model_max_length = _model.max_seq_length
        # Force truncation at the tokenizer level so sequences longer than
        # max_seq_length are silently clipped instead of causing index errors.
        _model.tokenizer.init_kwargs["truncation"] = True
        _model.tokenizer.init_kwargs["max_length"] = _model.max_seq_length
        logger.info("Embedding model loaded (dim=%d, max_seq=%d).",
                    _model.get_sentence_embedding_dimension(),
                    _model.max_seq_length)
    except ImportError:
        logger.warning(
            "sentence-transformers not installed – embeddings unavailable. "
            "ChromaDB will use its built-in default embedding function."
        )
        _model = None
    return _model


def _mps_available() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """Encode a batch of texts into embedding vectors.

    Returns None if the model is unavailable (ChromaDB will fall back to its
    own default embedding function).
    """
    model = _get_model()
    if model is None:
        return None

    batch_size = ingest_settings.embedding_batch_size
    all_embs: list[list[float]] = []
    max_seq = model.max_seq_length
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Suppress 'pin_memory not supported on MPS' and tokenizer
        # 'Token indices sequence length is longer than' warnings.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*pin_memory.*")
            warnings.filterwarnings(
                "ignore",
                message=".*Token indices sequence length.*",
            )
            embs = model.encode(
                batch,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        all_embs.extend(embs.tolist())

    return all_embs


def embed_query(query: str) -> list[float] | None:
    """Encode a single query string."""
    result = embed_texts([query])
    return result[0] if result else None
