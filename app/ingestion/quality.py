"""
Quality gates and validation for the ingestion pipeline.

Each gate is a pure function: input → (pass, reason).
The pipeline calls these to decide whether to keep or discard a chunk.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.ingestion.config import ingest_settings
from app.ingestion.schemas import (
    DocumentChunk,
    FigureChunk,
    TableChunk,
    chunk_to_text,
)

logger = logging.getLogger(__name__)


def validate_text_chunk(chunk: DocumentChunk) -> tuple[bool, str]:
    """Validate a text chunk before insertion."""
    text = chunk.text.strip()

    if len(text) < ingest_settings.min_chunk_length:
        return False, f"Too short ({len(text)} chars)"

    if len(text) > ingest_settings.max_chunk_length:
        return False, f"Too long ({len(text)} chars)"

    # Detect OCR garbage: high ratio of non-alphanumeric characters
    alnum = sum(1 for c in text if c.isalnum())
    if len(text) > 0 and alnum / len(text) < 0.30:
        return False, f"Low alphanumeric ratio ({alnum / len(text):.2f})"

    # Detect repeated character noise
    if _is_repetitive(text):
        return False, "Repetitive content detected"

    return True, "OK"


def validate_table_chunk(chunk: TableChunk) -> tuple[bool, str]:
    """Validate a table chunk."""
    if not chunk.markdown.strip():
        return False, "Empty table"

    # Check row count (from markdown lines)
    lines = [l for l in chunk.markdown.strip().split("\n") if l.strip().startswith("|")]
    data_lines = [l for l in lines if "---" not in l]
    if len(data_lines) < ingest_settings.table_min_rows:
        return False, f"Too few rows ({len(data_lines)})"

    return True, "OK"


def validate_figure_chunk(chunk: FigureChunk) -> tuple[bool, str]:
    """Validate a figure chunk."""
    text = chunk_to_text(chunk).strip()
    if len(text) < 10:
        return False, "Insufficient textual representation"
    return True, "OK"


def filter_chunks(
    chunks: list[DocumentChunk | TableChunk | FigureChunk],
) -> list[DocumentChunk | TableChunk | FigureChunk]:
    """Apply quality gates to a list of chunks, returning only valid ones."""
    passed: list[Any] = []
    rejected = 0

    for chunk in chunks:
        if isinstance(chunk, DocumentChunk):
            ok, reason = validate_text_chunk(chunk)
        elif isinstance(chunk, TableChunk):
            ok, reason = validate_table_chunk(chunk)
        elif isinstance(chunk, FigureChunk):
            ok, reason = validate_figure_chunk(chunk)
        else:
            ok, reason = True, "unknown type"

        if ok:
            passed.append(chunk)
        else:
            rejected += 1
            logger.debug("Chunk rejected (%s): %s", reason, chunk_to_text(chunk)[:80])

    if rejected:
        logger.info("Quality gate: %d chunks passed, %d rejected.", len(passed), rejected)
    return passed


def _is_repetitive(text: str, threshold: float = 0.5) -> bool:
    """Detect text that is mostly the same word/char repeated."""
    words = text.split()
    if len(words) < 5:
        return False
    unique = set(words)
    return len(unique) / len(words) < threshold
