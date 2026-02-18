"""
Chunking strategies tailored to each block type.

Text     → semantic-aware sliding window with overlap
Tables   → kept whole (or split by row groups for very large tables)
Figures  → single chunk per figure (caption + OCR + description)
Pages    → optional page-level summary chunk

Each strategy produces ``DocumentChunk | TableChunk | FigureChunk`` objects
with fully populated metadata and citations.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Sequence

from app.ingestion.config import ingest_settings
from app.ingestion.figures import ExtractedFigure
from app.ingestion.layout import LayoutBlock, LayoutLabel
from app.ingestion.schemas import (
    BlockType,
    ChunkMetadata,
    Citation,
    DocumentChunk,
    FigureChunk,
    FigureType,
    TableChunk,
)
from app.ingestion.tables import ExtractedTable

logger = logging.getLogger(__name__)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# Text chunking
# ═══════════════════════════════════════════════════════════════════════════

def chunk_text_blocks(
    blocks: list[LayoutBlock],
    doc_id: str,
    source_file: str,
    page_number: int,
    current_section: str = "",
) -> list[DocumentChunk]:
    """Merge paragraph / heading blocks into overlapping text chunks."""
    # Build section-aware flat text
    full_text = ""
    for b in blocks:
        if b.label == LayoutLabel.HEADING:
            current_section = b.text
            full_text += f"\n## {b.text}\n"
        elif b.label in (LayoutLabel.PARAGRAPH, LayoutLabel.FOOTNOTE):
            full_text += b.text + " "

    full_text = full_text.strip()
    if not full_text:
        return []

    size = ingest_settings.text_chunk_size
    overlap = ingest_settings.text_chunk_overlap
    chunks: list[DocumentChunk] = []
    start = 0
    idx = 0

    while start < len(full_text):
        end = start + size
        snippet = full_text[start:end].strip()
        if len(snippet) < ingest_settings.min_chunk_length:
            start += size - overlap
            continue

        ch = _content_hash(snippet)
        chunks.append(
            DocumentChunk(
                text=snippet,
                metadata=ChunkMetadata(
                    doc_id=doc_id,
                    source_file=source_file,
                    page=page_number,
                    block_type=BlockType.TEXT,
                    section=current_section,
                    content_hash=ch,
                ),
                citation=Citation(
                    source_file=source_file,
                    page=page_number,
                    block_type=BlockType.TEXT,
                ),
            )
        )
        idx += 1
        start += size - overlap

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# Table chunking
# ═══════════════════════════════════════════════════════════════════════════

def chunk_tables(
    tables: list[ExtractedTable],
    doc_id: str,
    source_file: str,
    nearest_section: str = "",
) -> list[TableChunk]:
    """One chunk per table (or split if very large)."""
    results: list[TableChunk] = []
    for tbl_idx, tbl in enumerate(tables, 1):
        exhibit_id = f"Table {tbl_idx} (p.{tbl.page_number})"
        md = tbl.markdown
        if not md.strip():
            continue

        ch = _content_hash(md)
        results.append(
            TableChunk(
                markdown=md,
                csv=tbl.csv_text,
                summary="",  # filled downstream if LLM available
                metadata=ChunkMetadata(
                    doc_id=doc_id,
                    source_file=source_file,
                    page=tbl.page_number,
                    block_type=BlockType.TABLE,
                    section=nearest_section,
                    exhibit_id=exhibit_id,
                    content_hash=ch,
                ),
                citation=Citation(
                    source_file=source_file,
                    page=tbl.page_number,
                    block_type=BlockType.TABLE,
                    exhibit_id=exhibit_id,
                ),
            )
        )
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Figure chunking
# ═══════════════════════════════════════════════════════════════════════════

def chunk_figures(
    figures: list[ExtractedFigure],
    doc_id: str,
    source_file: str,
    figure_types: list[FigureType] | None = None,
    chart_descriptions: list[str] | None = None,
    ocr_texts: list[str] | None = None,
    series_jsons: list[dict | None] | None = None,
    nearest_section: str = "",
) -> list[FigureChunk]:
    """One chunk per figure with all extracted signals."""
    results: list[FigureChunk] = []
    for i, fig in enumerate(figures):
        exhibit_id = f"Figure {fig.figure_index} (p.{fig.page_number})"
        ftype = figure_types[i] if figure_types and i < len(figure_types) else FigureType.UNKNOWN
        desc = chart_descriptions[i] if chart_descriptions and i < len(chart_descriptions) else ""
        ocr = ocr_texts[i] if ocr_texts and i < len(ocr_texts) else ""
        series = series_jsons[i] if series_jsons and i < len(series_jsons) else None

        text_repr = f"{fig.caption} {desc} {ocr}".strip()
        if len(text_repr) < ingest_settings.min_chunk_length:
            text_repr = f"Figure from {source_file} page {fig.page_number}"

        ch = _content_hash(text_repr)

        results.append(
            FigureChunk(
                caption=fig.caption,
                ocr_text=ocr,
                chart_description=desc,
                figure_type=ftype,
                series_json=series,
                image_path=fig.image_path,
                metadata=ChunkMetadata(
                    doc_id=doc_id,
                    source_file=source_file,
                    page=fig.page_number,
                    block_type=BlockType.FIGURE,
                    section=nearest_section,
                    exhibit_id=exhibit_id,
                    content_hash=ch,
                ),
                citation=Citation(
                    source_file=source_file,
                    page=fig.page_number,
                    block_type=BlockType.FIGURE,
                    exhibit_id=exhibit_id,
                ),
            )
        )
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Page summary
# ═══════════════════════════════════════════════════════════════════════════

def create_page_summary(
    raw_text: str,
    doc_id: str,
    source_file: str,
    page_number: int,
) -> DocumentChunk | None:
    """Create a single chunk summarising the entire page content."""
    if not ingest_settings.include_page_summary:
        return None
    text = raw_text.strip()
    if len(text) < ingest_settings.min_chunk_length:
        return None
    # Truncate to max chunk length
    text = text[: ingest_settings.max_chunk_length]
    ch = _content_hash(text)
    return DocumentChunk(
        text=f"[Page {page_number} overview] {text}",
        metadata=ChunkMetadata(
            doc_id=doc_id,
            source_file=source_file,
            page=page_number,
            block_type=BlockType.PAGE_SUMMARY,
            content_hash=ch,
        ),
        citation=Citation(
            source_file=source_file,
            page=page_number,
            block_type=BlockType.PAGE_SUMMARY,
        ),
    )
