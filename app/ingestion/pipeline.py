"""
End-to-end ingestion pipeline orchestrator.

Wires together: PDF parsing → layout → OCR fallback → tables → figures →
charts → chunking → quality gates → embedding → vector DB upsert.

Designed for:
- Incremental re-ingestion (skip already-ingested docs by file hash).
- Batch processing of a folder of PDFs.
- Configurable via ``IngestSettings``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.ingestion.charts import (
    chart_to_text,
    classify_chart_type,
    digitise_chart,
    extract_chart_text,
)
from app.ingestion.chunker import (
    chunk_figures,
    chunk_tables,
    chunk_text_blocks,
    create_page_summary,
)
from app.ingestion.config import ingest_settings
from app.ingestion.figures import extract_figures
from app.ingestion.layout import LayoutLabel, analyse_layout, group_paragraphs
from app.ingestion.ocr import ocr_image_bytes, ocr_to_text
from app.ingestion.pdf_parser import PageData, compute_file_hash, parse_pdf
from app.ingestion.quality import filter_chunks
from app.ingestion.schemas import DocumentChunk, FigureChunk, FigureType, TableChunk
from app.ingestion.tables import extract_tables_pdfplumber
from app.ingestion.vectordb import IngestVectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """Counters for a single ingestion run."""

    files_processed: int = 0
    files_skipped: int = 0
    pages_processed: int = 0
    text_chunks: int = 0
    table_chunks: int = 0
    figure_chunks: int = 0
    chunks_rejected: int = 0
    total_stored: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class _ProcessedDocHashes:
    """Track which docs are already in the DB to skip re-ingestion."""

    known: set[str] = field(default_factory=set)

    def is_known(self, doc_hash: str) -> bool:
        return doc_hash in self.known

    def mark(self, doc_hash: str) -> None:
        self.known.add(doc_hash)


# Module-level state
_store: IngestVectorStore | None = None
_hashes = _ProcessedDocHashes()


def get_store() -> IngestVectorStore:
    global _store
    if _store is None:
        _store = IngestVectorStore()
    return _store


def ingest_file(
    file_path: Path,
    *,
    force: bool = False,
) -> IngestionStats:
    """Ingest a single PDF file into the vector store.

    Args:
        file_path: Path to the PDF file.
        force: Re-ingest even if the file hash is already known.

    Returns:
        ``IngestionStats`` with counters.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error("File does not exist: %s", file_path)
        return IngestionStats()

    stats = IngestionStats()
    t0 = time.time()
    store = get_store()
    doc_hash = compute_file_hash(file_path)

    if not force and _hashes.is_known(doc_hash):
        logger.info("Skipping %s (already ingested).", file_path.name)
        stats.files_skipped = 1
        stats.elapsed_seconds = time.time() - t0
        return stats

    logger.info("═══ Ingesting: %s ═══", file_path.name)
    file_stats = _ingest_single(file_path, doc_hash, store)

    stats.files_processed = 1
    stats.pages_processed = file_stats["pages"]
    stats.text_chunks = file_stats["text"]
    stats.table_chunks = file_stats["tables"]
    stats.figure_chunks = file_stats["figures"]
    stats.chunks_rejected = file_stats["rejected"]
    stats.total_stored = file_stats["stored"]

    _hashes.mark(doc_hash)
    stats.elapsed_seconds = time.time() - t0
    logger.info(
        "Single file ingestion complete: %d pages, %d chunks stored in %.1fs.",
        stats.pages_processed,
        stats.total_stored,
        stats.elapsed_seconds,
    )
    return stats


def ingest_folder(
    pdf_dir: Path | None = None,
    *,
    force: bool = False,
) -> IngestionStats:
    """Ingest all PDFs in *pdf_dir* into the vector store.

    Args:
        pdf_dir: Directory containing PDF files. Defaults to config.
        force: Re-ingest even if the file hash is already known.

    Returns:
        ``IngestionStats`` with counters.
    """
    pdf_dir = pdf_dir or ingest_settings.pdf_dir
    if not pdf_dir.exists():
        logger.warning("PDF directory does not exist: %s", pdf_dir)
        return IngestionStats()

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.info("No PDF files found in %s.", pdf_dir)
        return IngestionStats()

    stats = IngestionStats()
    t0 = time.time()
    store = get_store()

    for pdf_path in pdf_files:
        doc_hash = compute_file_hash(pdf_path)

        if not force and _hashes.is_known(doc_hash):
            logger.info("Skipping %s (already ingested).", pdf_path.name)
            stats.files_skipped += 1
            continue

        logger.info("═══ Ingesting: %s ═══", pdf_path.name)
        file_stats = _ingest_single(pdf_path, doc_hash, store)

        stats.files_processed += 1
        stats.pages_processed += file_stats["pages"]
        stats.text_chunks += file_stats["text"]
        stats.table_chunks += file_stats["tables"]
        stats.figure_chunks += file_stats["figures"]
        stats.chunks_rejected += file_stats["rejected"]
        stats.total_stored += file_stats["stored"]

        _hashes.mark(doc_hash)

    stats.elapsed_seconds = time.time() - t0
    logger.info(
        "Ingestion complete: %d files, %d pages, %d chunks stored in %.1fs.",
        stats.files_processed,
        stats.pages_processed,
        stats.total_stored,
        stats.elapsed_seconds,
    )
    return stats


def _ingest_single(
    pdf_path: Path,
    doc_id: str,
    store: IngestVectorStore,
) -> dict[str, int]:
    """Process one PDF file. Returns per-file counters."""
    all_chunks: list[DocumentChunk | TableChunk | FigureChunk] = []
    pages_count = 0
    current_section = ""

    for page in parse_pdf(pdf_path):
        pages_count += 1
        logger.debug("  Page %d …", page.page_number)

        # ── Layout analysis ──────────────────────────────────────────
        layout_blocks = analyse_layout(page)

        # ── Text extraction (native or OCR fallback) ─────────────────
        text_blocks = [b for b in layout_blocks if b.label in (
            LayoutLabel.HEADING, LayoutLabel.PARAGRAPH, LayoutLabel.FOOTNOTE,
        )]

        if not page.has_text_layer and page.pixmap_bytes:
            # OCR fallback for scanned pages
            logger.debug("    Page %d: no text layer → OCR fallback.", page.page_number)
            ocr_boxes = ocr_image_bytes(page.pixmap_bytes)
            ocr_text = ocr_to_text(ocr_boxes)
            if ocr_text:
                from app.ingestion.layout import LayoutBlock
                text_blocks = [
                    LayoutBlock(
                        label=LayoutLabel.PARAGRAPH,
                        bbox=(0, 0, page.width, page.height),
                        text=ocr_text,
                        page_number=page.page_number,
                    )
                ]

        # Track section headings
        for b in text_blocks:
            if b.label == LayoutLabel.HEADING:
                current_section = b.text

        # Merge paragraphs
        merged = group_paragraphs(text_blocks)

        # Chunk text
        text_chunks = chunk_text_blocks(
            merged, doc_id, pdf_path.name, page.page_number, current_section
        )
        all_chunks.extend(text_chunks)

        # ── Tables ───────────────────────────────────────────────────
        tables = extract_tables_pdfplumber(pdf_path, page.page_number)
        if not tables and page.pixmap_bytes:
            # OCR fallback for tables detected by layout
            for lb in layout_blocks:
                if lb.label == LayoutLabel.TABLE and page.pixmap_bytes:
                    from app.ingestion.tables import extract_table_ocr
                    ocr_tbl = extract_table_ocr(
                        page.pixmap_bytes, lb.bbox, page.width, page.height, page.page_number
                    )
                    if ocr_tbl:
                        tables.append(ocr_tbl)

        table_chunks = chunk_tables(tables, doc_id, pdf_path.name, current_section)
        all_chunks.extend(table_chunks)

        # ── Figures & Charts ─────────────────────────────────────────
        figures = extract_figures(page, layout_blocks, doc_id)
        if figures:
            fig_types: list[FigureType] = []
            chart_descs: list[str] = []
            ocr_texts: list[str] = []
            series_jsons: list[dict | None] = []

            for fig in figures:
                
                # OCR the figure region
                ocr_txt = extract_chart_text(fig.image_bytes)
                ocr_texts.append(ocr_txt)
                
                # Classify chart type
                ftype = classify_chart_type(fig.caption, ocr_txt)
                fig_types.append(ftype)
               

                # Chart-to-text description
                desc = chart_to_text(fig.image_bytes, fig.caption, ocr_txt)
                chart_descs.append(desc)
                

                # Digitisation
                series = digitise_chart(fig.image_bytes) if ftype != FigureType.UNKNOWN else None
                series_jsons.append(series)
                
            

            fig_chunks = chunk_figures(
                figures, doc_id, pdf_path.name,
                figure_types=fig_types,
                chart_descriptions=chart_descs,
                ocr_texts=ocr_texts,
                series_jsons=series_jsons,
                nearest_section=current_section,
            )
            all_chunks.extend(fig_chunks)

        # ── Page summary ─────────────────────────────────────────────
        page_summary = create_page_summary(
            page.raw_text, doc_id, pdf_path.name, page.page_number
        )
        if page_summary:
            all_chunks.append(page_summary)

    # ── Quality gates ────────────────────────────────────────────────
    before_count = len(all_chunks)
    all_chunks = filter_chunks(all_chunks)
    rejected = before_count - len(all_chunks)

    # ── Store ────────────────────────────────────────────────────────
    stored = store.upsert_chunks(all_chunks)

    text_count = sum(1 for c in all_chunks if isinstance(c, DocumentChunk))
    table_count = sum(1 for c in all_chunks if isinstance(c, TableChunk))
    figure_count = sum(1 for c in all_chunks if isinstance(c, FigureChunk))

    logger.info(
        "  %s: %d pages → %d text, %d table, %d figure chunks "
        "(%d rejected, %d stored).",
        pdf_path.name,
        pages_count,
        text_count,
        table_count,
        figure_count,
        rejected,
        stored,
    )

    return {
        "pages": pages_count,
        "text": text_count,
        "tables": table_count,
        "figures": figure_count,
        "rejected": rejected,
        "stored": stored,
    }
