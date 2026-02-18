"""
Ingestion pipeline configuration.

All values can be overridden via environment variables prefixed with
``INGEST_`` (e.g. ``INGEST_DPI=300``).
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings

_BASE = Path(__file__).resolve().parent.parent.parent


class IngestSettings(BaseSettings):
    """Tuneable knobs for every pipeline stage."""

    # ── Paths ────────────────────────────────────────────────────────────
    pdf_dir: Path = _BASE / "data" / "unstructured"
    storage_dir: Path = _BASE / "storage"
    chroma_dir: Path = _BASE / "storage" / "chroma_db"
    resources_dir: Path = _BASE / "storage" / "resources"

    # ── PDF rendering ────────────────────────────────────────────────────
    dpi: int = 100  # render resolution for image-based stages
    max_pages: int = 0  # 0 = unlimited

    # ── OCR ──────────────────────────────────────────────────────────────
    ocr_languages: list[str] = ["en"]
    ocr_confidence_threshold: float = 0.40  # drop OCR boxes below this
    ocr_gpu: bool = False  # set True if MPS / CUDA available

    # ── Layout detection ─────────────────────────────────────────────────
    layout_confidence_threshold: float = 0.50

    # ── Table extraction ─────────────────────────────────────────────────
    table_min_rows: int = 2
    table_min_cols: int = 2

    # ── Figure / chart ───────────────────────────────────────────────────
    figure_min_area_ratio: float = 0.02  # skip tiny decorations
    chart_description_max_tokens: int = 300

    # ── Chunking ─────────────────────────────────────────────────────────
    text_chunk_size: int = 450  # characters (keeps tokens under 512 limit)
    text_chunk_overlap: int = 50  # 10% of text_chunk_size
    table_chunk_max_rows: int = 50  # split large tables
    include_page_summary: bool = True

    # ── Embeddings ───────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 64

    # ── Quality gates ────────────────────────────────────────────────────
    min_chunk_length: int = 30  # chars – discard noise
    max_chunk_length: int = 8000
    dedup_enabled: bool = True

    model_config = {
        "env_prefix": "INGEST_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


ingest_settings = IngestSettings()
