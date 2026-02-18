"""
Pydantic models for every artifact that flows through the ingestion pipeline.

Three top-level chunk types are stored in the vector DB:
  - DocumentChunk  – prose / paragraph text
  - TableChunk     – extracted table (markdown + optional CSV + summary)
  - FigureChunk    – extracted figure (caption + OCR overlay + chart description)

All share a common ``ChunkMetadata`` base so retrieval code can treat them
uniformly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────

class BlockType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    PAGE_SUMMARY = "page_summary"


class FigureType(str, Enum):
    LINE_CHART = "line_chart"
    MULTI_LINE_CHART = "multi_line_chart"
    AREA_CHART = "area_chart"
    BAR_CHART = "bar_chart"
    STACKED_BAR_CHART = "stacked_bar_chart"
    PIE_CHART = "pie_chart"
    DONUT_CHART = "donut_chart"
    SCATTER_CHART = "scatter_chart"
    BUBBLE_CHART = "bubble_chart"
    BOX_WHISKER = "box_whisker"
    WATERFALL = "waterfall"
    HEATMAP = "heatmap"
    CANDLESTICK = "candlestick"
    HISTOGRAM = "histogram"
    NETWORK_GRAPH = "network_graph"
    PARALLEL_COORDINATES = "parallel_coordinates"
    PHOTO = "photo"
    DIAGRAM = "diagram"
    LOGO = "logo"
    UNKNOWN = "unknown"


# ── Metadata ─────────────────────────────────────────────────────────────

class ChunkMetadata(BaseModel):
    """Fields shared by every chunk stored in the vector DB."""

    doc_id: str = Field(..., description="SHA-256 of the source file content")
    source_file: str
    page: int
    block_type: BlockType
    section: str = ""
    exhibit_id: str = ""  # e.g. "Figure 3", "Table 2"
    entities: list[str] = Field(default_factory=list)
    time_range: str = ""
    units: str = ""
    content_hash: str = Field(..., description="SHA-256 of the chunk text")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class Citation(BaseModel):
    """How to cite back to the original source."""

    source_file: str
    page: int
    block_type: BlockType
    exhibit_id: str = ""

    def __str__(self) -> str:
        parts = [f"{self.source_file}, p.{self.page}"]
        if self.exhibit_id:
            parts.append(self.exhibit_id)
        return " – ".join(parts)


# ── Chunk schemas ────────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    """A chunk of narrative / paragraph text."""

    text: str
    metadata: ChunkMetadata
    citation: Citation


class TableChunk(BaseModel):
    """An extracted table stored as markdown, optional CSV, and a summary."""

    markdown: str
    csv: str = ""
    summary: str = ""
    metadata: ChunkMetadata
    citation: Citation


class FigureChunk(BaseModel):
    """An extracted figure / chart image with rich textual representations."""

    caption: str = ""
    ocr_text: str = ""
    chart_description: str = ""
    figure_type: FigureType = FigureType.UNKNOWN
    series_json: dict[str, Any] | None = None  # optional digitised data
    image_path: str = ""  # relative to storage/resources
    metadata: ChunkMetadata
    citation: Citation


# ── Helpers ──────────────────────────────────────────────────────────────

def chunk_to_text(chunk: DocumentChunk | TableChunk | FigureChunk) -> str:
    """Flatten any chunk type into the string that will be embedded."""
    if isinstance(chunk, DocumentChunk):
        return chunk.text
    if isinstance(chunk, TableChunk):
        parts = [chunk.markdown]
        if chunk.summary:
            parts.append(f"Summary: {chunk.summary}")
        return "\n".join(parts)
    if isinstance(chunk, FigureChunk):
        parts: list[str] = []
        if chunk.caption:
            parts.append(f"Caption: {chunk.caption}")
        if chunk.chart_description:
            parts.append(chunk.chart_description)
        if chunk.ocr_text:
            parts.append(f"OCR overlay: {chunk.ocr_text}")
        return "\n".join(parts) if parts else "(figure – no text extracted)"
    return str(chunk)


def chunk_to_metadata_dict(chunk: DocumentChunk | TableChunk | FigureChunk) -> dict[str, Any]:
    """Serialise metadata into a flat dict suitable for ChromaDB."""
    meta = chunk.metadata.model_dump()
    meta["citation"] = str(chunk.citation)
    # ChromaDB metadata values must be str | int | float | bool
    meta["entities"] = ", ".join(meta.get("entities", []))
    meta["block_type"] = meta["block_type"].value if hasattr(meta["block_type"], "value") else str(meta["block_type"])
    if isinstance(chunk, FigureChunk):
        meta["figure_type"] = chunk.figure_type.value
        meta["image_path"] = chunk.image_path
    return meta
