"""
Table extraction from PDF pages.

Strategy (with fallback chain)
------------------------------
1. **pdfplumber** (MIT) – extracts vector / digitally-drawn tables by
   detecting ruled lines.  Works excellently on most financial PDFs.
2. **OCR-based fallback** – for scanned / raster tables, crop the table
   region and run EasyOCR, then attempt to reconstruct rows/columns
   via y-coordinate clustering.
3. If both fail, the raw text layer from PyMuPDF is used.

Output
------
Each table is returned as ``ExtractedTable`` with markdown, CSV, and row data.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.ingestion.config import ingest_settings

logger = logging.getLogger(__name__)


@dataclass
class ExtractedTable:
    """Result of table extraction."""

    page_number: int
    bbox: tuple[float, float, float, float]
    rows: list[list[str]] = field(default_factory=list)
    markdown: str = ""
    csv_text: str = ""
    extraction_method: str = ""  # "pdfplumber" | "ocr" | "text_layer"


# ═══════════════════════════════════════════════════════════════════════════
# 1. pdfplumber-based extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_tables_pdfplumber(pdf_path: Path, page_number: int) -> list[ExtractedTable]:
    """Extract tables from a single page using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed – skipping vector table extraction.")
        return []

    tables: list[ExtractedTable] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_number < 1 or page_number > len(pdf.pages):
                return []
            p = pdf.pages[page_number - 1]
            found = p.find_tables()
            for tbl in found:
                raw = tbl.extract()
                if not raw:
                    continue
                # Clean None → ""
                rows = [[str(cell) if cell else "" for cell in row] for row in raw]
                if len(rows) < ingest_settings.table_min_rows:
                    continue
                if rows and len(rows[0]) < ingest_settings.table_min_cols:
                    continue

                bbox_rect = tbl.bbox  # (x0, y0, x1, y1) in pdfplumber coords
                tables.append(
                    ExtractedTable(
                        page_number=page_number,
                        bbox=tuple(bbox_rect),
                        rows=rows,
                        markdown=_rows_to_markdown(rows),
                        csv_text=_rows_to_csv(rows),
                        extraction_method="pdfplumber",
                    )
                )
    except Exception as exc:
        logger.warning("pdfplumber failed on page %d: %s", page_number, exc)

    return tables


# ═══════════════════════════════════════════════════════════════════════════
# 2. OCR-based table fallback
# ═══════════════════════════════════════════════════════════════════════════

def extract_table_ocr(
    page_png: bytes,
    bbox: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
    page_number: int,
) -> ExtractedTable | None:
    """Crop a table region, OCR it, and cluster into rows."""
    from app.ingestion.ocr import crop_region, ocr_image_bytes

    region = crop_region(page_png, bbox, page_width, page_height)
    boxes = ocr_image_bytes(region)
    if not boxes:
        return None

    # Cluster by Y coordinate (simple row detection)
    rows_map: dict[int, list[tuple[int, str]]] = {}
    tolerance = 12  # pixels
    for box in boxes:
        y_center = (box.bbox[1] + box.bbox[3]) // 2
        # Find existing row key within tolerance
        matched = False
        for key in list(rows_map.keys()):
            if abs(key - y_center) < tolerance:
                rows_map[key].append((box.bbox[0], box.text))
                matched = True
                break
        if not matched:
            rows_map[y_center] = [(box.bbox[0], box.text)]

    # Sort rows top→bottom, cells left→right
    sorted_keys = sorted(rows_map.keys())
    rows = []
    for key in sorted_keys:
        cells = sorted(rows_map[key], key=lambda c: c[0])
        rows.append([c[1] for c in cells])

    if len(rows) < ingest_settings.table_min_rows:
        return None

    return ExtractedTable(
        page_number=page_number,
        bbox=bbox,
        rows=rows,
        markdown=_rows_to_markdown(rows),
        csv_text=_rows_to_csv(rows),
        extraction_method="ocr",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Formatting helpers
# ═══════════════════════════════════════════════════════════════════════════

def _rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    # Pad rows to uniform column count
    max_cols = max(len(r) for r in rows)
    padded = [r + [""] * (max_cols - len(r)) for r in rows]

    lines: list[str] = []
    header = "| " + " | ".join(padded[0]) + " |"
    separator = "| " + " | ".join(["---"] * max_cols) + " |"
    lines.append(header)
    lines.append(separator)
    for row in padded[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _rows_to_csv(rows: list[list[str]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()
