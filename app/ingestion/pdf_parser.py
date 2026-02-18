"""
PDF parsing & rendering via PyMuPDF (fitz).

Responsibilities
- Open a PDF and yield page-level data (text blocks, images, rendered pixmaps).
- Provide a consistent ``PageData`` object consumed downstream by layout /
  OCR / table / figure modules.
- Detect whether a page has a usable text layer or is image-only (scanned).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF

from app.ingestion.config import ingest_settings

logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """A single text span extracted from the native PDF text layer."""

    text: str
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    font_size: float = 0.0
    font_name: str = ""
    is_bold: bool = False


@dataclass
class ImageInfo:
    """Metadata about an embedded image (before extraction)."""

    xref: int
    bbox: tuple[float, float, float, float]
    width: int
    height: int


@dataclass
class DrawingCluster:
    """A cluster of vector drawing operations that may form a chart/diagram."""

    bbox: tuple[float, float, float, float]  # bounding box of the cluster
    path_count: int  # number of drawing paths in the cluster
    has_fill: bool  # contains filled shapes
    has_stroke: bool  # contains stroked paths


@dataclass
class PageData:
    """Everything we know about one PDF page after initial parsing."""

    page_number: int  # 1-based
    width: float
    height: float
    text_blocks: list[TextBlock] = field(default_factory=list)
    images: list[ImageInfo] = field(default_factory=list)
    vector_graphics: list[DrawingCluster] = field(default_factory=list)
    raw_text: str = ""
    has_text_layer: bool = True
    pixmap_bytes: bytes | None = None  # PNG rendered at configured DPI


def compute_file_hash(path: Path) -> str:
    """SHA-256 of file contents – used as ``doc_id``."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_pdf(pdf_path: Path) -> Iterator[PageData]:
    """Yield one ``PageData`` per page.

    If ``ingest_settings.max_pages > 0`` only the first N pages are processed.
    """
    doc = fitz.open(str(pdf_path))
    total = len(doc)
    limit = ingest_settings.max_pages or total

    for idx in range(min(total, limit)):
        page: fitz.Page = doc[idx]
        page_number = idx + 1

        # ── native text blocks ───────────────────────────────────────
        blocks: list[TextBlock] = []
        raw_parts: list[str] = []
        for b in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]:
            if b["type"] == 0:  # text block
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        txt = span.get("text", "").strip()
                        if txt:
                            blocks.append(
                                TextBlock(
                                    text=txt,
                                    bbox=tuple(span["bbox"]),
                                    font_size=span.get("size", 0),
                                    font_name=span.get("font", ""),
                                    is_bold="bold" in span.get("font", "").lower(),
                                )
                            )
                            raw_parts.append(txt)

        raw_text = " ".join(raw_parts)
        has_text = len(raw_text.strip()) > 20  # heuristic

        # ── embedded images metadata ─────────────────────────────────
        image_infos: list[ImageInfo] = []
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                img_rect = page.get_image_rects(xref)
                if img_rect:
                    r = img_rect[0]
                    image_infos.append(
                        ImageInfo(
                            xref=xref,
                            bbox=(r.x0, r.y0, r.x1, r.y1),
                            width=int(r.width),
                            height=int(r.height),
                        )
                    )
            except Exception:
                pass  # some images lack rect info

        # ── vector drawings (charts, diagrams drawn with paths) ───────
        drawing_clusters = []
        #drawing_clusters = _cluster_drawings(page)

        # ── rendered pixmap (for OCR / layout / figure stages) ───────
        mat = fitz.Matrix(ingest_settings.dpi / 72, ingest_settings.dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        pixmap_bytes = pix.tobytes("png")

        yield PageData(
            page_number=page_number,
            width=page.rect.width,
            height=page.rect.height,
            text_blocks=blocks,
            images=image_infos,
            vector_graphics=drawing_clusters,
            raw_text=raw_text,
            has_text_layer=has_text,
            pixmap_bytes=pixmap_bytes,
        )

    doc.close()
    logger.info("Parsed %d pages from %s.", min(total, limit), pdf_path.name)


def _cluster_drawings(page: fitz.Page) -> list[DrawingCluster]:
    """Detect clusters of vector drawing paths that likely form charts/diagrams.

    PyMuPDF's ``get_drawings()`` returns individual path operations (lines,
    curves, rects).  We cluster nearby paths into groups and keep only those
    large enough to be a chart or diagram (filtered by
    ``figure_min_area_ratio``).
    """
    try:
        drawings = page.get_drawings()
    except Exception:
        return []

    if not drawings:
        return []

    page_area = max(page.rect.width * page.rect.height, 1)
    min_paths = 5  # a chart typically has many strokes

    # Collect bounding rects and properties for each drawing path
    rects: list[dict] = []
    for d in drawings:
        r = d.get("rect")
        if r is None:
            continue
        x0, y0, x1, y1 = r.x0, r.y0, r.x1, r.y1
        w, h = x1 - x0, y1 - y0
        if w < 2 or h < 2:
            continue  # skip hairlines / dots
        rects.append({
            "bbox": (x0, y0, x1, y1),
            "has_fill": d.get("fill") is not None,
            "has_stroke": d.get("color") is not None,
        })

    if len(rects) < min_paths:
        return []

    # Simple spatial clustering: merge overlapping / nearby drawing rects
    clusters = _merge_drawing_rects(rects, merge_gap=10.0)

    # Filter by minimum area
    results: list[DrawingCluster] = []
    for cl in clusters:
        bbox = cl["bbox"]
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area / page_area < ingest_settings.figure_min_area_ratio:
            continue
        if cl["count"] < min_paths:
            continue
        results.append(
            DrawingCluster(
                bbox=bbox,
                path_count=cl["count"],
                has_fill=cl["has_fill"],
                has_stroke=cl["has_stroke"],
            )
        )

    return results


def _merge_drawing_rects(
    rects: list[dict],
    merge_gap: float = 10.0,
) -> list[dict]:
    """Greedily merge drawing rects that overlap or are within *merge_gap* pts."""
    clusters: list[dict] = []
    for r in rects:
        bx = r["bbox"]
        merged = False
        for cl in clusters:
            cb = cl["bbox"]
            # Check if bounding boxes overlap or are within merge_gap
            if (bx[0] <= cb[2] + merge_gap and bx[2] >= cb[0] - merge_gap and
                    bx[1] <= cb[3] + merge_gap and bx[3] >= cb[1] - merge_gap):
                # Expand cluster bbox
                cl["bbox"] = (
                    min(cb[0], bx[0]),
                    min(cb[1], bx[1]),
                    max(cb[2], bx[2]),
                    max(cb[3], bx[3]),
                )
                cl["count"] += 1
                cl["has_fill"] = cl["has_fill"] or r["has_fill"]
                cl["has_stroke"] = cl["has_stroke"] or r["has_stroke"]
                merged = True
                break
        if not merged:
            clusters.append({
                "bbox": bx,
                "count": 1,
                "has_fill": r["has_fill"],
                "has_stroke": r["has_stroke"],
            })
    return clusters
