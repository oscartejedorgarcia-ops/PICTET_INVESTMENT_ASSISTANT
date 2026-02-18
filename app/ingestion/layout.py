"""
Document layout analysis / segmentation.

Strategy
--------
**Primary**: heuristic classification using PyMuPDF block metadata (font size,
position, image presence).  This is fast, dependency-free, and works well for
digitally-born financial PDFs.

**Optional upgrade path**: swap in a LayoutLM / DiT / YOLO-based detector
(e.g., ``layout-parser`` with Detectron2, or ``surya`` layout model) when GPU
budget allows.  The interface stays the same — a function that returns
``LayoutBlock`` objects with bounding boxes and labels.

Labels
------
HEADING, PARAGRAPH, TABLE, FIGURE, CAPTION, FOOTNOTE, HEADER, FOOTER, OTHER
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

from app.ingestion.pdf_parser import PageData, TextBlock

logger = logging.getLogger(__name__)


class LayoutLabel(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    HEADER = "header"
    FOOTER = "footer"
    OTHER = "other"


@dataclass
class LayoutBlock:
    label: LayoutLabel
    bbox: tuple[float, float, float, float]
    text: str
    confidence: float = 1.0
    page_number: int = 0


# ── Heuristic constants ─────────────────────────────────────────────────

_HEADING_FONT_RATIO = 1.25  # span font-size >= median * ratio → heading
_FOOTER_Y_RATIO = 0.92  # below 92% of page height → footer
_HEADER_Y_RATIO = 0.06  # above 6% of page height → header
_CAPTION_PATTERNS = re.compile(
    r"^(figure|fig\.?|table|exhibit|chart|graph|source|note)\s",
    re.IGNORECASE,
)


def _median_font_size(blocks: list[TextBlock]) -> float:
    sizes = [b.font_size for b in blocks if b.font_size > 0]
    if not sizes:
        return 12.0
    sizes.sort()
    mid = len(sizes) // 2
    return sizes[mid]


def analyse_layout(page: PageData) -> list[LayoutBlock]:
    """Return layout blocks for a single page using heuristics."""
    results: list[LayoutBlock] = []
    median_fs = _median_font_size(page.text_blocks)

    for tb in page.text_blocks:
        label = _classify_block(tb, median_fs, page.height)
        results.append(
            LayoutBlock(
                label=label,
                bbox=tb.bbox,
                text=tb.text,
                page_number=page.page_number,
            )
        )

    # Mark embedded-image regions as FIGURE
    for img in page.images:
        area_ratio = (img.width * img.height) / (page.width * page.height) if page.width and page.height else 0
        if area_ratio < 0.01:
            continue  # skip tiny icons
        results.append(
            LayoutBlock(
                label=LayoutLabel.FIGURE,
                bbox=img.bbox,
                text="",
                page_number=page.page_number,
            )
        )

    return results


def _classify_block(
    tb: TextBlock, median_fs: float, page_height: float
) -> LayoutLabel:
    """Classify a single text block."""
    y_mid = (tb.bbox[1] + tb.bbox[3]) / 2
    rel_y = y_mid / page_height if page_height else 0.5

    # Positional
    if rel_y < _HEADER_Y_RATIO:
        return LayoutLabel.HEADER
    if rel_y > _FOOTER_Y_RATIO:
        return LayoutLabel.FOOTNOTE

    # Caption patterns
    if _CAPTION_PATTERNS.match(tb.text.strip()):
        return LayoutLabel.CAPTION

    # Heading by font size / boldness
    if tb.font_size >= median_fs * _HEADING_FONT_RATIO or (
        tb.is_bold and len(tb.text.split()) < 15
    ):
        return LayoutLabel.HEADING

    return LayoutLabel.PARAGRAPH


def group_paragraphs(blocks: list[LayoutBlock]) -> list[LayoutBlock]:
    """Merge consecutive PARAGRAPH blocks on the same page into one."""
    if not blocks:
        return []
    merged: list[LayoutBlock] = []
    current: LayoutBlock | None = None

    for b in blocks:
        if b.label == LayoutLabel.PARAGRAPH:
            if current is not None:
                current.text += " " + b.text
                current.bbox = (
                    min(current.bbox[0], b.bbox[0]),
                    min(current.bbox[1], b.bbox[1]),
                    max(current.bbox[2], b.bbox[2]),
                    max(current.bbox[3], b.bbox[3]),
                )
            else:
                current = LayoutBlock(
                    label=LayoutLabel.PARAGRAPH,
                    bbox=b.bbox,
                    text=b.text,
                    page_number=b.page_number,
                )
        else:
            if current is not None:
                merged.append(current)
                current = None
            merged.append(b)

    if current is not None:
        merged.append(current)

    return merged
