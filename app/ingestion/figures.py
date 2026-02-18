"""
Figure detection, cropping, saving, and caption linking.

Strategy
--------
- Detect figures from embedded images (PyMuPDF xref) and from layout blocks
  labelled FIGURE.
- Crop the region from the rendered page pixmap.
- Save the crop to ``storage/resources/<doc_id>/page_<N>_fig_<M>.png``.
- Link to the nearest CAPTION block (spatially, on the same page).

Dependencies: PIL (Pillow), PyMuPDF (already imported via pdf_parser).
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.ingestion.config import ingest_settings
from app.ingestion.layout import LayoutBlock, LayoutLabel
from app.ingestion.pdf_parser import PageData

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFigure:
    page_number: int
    bbox: tuple[float, float, float, float]
    image_bytes: bytes  # PNG
    image_path: str  # relative to storage root
    caption: str
    figure_index: int  # 1-based within the page


def _find_nearest_caption(
    fig_bbox: tuple[float, float, float, float],
    captions: list[LayoutBlock],
) -> str:
    """Return the text of the caption block closest to the figure bbox."""
    if not captions:
        return ""
    fig_cy = (fig_bbox[1] + fig_bbox[3]) / 2
    fig_cx = (fig_bbox[0] + fig_bbox[2]) / 2

    best_caption = ""
    best_dist = float("inf")
    for cap in captions:
        cap_cy = (cap.bbox[1] + cap.bbox[3]) / 2
        cap_cx = (cap.bbox[0] + cap.bbox[2]) / 2
        dist = ((fig_cx - cap_cx) ** 2 + (fig_cy - cap_cy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_caption = cap.text
    return best_caption


def extract_figures(
    page: PageData,
    layout_blocks: list[LayoutBlock],
    doc_id: str,
) -> list[ExtractedFigure]:
    """Detect, crop, and save figure regions from a page."""
    if page.pixmap_bytes is None:
        return []

    page_img = Image.open(io.BytesIO(page.pixmap_bytes))
    scale = ingest_settings.dpi / 72.0

    # Collect figure bboxes from layout
    fig_bboxes: list[tuple[float, float, float, float]] = []
    for lb in layout_blocks:
        if lb.label == LayoutLabel.FIGURE:
            fig_bboxes.append(lb.bbox)

    # Also check embedded images not already covered
    for img in page.images:
        already = any(
            _iou(img.bbox, fb) > 0.3 for fb in fig_bboxes
        )
        if not already:
            area_ratio = (img.width * img.height) / max(page.width * page.height, 1)
            if area_ratio >= ingest_settings.figure_min_area_ratio:
                fig_bboxes.append(img.bbox)

    # Also check vector graphics clusters (charts/diagrams drawn with paths)
    for vg in page.vector_graphics:
        already = any(
            _iou(vg.bbox, fb) > 0.3 for fb in fig_bboxes
        )
        if not already:
            area = (vg.bbox[2] - vg.bbox[0]) * (vg.bbox[3] - vg.bbox[1])
            area_ratio = area / max(page.width * page.height, 1)
            if area_ratio >= ingest_settings.figure_min_area_ratio:
                logger.debug(
                    "Page %d: vector graphics cluster detected (%d paths, bbox=%s).",
                    page.page_number, vg.path_count, vg.bbox,
                )
                fig_bboxes.append(vg.bbox)

    # Caption blocks
    captions = [lb for lb in layout_blocks if lb.label == LayoutLabel.CAPTION]

    # Crop & save
    out_dir = Path(ingest_settings.resources_dir) / doc_id[:16]
    out_dir.mkdir(parents=True, exist_ok=True)

    figures: list[ExtractedFigure] = []
    for idx, bbox in enumerate(fig_bboxes, 1):
        x0 = max(0, int(bbox[0] * scale))
        y0 = max(0, int(bbox[1] * scale))
        x1 = min(page_img.width, int(bbox[2] * scale))
        y1 = min(page_img.height, int(bbox[3] * scale))
        if x1 - x0 < 20 or y1 - y0 < 20:
            continue

        cropped = page_img.crop((x0, y0, x1, y1))
        fname = f"page_{page.page_number}_fig_{idx}.png"
        save_path = out_dir / fname
        cropped.save(str(save_path), format="PNG")

        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        caption = _find_nearest_caption(bbox, captions)

        rel_path = str(save_path.relative_to(ingest_settings.storage_dir))

        figures.append(
            ExtractedFigure(
                page_number=page.page_number,
                bbox=bbox,
                image_bytes=image_bytes,
                image_path=rel_path,
                caption=caption,
                figure_index=idx,
            )
        )

    logger.debug("Page %d: extracted %d figures.", page.page_number, len(figures))
    return figures


def _iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Intersection-over-union of two bboxes."""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
