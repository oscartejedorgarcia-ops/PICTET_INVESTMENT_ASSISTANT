"""
OCR engine – page-level and region-level text extraction.

Stack choice
------------
**EasyOCR** (Apache-2.0) is the primary engine:
  - Works well on M-series Macs (MPS-compatible PyTorch backend).
  - Good accuracy on financial docs with clean fonts.
  - Supports region crops (table cells, figure areas).

Fallback
--------
If EasyOCR is unavailable, fall back to PyMuPDF's built-in text layer
(which is already extracted by pdf_parser).

Alternative upgrade path: ``surya-ocr`` (Apache-2.0) for higher accuracy
on complex layouts.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

from PIL import Image

from app.ingestion.config import ingest_settings

logger = logging.getLogger(__name__)

# Suppress noisy "Using CPU" warning from EasyOCR
logging.getLogger("easyocr.easyocr").setLevel(logging.ERROR)

# Lazy-loaded singleton
_reader = None


@dataclass
class OCRBox:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1)


def _get_reader():
    """Lazy-initialise EasyOCR reader."""
    global _reader
    if _reader is not None:
        return _reader
    try:
        import easyocr  # noqa: F811

        _reader = easyocr.Reader(
            ingest_settings.ocr_languages,
            gpu=ingest_settings.ocr_gpu,
        )
        logger.info("EasyOCR reader initialised (gpu=%s).", ingest_settings.ocr_gpu)
    except ImportError:
        logger.warning("EasyOCR not installed – OCR will be unavailable.")
        _reader = None
    return _reader


def ocr_image_bytes(
    png_bytes: bytes,
    confidence_threshold: float | None = None,
) -> list[OCRBox]:
    """Run OCR on a PNG image (full page or cropped region).

    Returns a list of ``OCRBox`` sorted top-to-bottom, left-to-right.
    """
    reader = _get_reader()
    if reader is None:
        return []

    threshold = confidence_threshold or ingest_settings.ocr_confidence_threshold

    results = reader.readtext(png_bytes)
    boxes: list[OCRBox] = []
    for bbox_pts, text, conf in results:
        if conf < threshold:
            continue
        # bbox_pts is [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
        xs = [p[0] for p in bbox_pts]
        ys = [p[1] for p in bbox_pts]
        boxes.append(
            OCRBox(
                text=text.strip(),
                confidence=conf,
                bbox=(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
            )
        )

    # Sort top-to-bottom, left-to-right
    boxes.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
    return boxes


def ocr_to_text(boxes: list[OCRBox]) -> str:
    """Join OCR boxes into a single string preserving rough reading order."""
    return " ".join(b.text for b in boxes)


def crop_region(
    page_png: bytes,
    bbox: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
    dpi: int | None = None,
) -> bytes:
    """Crop a region from a rendered page PNG.

    ``bbox`` is in PDF coordinate space; the function maps it to pixel
    coordinates using the configured DPI.
    """
    dpi = dpi or ingest_settings.dpi
    scale = dpi / 72.0
    img = Image.open(io.BytesIO(page_png))
    x0 = int(bbox[0] * scale)
    y0 = int(bbox[1] * scale)
    x1 = int(bbox[2] * scale)
    y1 = int(bbox[3] * scale)
    # clamp
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.width, x1), min(img.height, y1)

    cropped = img.crop((x0, y0, x1, y1))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()


def ocr_region(
    page_png: bytes,
    bbox: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
) -> str:
    """Convenience: crop + OCR a region and return text."""
    region_bytes = crop_region(page_png, bbox, page_width, page_height)
    boxes = ocr_image_bytes(region_bytes)
    return ocr_to_text(boxes)
