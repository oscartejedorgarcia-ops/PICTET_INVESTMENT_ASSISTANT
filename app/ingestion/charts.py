"""
Chart analysis: classification, text extraction, chart-to-text summarisation,
and optional data digitisation.

Model choices (all open-source, Apache-2.0 / MIT)
--------------------------------------------------
- **Chart type classification**: lightweight CNN / ViT fine-tuned on chart
  images — we use a rule-based OCR keyword heuristic as default, with an
  optional ``chart_classifier_model`` hook for a HuggingFace classifier
  (e.g., ``nickmuchi/chart-type-classifier`` or similar).
- **Chart-to-text**: ``google/deplot`` (Apache-2.0) converts chart images to
  a linearised data table which we then feed to a local summarisation model
  (``google/flan-t5-base``, Apache-2.0) to produce a natural-language
  description.
- **OCR overlay**: EasyOCR to pull axis labels, titles, legends.

All models are lazy-loaded and cached to avoid GPU memory pressure on first
import.
"""

from __future__ import annotations

import logging
import os
import re
from enum import Enum
from pathlib import Path

from app.ingestion.config import ingest_settings
from app.ingestion.ocr import ocr_image_bytes, ocr_to_text
from app.ingestion.schemas import FigureType

logger = logging.getLogger(__name__)

# ── Lazy-loaded models ───────────────────────────────────────────────────

_deplot_processor = None
_deplot_model = None
_summariser = None
_local_font_path: str | None = None


def _find_local_font() -> str | None:
    """Locate a local TrueType font to avoid downloading from HuggingFace.

    The Pix2StructProcessor (DePlot) tries to download Arial.TTF from
    huggingface.co/ybelkada/fonts which can fail with connection errors.
    We resolve a local system font instead.
    """
    global _local_font_path
    if _local_font_path is not None:
        return _local_font_path

    # Common macOS / Linux / Windows font paths
    candidates = [
        # macOS
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
    ]
    for p in candidates:
        if os.path.isfile(p):
            _local_font_path = p
            logger.info("Using local font for DePlot: %s", p)
            return p

    logger.warning("No local font found – DePlot will attempt HuggingFace download.")
    return None


def _load_deplot():
    """Lazy-load DePlot (chart → linearised table)."""
    global _deplot_processor, _deplot_model
    if _deplot_model is not None:
        return
    try:
        from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration

        model_name = "google/deplot"
        logger.info("Loading DePlot model '%s' …", model_name)
        _deplot_processor = Pix2StructProcessor.from_pretrained(model_name)
        _deplot_model = Pix2StructForConditionalGeneration.from_pretrained(model_name)
        logger.info("DePlot model loaded.")
    except Exception as exc:
        logger.warning("Could not load DePlot: %s — chart digitisation disabled.", exc)


def _load_summariser():
    """Lazy-load a small text summarisation / instruction model."""
    global _summariser
    if _summariser is not None:
        return
    try:
        from transformers import pipeline

        _summariser = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=ingest_settings.chart_description_max_tokens,
        )
        logger.info("Flan-T5 summariser loaded.")
    except Exception as exc:
        logger.warning("Could not load summariser: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# Chart type classification
# ═══════════════════════════════════════════════════════════════════════════

_KEYWORD_MAP: list[tuple[re.Pattern, FigureType]] = [
    (re.compile(r"\bpie\b", re.I), FigureType.PIE_CHART),
    (re.compile(r"\bdonut\b", re.I), FigureType.DONUT_CHART),
    (re.compile(r"\bscatter\b", re.I), FigureType.SCATTER_CHART),
    (re.compile(r"\bbubble\b", re.I), FigureType.BUBBLE_CHART),
    (re.compile(r"\bcandle|ohlc\b", re.I), FigureType.CANDLESTICK),
    (re.compile(r"\bwaterfall\b", re.I), FigureType.WATERFALL),
    (re.compile(r"\bheat\s*map\b", re.I), FigureType.HEATMAP),
    (re.compile(r"\bbox\b.*\bwhisker|box\s*plot\b", re.I), FigureType.BOX_WHISKER),
    (re.compile(r"\bhistogram\b", re.I), FigureType.HISTOGRAM),
    (re.compile(r"\bnetwork\b", re.I), FigureType.NETWORK_GRAPH),
    (re.compile(r"\bparallel\s*coord", re.I), FigureType.PARALLEL_COORDINATES),
    (re.compile(r"\bstacked\s*(bar|column)\b", re.I), FigureType.STACKED_BAR_CHART),
    (re.compile(r"\bbar\b|\bcolumn\b", re.I), FigureType.BAR_CHART),
    (re.compile(r"\barea\b", re.I), FigureType.AREA_CHART),
    (re.compile(r"\bline\b.*\bline\b|\bmulti.?line\b", re.I), FigureType.MULTI_LINE_CHART),
    (re.compile(r"\bline\b", re.I), FigureType.LINE_CHART),
]


def classify_chart_type(caption: str, ocr_text: str) -> FigureType:
    """Classify chart type from caption + OCR text using keyword heuristics."""
    combined = f"{caption} {ocr_text}"
    for pattern, ftype in _KEYWORD_MAP:
        if pattern.search(combined):
            return ftype
    return FigureType.UNKNOWN


# ═══════════════════════════════════════════════════════════════════════════
# Chart OCR overlay (axes, legends, titles)
# ═══════════════════════════════════════════════════════════════════════════

def extract_chart_text(image_bytes: bytes) -> str:
    """Run OCR on a chart image and return concatenated text."""
    boxes = ocr_image_bytes(image_bytes, confidence_threshold=0.30)
    return ocr_to_text(boxes)


# ═══════════════════════════════════════════════════════════════════════════
# Chart-to-text summarisation
# ═══════════════════════════════════════════════════════════════════════════

def chart_to_text(
    image_bytes: bytes,
    caption: str = "",
    ocr_text: str = "",
) -> str:
    """Generate a natural-language description of a chart image.

    Pipeline: DePlot (image → linearised table) → Flan-T5 (table → text).
    Falls back to OCR-only summary if models are unavailable.
    """
    linearised = _deplot_image(image_bytes)

    if linearised:
        description = _summarise_linearised(linearised, caption)
        if description:
            return description

    # Fallback: compose from OCR + caption
    parts: list[str] = []
    if caption:
        parts.append(f"This figure is captioned: \"{caption}\".")
    if ocr_text:
        parts.append(f"Text visible in the chart: {ocr_text}")
    return " ".join(parts) if parts else ""


def _deplot_image(image_bytes: bytes) -> str:
    """Use DePlot to convert chart image → linearised data table."""
    _load_deplot()
    if _deplot_model is None or _deplot_processor is None:
        return ""
    try:
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        header_text = "Generate underlying data table of the figure below:"

        # Render the header onto the image via the image processor,
        # passing font_path directly (the top-level Processor ignores it).
        font = _find_local_font()
        img_kwargs: dict = {}
        if font:
            img_kwargs["font_path"] = font
        img_out = _deplot_processor.image_processor(
            images=img,
            header_text=header_text,
            return_tensors="pt",
            **img_kwargs,
        )
        # Pix2Struct bakes the text prompt into the image as a rendered
        # header, so we only pass image outputs — no separate tokenizer
        # input_ids (that would cause a tensor shape mismatch).
        inputs = {
            "flattened_patches": img_out.get(
                "flattened_patches", img_out.get("pixel_values")
            ),
            "attention_mask": img_out.get("attention_mask"),
        }

        predictions = _deplot_model.generate(**inputs, max_new_tokens=400)
        result = _deplot_processor.decode(predictions[0], skip_special_tokens=True)
        return result.strip()
    except Exception as exc:
        logger.warning("DePlot inference failed: %s", exc)
        return ""


def _summarise_linearised(linearised: str, caption: str) -> str:
    """Turn a DePlot linearised table into a prose summary."""
    _load_summariser()
    if _summariser is None:
        return ""
    try:
        prompt = (
            f"Describe the following chart data in 2-3 sentences, "
            f"highlighting trends, comparisons, and key values.\n"
            f"Caption: {caption}\nData:\n{linearised}"
        )
        result = _summariser(prompt)
        return result[0]["generated_text"].strip()
    except Exception as exc:
        logger.warning("Summarisation failed: %s", exc)
        return ""


# ═══════════════════════════════════════════════════════════════════════════
# Optional: chart data digitisation
# ═══════════════════════════════════════════════════════════════════════════

def digitise_chart(image_bytes: bytes) -> dict | None:
    """Attempt to extract structured data series from a chart image.

    Returns a dict like ``{"columns": [...], "data": [...]}`` or None.
    Uses DePlot's linearised table output and parses it.
    """
    linearised = _deplot_image(image_bytes)
    if not linearised:
        return None
    return _parse_linearised(linearised)


def _parse_linearised(text: str) -> dict | None:
    """Parse DePlot's linearised table format into structured JSON.

    DePlot outputs rows like: ``Title | col1 | col2 <0x0A> val1 | val2 | val3``
    """
    lines = text.replace("<0x0A>", "\n").strip().split("\n")
    if len(lines) < 2:
        return None
    rows = [line.split("|") for line in lines]
    rows = [[cell.strip() for cell in row] for row in rows]
    return {"columns": rows[0], "data": rows[1:]}
