"""
Tests for the ingestion pipeline modules.

Run: python -m pytest tests/ -v
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════════
# Schema tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSchemas:
    def test_document_chunk_creation(self):
        from app.ingestion.schemas import BlockType, ChunkMetadata, Citation, DocumentChunk

        chunk = DocumentChunk(
            text="GDP grew by 3.2% in Q3 2024.",
            metadata=ChunkMetadata(
                doc_id="abc123",
                source_file="report.pdf",
                page=5,
                block_type=BlockType.TEXT,
                section="Economic Growth",
                content_hash=hashlib.sha256(b"test").hexdigest(),
            ),
            citation=Citation(
                source_file="report.pdf",
                page=5,
                block_type=BlockType.TEXT,
            ),
        )
        assert chunk.text == "GDP grew by 3.2% in Q3 2024."
        assert chunk.metadata.page == 5
        assert "report.pdf" in str(chunk.citation)

    def test_table_chunk_creation(self):
        from app.ingestion.schemas import BlockType, ChunkMetadata, Citation, TableChunk

        chunk = TableChunk(
            markdown="| A | B |\n|---|---|\n| 1 | 2 |",
            csv="A,B\n1,2\n",
            metadata=ChunkMetadata(
                doc_id="abc123",
                source_file="report.pdf",
                page=3,
                block_type=BlockType.TABLE,
                exhibit_id="Table 1",
                content_hash=hashlib.sha256(b"tbl").hexdigest(),
            ),
            citation=Citation(
                source_file="report.pdf",
                page=3,
                block_type=BlockType.TABLE,
                exhibit_id="Table 1",
            ),
        )
        assert chunk.markdown.startswith("| A")
        assert "Table 1" in str(chunk.citation)

    def test_figure_chunk_creation(self):
        from app.ingestion.schemas import (
            BlockType, ChunkMetadata, Citation, FigureChunk, FigureType,
        )

        chunk = FigureChunk(
            caption="Figure 1: GDP Growth",
            chart_description="Line chart showing GDP growth trending upward.",
            figure_type=FigureType.LINE_CHART,
            image_path="resources/abc/page_1_fig_1.png",
            metadata=ChunkMetadata(
                doc_id="abc123",
                source_file="report.pdf",
                page=7,
                block_type=BlockType.FIGURE,
                exhibit_id="Figure 1",
                content_hash=hashlib.sha256(b"fig").hexdigest(),
            ),
            citation=Citation(
                source_file="report.pdf",
                page=7,
                block_type=BlockType.FIGURE,
                exhibit_id="Figure 1",
            ),
        )
        assert chunk.figure_type == FigureType.LINE_CHART
        assert chunk.image_path.endswith(".png")

    def test_chunk_to_text(self):
        from app.ingestion.schemas import (
            BlockType, ChunkMetadata, Citation, DocumentChunk,
            FigureChunk, FigureType, TableChunk, chunk_to_text,
        )

        doc = DocumentChunk(
            text="Hello world",
            metadata=ChunkMetadata(
                doc_id="x", source_file="f.pdf", page=1,
                block_type=BlockType.TEXT,
                content_hash=hashlib.sha256(b"hw").hexdigest(),
            ),
            citation=Citation(source_file="f.pdf", page=1, block_type=BlockType.TEXT),
        )
        assert chunk_to_text(doc) == "Hello world"

        tbl = TableChunk(
            markdown="| A |\n|---|\n| 1 |",
            summary="One column table.",
            metadata=ChunkMetadata(
                doc_id="x", source_file="f.pdf", page=1,
                block_type=BlockType.TABLE,
                content_hash=hashlib.sha256(b"t").hexdigest(),
            ),
            citation=Citation(source_file="f.pdf", page=1, block_type=BlockType.TABLE),
        )
        text = chunk_to_text(tbl)
        assert "| A |" in text
        assert "Summary" in text

    def test_metadata_dict_serialisation(self):
        from app.ingestion.schemas import (
            BlockType, ChunkMetadata, Citation, DocumentChunk,
            chunk_to_metadata_dict,
        )

        chunk = DocumentChunk(
            text="test",
            metadata=ChunkMetadata(
                doc_id="x", source_file="f.pdf", page=1,
                block_type=BlockType.TEXT,
                entities=["GDP", "CPI"],
                content_hash="abc",
            ),
            citation=Citation(source_file="f.pdf", page=1, block_type=BlockType.TEXT),
        )
        d = chunk_to_metadata_dict(chunk)
        # entities must be flattened to string for ChromaDB
        assert isinstance(d["entities"], str)
        assert "GDP" in d["entities"]
        assert d["block_type"] == "text"


# ═══════════════════════════════════════════════════════════════════════════
# Layout tests
# ═══════════════════════════════════════════════════════════════════════════

class TestLayout:
    def test_heading_detection(self):
        from app.ingestion.layout import LayoutLabel, _classify_block
        from app.ingestion.pdf_parser import TextBlock

        block = TextBlock(
            text="Executive Summary",
            bbox=(50, 100, 400, 120),
            font_size=18.0,
            is_bold=True,
        )
        label = _classify_block(block, median_fs=11.0, page_height=800)
        assert label == LayoutLabel.HEADING

    def test_paragraph_detection(self):
        from app.ingestion.layout import LayoutLabel, _classify_block
        from app.ingestion.pdf_parser import TextBlock

        block = TextBlock(
            text="The economy grew by 3.2% in the third quarter of 2024.",
            bbox=(50, 200, 500, 220),
            font_size=11.0,
            is_bold=False,
        )
        label = _classify_block(block, median_fs=11.0, page_height=800)
        assert label == LayoutLabel.PARAGRAPH

    def test_caption_detection(self):
        from app.ingestion.layout import LayoutLabel, _classify_block
        from app.ingestion.pdf_parser import TextBlock

        block = TextBlock(
            text="Figure 3: Inflation trends in OECD countries",
            bbox=(50, 600, 500, 615),
            font_size=9.0,
            is_bold=False,
        )
        label = _classify_block(block, median_fs=11.0, page_height=800)
        assert label == LayoutLabel.CAPTION

    def test_group_paragraphs(self):
        from app.ingestion.layout import LayoutBlock, LayoutLabel, group_paragraphs

        blocks = [
            LayoutBlock(label=LayoutLabel.PARAGRAPH, bbox=(0, 0, 100, 20), text="First."),
            LayoutBlock(label=LayoutLabel.PARAGRAPH, bbox=(0, 20, 100, 40), text="Second."),
            LayoutBlock(label=LayoutLabel.HEADING, bbox=(0, 50, 100, 70), text="Title"),
            LayoutBlock(label=LayoutLabel.PARAGRAPH, bbox=(0, 80, 100, 100), text="Third."),
        ]
        merged = group_paragraphs(blocks)
        assert len(merged) == 3  # merged para, heading, para
        assert "First." in merged[0].text
        assert "Second." in merged[0].text


# ═══════════════════════════════════════════════════════════════════════════
# Table formatting tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTables:
    def test_rows_to_markdown(self):
        from app.ingestion.tables import _rows_to_markdown

        rows = [["Name", "Value"], ["GDP", "3.2%"], ["CPI", "2.1%"]]
        md = _rows_to_markdown(rows)
        assert "| Name | Value |" in md
        assert "| GDP | 3.2% |" in md
        assert "---" in md

    def test_rows_to_csv(self):
        from app.ingestion.tables import _rows_to_csv

        rows = [["A", "B"], ["1", "2"]]
        csv_text = _rows_to_csv(rows)
        assert "A,B" in csv_text
        assert "1,2" in csv_text


# ═══════════════════════════════════════════════════════════════════════════
# Quality gate tests
# ═══════════════════════════════════════════════════════════════════════════

class TestQuality:
    def test_reject_short_text(self):
        from app.ingestion.quality import validate_text_chunk
        from app.ingestion.schemas import BlockType, ChunkMetadata, Citation, DocumentChunk

        chunk = DocumentChunk(
            text="Hi",
            metadata=ChunkMetadata(
                doc_id="x", source_file="f.pdf", page=1,
                block_type=BlockType.TEXT, content_hash="h",
            ),
            citation=Citation(source_file="f.pdf", page=1, block_type=BlockType.TEXT),
        )
        ok, reason = validate_text_chunk(chunk)
        assert not ok
        assert "short" in reason.lower()

    def test_reject_garbage_text(self):
        from app.ingestion.quality import validate_text_chunk
        from app.ingestion.schemas import BlockType, ChunkMetadata, Citation, DocumentChunk

        chunk = DocumentChunk(
            text="##$%^&*()!@#$%^&*()!@#$%^&*()!@#$%^&*()!@#$%^&*()!!",
            metadata=ChunkMetadata(
                doc_id="x", source_file="f.pdf", page=1,
                block_type=BlockType.TEXT, content_hash="h",
            ),
            citation=Citation(source_file="f.pdf", page=1, block_type=BlockType.TEXT),
        )
        ok, reason = validate_text_chunk(chunk)
        assert not ok
        assert "alphanumeric" in reason.lower()

    def test_accept_good_text(self):
        from app.ingestion.quality import validate_text_chunk
        from app.ingestion.schemas import BlockType, ChunkMetadata, Citation, DocumentChunk

        chunk = DocumentChunk(
            text="The global economy is projected to grow by 3.1% in 2025, driven by strong performance in emerging markets.",
            metadata=ChunkMetadata(
                doc_id="x", source_file="f.pdf", page=1,
                block_type=BlockType.TEXT, content_hash="h",
            ),
            citation=Citation(source_file="f.pdf", page=1, block_type=BlockType.TEXT),
        )
        ok, reason = validate_text_chunk(chunk)
        assert ok

    def test_filter_chunks(self):
        from app.ingestion.quality import filter_chunks
        from app.ingestion.schemas import BlockType, ChunkMetadata, Citation, DocumentChunk

        good = DocumentChunk(
            text="The Federal Reserve raised rates by 25 basis points in December.",
            metadata=ChunkMetadata(
                doc_id="x", source_file="f.pdf", page=1,
                block_type=BlockType.TEXT, content_hash="good",
            ),
            citation=Citation(source_file="f.pdf", page=1, block_type=BlockType.TEXT),
        )
        bad = DocumentChunk(
            text="x",
            metadata=ChunkMetadata(
                doc_id="x", source_file="f.pdf", page=1,
                block_type=BlockType.TEXT, content_hash="bad",
            ),
            citation=Citation(source_file="f.pdf", page=1, block_type=BlockType.TEXT),
        )
        result = filter_chunks([good, bad])
        assert len(result) == 1
        assert result[0].text == good.text


# ═══════════════════════════════════════════════════════════════════════════
# Chart classification tests
# ═══════════════════════════════════════════════════════════════════════════

class TestChartClassification:
    def test_line_chart(self):
        from app.ingestion.charts import classify_chart_type
        from app.ingestion.schemas import FigureType

        ft = classify_chart_type("Figure 1: Line chart of GDP", "")
        assert ft == FigureType.LINE_CHART

    def test_bar_chart(self):
        from app.ingestion.charts import classify_chart_type
        from app.ingestion.schemas import FigureType

        ft = classify_chart_type("Bar chart: sector returns", "")
        assert ft == FigureType.BAR_CHART

    def test_pie_chart(self):
        from app.ingestion.charts import classify_chart_type
        from app.ingestion.schemas import FigureType

        ft = classify_chart_type("", "pie 45% 30% 25%")
        assert ft == FigureType.PIE_CHART

    def test_unknown(self):
        from app.ingestion.charts import classify_chart_type
        from app.ingestion.schemas import FigureType

        ft = classify_chart_type("An illustration of flows", "")
        assert ft == FigureType.UNKNOWN


# ═══════════════════════════════════════════════════════════════════════════
# Chunker tests
# ═══════════════════════════════════════════════════════════════════════════

class TestChunker:
    def test_text_chunking(self):
        from app.ingestion.chunker import chunk_text_blocks
        from app.ingestion.layout import LayoutBlock, LayoutLabel

        blocks = [
            LayoutBlock(
                label=LayoutLabel.PARAGRAPH,
                bbox=(0, 0, 100, 20),
                text="Word " * 200,  # ~1000 chars
                page_number=1,
            ),
        ]
        chunks = chunk_text_blocks(blocks, "doc1", "test.pdf", 1)
        assert len(chunks) >= 1
        assert all(c.metadata.source_file == "test.pdf" for c in chunks)

    def test_table_chunking(self):
        from app.ingestion.chunker import chunk_tables
        from app.ingestion.tables import ExtractedTable

        tbl = ExtractedTable(
            page_number=2,
            bbox=(0, 0, 100, 100),
            rows=[["A", "B"], ["1", "2"], ["3", "4"]],
            markdown="| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |",
            csv_text="A,B\n1,2\n3,4\n",
            extraction_method="pdfplumber",
        )
        chunks = chunk_tables([tbl], "doc1", "test.pdf")
        assert len(chunks) == 1
        assert chunks[0].metadata.exhibit_id.startswith("Table")


# ═══════════════════════════════════════════════════════════════════════════
# PDF parser tests (unit, no real PDF needed)
# ═══════════════════════════════════════════════════════════════════════════

class TestPdfParser:
    def test_compute_file_hash(self, tmp_path):
        from app.ingestion.pdf_parser import compute_file_hash

        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h = compute_file_hash(f)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex
