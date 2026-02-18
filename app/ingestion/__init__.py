"""
Ingestion pipeline for financial / macro-economic PDF documents.

Modules
-------
config       – Pipeline-specific settings (DPI, chunk sizes, thresholds …)
schemas      – Pydantic models for DocumentChunk, TableChunk, FigureChunk
pdf_parser   – PDF rendering & native text-layer extraction (PyMuPDF)
layout       – Document layout analysis / segmentation (DocTR / heuristics)
ocr          – OCR engine (EasyOCR) with region-level + page-level modes
tables       – Table extraction (pdfplumber vector + OCR fallback)
figures      – Figure crop, caption linking, chart classification
charts       – Chart-to-text summarisation & optional data digitisation
chunker      – Chunking strategies per block type (text, table, figure)
embeddings   – Sentence-transformer embeddings + optional BM25 sparse index
vectordb     – ChromaDB storage with hybrid retrieval helpers
pipeline     – End-to-end orchestrator wiring everything together
quality      – Quality gates, validators, confidence filters
"""
