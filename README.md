# Stock Investment Research Assistant

A GenAI-powered assistant that answers investment research questions by combining **macroeconomic PDF documents** (unstructured data) with **stock CSV data** (structured data) through a REST API.

---

## Architecture

```
┌───────────┐         ┌─────────────┐
│  FastAPI   │◄───────►│ Orchestrator│
│  /api/ask  │         │  (classify  │
└───────────┘         │   + merge)  │
                       └──────┬──────┘
                  ┌───────────┴───────────┐
                  ▼                       ▼
         ┌──────────────┐       ┌──────────────┐
         │  ChromaDB    │       │   SQLite      │
         │ (PDF chunks) │       │ (stock data)  │
         └──────────────┘       └──────────────┘
                  ▲                       ▲
           RAG retrieval           Text-to-SQL
                  │                       │
      Ingestion Pipeline           CSV Loader
```

### PDF Ingestion Pipeline (SOTA, fully open-source)

```
PDF file
  │
  ├─ PyMuPDF ──► page render (PNG) + native text layer
  │
  ├─ Layout Analysis (heuristic / font-size + position)
  │    ├─ Headings / Paragraphs / Footnotes
  │    ├─ Tables  ──► pdfplumber (vector) │ EasyOCR (scanned fallback)
  │    ├─ Figures ──► crop + save PNG
  │    │    ├─ EasyOCR ──► axes / legends / titles
  │    │    ├─ Chart classifier (keyword heuristic)
  │    │    ├─ DePlot (google/deplot) ──► linearised data table
  │    │    └─ Flan-T5 (google/flan-t5-base) ──► chart-to-text summary
  │    └─ Captions ──► linked to nearest figure
  │
  ├─ Chunking (type-aware: text / table / figure / page-summary)
  ├─ Quality gates (length, OCR noise, dedup)
  ├─ Sentence-Transformer embeddings (all-MiniLM-L6-v2)
  └─ ChromaDB upsert (3 collections: text / tables / figures)
```

### Key Components

| Module | Responsibility |
|---|---|
| **app/main.py** | FastAPI app with lifespan-based data ingestion |
| **app/api/routes.py** | REST endpoints (`/api/ask`, `/api/health`) |
| **app/services/orchestrator.py** | Classifies queries, retrieves context, synthesises answers |
| **app/services/text_to_sql.py** | Converts natural-language → SQL via LLM |
| **app/services/csv_loader.py** | Loads stock CSV into SQLite |
| **app/services/llm.py** | OpenAI chat-completion wrapper |
| **app/ingestion/pipeline.py** | End-to-end PDF ingestion orchestrator |
| **app/ingestion/pdf_parser.py** | PDF rendering & text extraction (PyMuPDF) |
| **app/ingestion/layout.py** | Document layout analysis (headings, paragraphs, tables, figures) |
| **app/ingestion/ocr.py** | EasyOCR engine (page + region level) |
| **app/ingestion/tables.py** | Table extraction (pdfplumber + OCR fallback) |
| **app/ingestion/figures.py** | Figure detection, cropping, caption linking |
| **app/ingestion/charts.py** | Chart classification, chart-to-text (DePlot + Flan-T5) |
| **app/ingestion/chunker.py** | Type-aware chunking strategies |
| **app/ingestion/embeddings.py** | Sentence-transformer embeddings (local, open-source) |
| **app/ingestion/vectordb.py** | ChromaDB multi-collection store with hybrid retrieval |
| **app/ingestion/quality.py** | Quality gates & validation filters |
| **app/ingestion/schemas.py** | Pydantic models (DocumentChunk, TableChunk, FigureChunk) |
| **app/ingestion/config.py** | Pipeline-specific settings |
| **app/services/pdf_loader.py** | CLI for ingest / stats / query |

---

## Setup

### Prerequisites

- Python 3.11+
- An OpenAI API key (for the Q&A / text-to-SQL endpoint)
- macOS (Apple Silicon M-series) or Linux

### 1. Clone & create virtual environment

```bash
cd Pictet_investment_assistant
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 4. Add your data

| Data | Location | Format |
|---|---|---|
| Macro / strategy PDFs | `data/unstructured/` | `.pdf` |
| Stock CSV | `data/stocks.csv` | `.csv` with columns: Stock Symbol, Company Name, ISIN, Sector, Stock Price, Target Price, Dividend Yield |

### 5. Run the server

```bash
uvicorn app.main:app --reload
```

The API will be available at **http://localhost:8000**.  
Interactive docs at **http://localhost:8000/docs**.

On startup, the server automatically:
1. Loads the stock CSV into SQLite
2. Ingests all PDFs through the full pipeline (text + tables + figures + charts) into ChromaDB

### 5b. (Alternative) Run ingestion via CLI

```bash
# Ingest PDFs
python -m app.services.pdf_loader ingest

# Force re-ingestion
python -m app.services.pdf_loader ingest --force

# Check stats
python -m app.services.pdf_loader stats

# Test retrieval
python -m app.services.pdf_loader query "inflation outlook 2025"
```

---

## API Endpoints

### `POST /api/ask`

Submit a research question.

**Request body:**
```json
{
  "question": "What is the target price of Tesla, and how does it compare to the current macroeconomic trends?"
}
```

**Response:**
```json
{
  "answer": "The target price of Tesla is $35.20. According to the latest macroeconomic report...",
  "sources_used": ["structured", "unstructured"],
  "sql": "SELECT * FROM stocks WHERE company_name LIKE '%Tesla%'"
}
```

### `GET /api/health`

Returns system health and data-readiness status.

---

## Example Queries

| Query | Sources Used |
|---|---|
| *"What is the dividend yield of Johnson & Johnson?"* | structured |
| *"Which stocks in the Technology sector have the highest target price?"* | structured |
| *"What are the current macroeconomic trends affecting markets?"* | unstructured |
| *"What is the target price of Tesla, and how does it compare to the current macroeconomic trends?"* | structured + unstructured |
| *"Which healthcare stocks offer good value given the current macro outlook?"* | structured + unstructured |

---

## Tooling Choices (A–L)

| Step | Primary Tool | License | Alternative |
|---|---|---|---|
| **A. PDF parsing** | PyMuPDF (fitz) | AGPL-3.0 | pdfplumber, pikepdf |
| **B. Layout detection** | Heuristic (font-size + position) | — | layout-parser + Detectron2, surya |
| **C. OCR** | EasyOCR | Apache-2.0 | surya-ocr, Tesseract |
| **D. Table extraction** | pdfplumber (vector) + EasyOCR (scanned) | MIT / Apache-2.0 | Camelot, img2table |
| **E. Figure extraction** | PyMuPDF xref + layout crop | — | YOLOv8 figure detector |
| **F. Chart classification** | Keyword heuristic | — | nickmuchi/chart-type-classifier |
| **G. Chart-to-text** | google/deplot → google/flan-t5-base | Apache-2.0 | UniChart, ChartLlama |
| **H. Chart digitisation** | DePlot linearised table → JSON | Apache-2.0 | ChartOCR |
| **I. Entity extraction** | (via LLM at query time) | — | spaCy, FinBERT-NER |
| **J. Chunking** | Type-aware sliding window | — | semantic chunking (LangChain) |
| **K. Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | Apache-2.0 | BAAI/bge-base-en-v1.5, nomic-embed |
| **L. Vector DB** | ChromaDB (3 collections, cosine) | Apache-2.0 | Qdrant, Weaviate |

---

## Output Schemas

### DocumentChunk (text)
```json
{
  "text": "GDP grew by 3.2% in Q3 2024, driven by…",
  "metadata": {
    "doc_id": "a1b2c3…",
    "source_file": "WORLD ECONOMIC OUTLOOK.pdf",
    "page": 12,
    "block_type": "text",
    "section": "Global Growth Projections",
    "content_hash": "f4e5d6…",
    "created_at": "2025-02-16T10:30:00Z"
  },
  "citation": "WORLD ECONOMIC OUTLOOK.pdf, p.12"
}
```

### TableChunk
```json
{
  "markdown": "| Region | GDP Growth | Inflation |\n|---|---|---|\n| US | 2.8% | 3.1% |",
  "csv": "Region,GDP Growth,Inflation\nUS,2.8%,3.1%\n",
  "summary": "Table shows GDP growth and inflation by region.",
  "metadata": { "block_type": "table", "exhibit_id": "Table 2 (p.5)" },
  "citation": "report.pdf, p.5 – Table 2 (p.5)"
}
```

### FigureChunk
```json
{
  "caption": "Figure 3: Inflation trends in OECD countries",
  "ocr_text": "2020 2021 2022 2023 2024 CPI % 0 2 4 6 8",
  "chart_description": "Line chart showing inflation peaking in 2022 at ~8% then declining to ~3% by 2024.",
  "figure_type": "line_chart",
  "series_json": {"columns": ["Year", "CPI %"], "data": [["2020", "1.4"], ["2022", "8.0"]]},
  "image_path": "resources/a1b2c3/page_7_fig_1.png",
  "metadata": { "block_type": "figure", "exhibit_id": "Figure 3 (p.7)" },
  "citation": "report.pdf, p.7 – Figure 3 (p.7)"
}
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Configuration

All ingestion settings can be overridden via environment variables (prefixed `INGEST_`):

| Variable | Default | Description |
|---|---|---|
| `INGEST_DPI` | 200 | Page render resolution |
| `INGEST_OCR_GPU` | false | Enable MPS/CUDA for EasyOCR |
| `INGEST_TEXT_CHUNK_SIZE` | 800 | Characters per text chunk |
| `INGEST_TEXT_CHUNK_OVERLAP` | 200 | Overlap between text chunks |
| `INGEST_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model name |
| `INGEST_OCR_CONFIDENCE_THRESHOLD` | 0.40 | Minimum OCR box confidence |
| `INGEST_MIN_CHUNK_LENGTH` | 30 | Discard chunks shorter than this |

---

## Performance & Quality Gates

- **Deduplication**: Content-hash-based upsert prevents duplicate chunks on re-ingestion.
- **Quality filters**: Reject chunks that are too short, too long, or have high OCR noise (low alphanumeric ratio, repetitive content).
- **Batched embeddings**: Sentence-transformer encodes in configurable batch sizes.
- **MPS acceleration**: Apple Silicon GPU used for both EasyOCR and sentence-transformers when available.
- **Incremental ingestion**: File-hash tracking skips already-processed PDFs.

---

## Assumptions & Limitations

- **LLM provider**: OpenAI (GPT-4o) is used only for the Q&A / text-to-SQL endpoints. The entire ingestion pipeline is fully local and open-source.
- **Chart understanding**: DePlot + Flan-T5 provide good chart-to-text for common chart types. Complex multi-panel or 3D charts may produce lower-quality descriptions.
- **OCR accuracy**: EasyOCR works well on clean financial PDFs. Heavily degraded scans may need Tesseract or surya-ocr as an upgrade.
- **Layout detection**: The heuristic approach works for most digitally-born financial PDFs. For complex scanned layouts, a model-based detector (layout-parser, surya) would improve accuracy.
- **No authentication**: The API is open. Add API-key middleware for production.
- **Data refresh**: Data is ingested at startup. Restart the server or use the CLI after adding new documents.
