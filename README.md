# Stock Investment Research Assistant

A GenAI-powered assistant that answers investment research questions by combining **macroeconomic PDF reports** (unstructured) with **equity/stock data from CSV/Excel** (structured). It exposes a single FastAPI endpoint that classifies each question, retrieves context from the right data store, and synthesises an answer using a local Ollama LLM.

### Key capabilities

- **RAG over PDFs** — full ingestion pipeline (text, tables, figures, charts) into ChromaDB with sentence-transformer embeddings.
- **Text-to-SQL over equities data** — natural-language questions translated to SQL against a SQLite database, augmented by ChromaDB-stored column descriptions and taxonomy embeddings.
- **Automatic query routing** — the LLM classifies every question as `structured`, `unstructured`, or both, then merges results before responding.
- **Citations** — every retrieved chunk carries a citation (`source_file, page, exhibit_id`) that is included in prompt context.
- **Fully local ingestion** — the entire PDF pipeline runs without external API calls (EasyOCR, sentence-transformers, DePlot, Flan-T5).

### In scope

- Single-endpoint Q&A (`POST /api/ask`) with source attribution.
- PDF ingestion: text extraction, layout analysis, table extraction, figure/chart extraction with OCR, chart-to-text (DePlot → Flan-T5), type-aware chunking, quality gates, ChromaDB upsert.
- CSV/Excel ingestion: load into SQLite, LLM-generated column descriptions stored in a metadata table and as ChromaDB embeddings, taxonomy embeddings for semantic SQL context retrieval.
- CLI for manual ingestion, stats, and vector-store queries.
- Docker Compose setup with Ollama sidecar.

### Out of scope (not implemented)

- Authentication / authorisation.
- Streaming responses.
- Multi-turn conversation memory.
- Fine-tuned or hosted LLM — uses Ollama locally (default model: `llama3.2`).

---

## Architecture

```
┌────────────┐        ┌──────────────┐
│  FastAPI    │◄──────►│ Orchestrator │
│  /api/ask   │        │ classify +   │
│  /api/health│        │ merge        │
└────────────┘        └──────┬───────┘
                    ┌────────┴────────┐
                    ▼                 ▼
           ┌──────────────┐  ┌──────────────┐
           │  ChromaDB    │  │   SQLite      │
           │ (PDF chunks) │  │ (stock data)  │
           └──────────────┘  └──────────────┘
                    ▲                 ▲
             RAG retrieval     Text-to-SQL
                    │                 │
         Ingestion Pipeline    CSV/Excel Loader
                    │                 │
            data/unstructured/  data/structured/
```

### Data flow

1. **Startup** (`app/main.py` → `lifespan`): checks if SQLite and ChromaDB already contain data. If empty, runs CSV/Excel load and PDF ingestion automatically.
2. **Question received** (`POST /api/ask`): the orchestrator calls the LLM to classify the query into `structured` / `unstructured` / both.
3. **Structured path**: `text_to_sql.py` builds context by querying two ChromaDB collections (`stocks_db_structure`, `stocks_taxonomy`), asks the LLM to generate SQL, executes it against SQLite, and returns results.
4. **Unstructured path**: queries ChromaDB collections (`ingest_text`, `ingest_tables`, `ingest_figures`) using sentence-transformer embeddings, returns top-k chunks with citations.
5. **Synthesis**: both context blocks are assembled into a single prompt. The LLM produces a final answer returned to the client.

---

## PDF Ingestion Pipeline

```
PDF file
  │
  ├─ PyMuPDF ──► page render (PNG @ configurable DPI) + native text layer
  │
  ├─ Layout analysis (font-size + position heuristics)
  │    ├─ Headings / Paragraphs / Footnotes
  │    ├─ Tables  ──► pdfplumber (vector) │ EasyOCR (scanned fallback)
  │    ├─ Figures ──► crop + save PNG
  │    │    ├─ EasyOCR ──► axes / legends / titles
  │    │    ├─ Chart classifier (keyword heuristic)
  │    │    ├─ DePlot (google/deplot) ──► linearised data table
  │    │    └─ Flan-T5 (google/flan-t5-base) ──► chart-to-text summary
  │    └─ Captions ──► linked to nearest figure (spatial proximity)
  │
  ├─ Chunking (type-aware: text / table / figure / page-summary)
  ├─ Quality gates (min/max length, OCR noise ratio, repetitive content, dedup)
  ├─ Sentence-transformer embeddings (all-MiniLM-L6-v2)
  └─ ChromaDB upsert (3 collections: ingest_text, ingest_tables, ingest_figures)
```

### Step-by-step

| Step | What happens | Module |
|---|---|---|
| 1. Discovery | Glob `data/unstructured/*.pdf`; skip already-ingested files via SHA-256 file hash | `pipeline.py` → `ingest_folder()` |
| 2. Parsing | PyMuPDF renders each page as PNG and extracts native text spans with font metadata | `pdf_parser.py` → `parse_pdf()`, `PageData` |
| 3. Layout | Heuristic classifier labels each text block as HEADING / PARAGRAPH / CAPTION / FOOTNOTE / HEADER / FOOTER based on font size, position, and regex patterns. Embedded images flagged as FIGURE | `layout.py` → `analyse_layout()`, `LayoutLabel` |
| 4. OCR fallback | Pages without a usable text layer are sent to EasyOCR | `ocr.py` → `ocr_image_bytes()` |
| 5. Tables | pdfplumber extracts vector-drawn tables; OCR-based row clustering used as fallback for scanned tables | `tables.py` → `extract_tables_pdfplumber()`, `extract_table_ocr()` |
| 6. Figures | Regions labelled FIGURE are cropped from the rendered pixmap and saved to `storage/resources/<doc_id>/` | `figures.py` → `extract_figures()` |
| 7. Charts | Each figure is OCR'd for axis/legend text, classified by keyword heuristic, described via DePlot → Flan-T5, and optionally digitised to JSON | `charts.py` → `classify_chart_type()`, `chart_to_text()`, `digitise_chart()` |
| 8. Chunking | Text: sliding window (450 chars, 50 overlap). Tables: one chunk per table. Figures: one chunk per figure. Optional page-summary chunks | `chunker.py` → `chunk_text_blocks()`, `chunk_tables()`, `chunk_figures()`, `create_page_summary()` |
| 9. Quality gates | Discard chunks < 30 chars, > 8 000 chars, low alphanumeric ratio, or repetitive content | `quality.py` → `filter_chunks()` |
| 10. Embedding | `all-MiniLM-L6-v2` (384-dim), batched, with MPS acceleration when available | `embeddings.py` → `embed_texts()` |
| 11. Storage | Content-hash-based upsert into ChromaDB (cosine distance) | `vectordb.py` → `IngestVectorStore.upsert_chunks()` |

---

## CSV/Excel Ingestion (Structured Data)

| Step | What happens | Module |
|---|---|---|
| 1. Load | Read CSV or Excel (`.xlsx`) via Pandas | `csv_loader.py` → `load_csv_to_sqlite()` |
| 2. Normalise | Column names: strip, lowercase, replace spaces with underscores | same function |
| 3. Write to SQLite | `df.to_sql("stocks", …, if_exists="replace")` into `storage/sqlite/stocks.db` | same function |
| 4. Column descriptions | For each column, the LLM generates a one-sentence description from sample values; stored in a `column_metadata` SQLite table | `_generate_column_descriptions_from_db()`, `_store_column_metadata()` |
| 5. Column embeddings | Column descriptions embedded and upserted to ChromaDB collection `stocks_db_structure` | `_store_embeddings_in_chromadb()` |
| 6. Taxonomy embeddings | Taxonomy fields (company, sector, industry, region) are embedded row-by-row into ChromaDB collection `stocks_taxonomy` | `_store_taxonomy_embeddings_in_chromadb()` |

At query time, `text_to_sql.py` queries both ChromaDB collections to build context (column names + sample taxonomy values), then asks the LLM to generate a SQL `SELECT` statement which is executed read-only against SQLite.

---

## End-to-End Question Answering Flow

```
User question
  │
  ▼
1. POST /api/ask            (app/api/routes.py → ask_question)
  │
  ▼
2. Classify query            (orchestrator.py → _classify)
   LLM returns {"sources": ["structured", "unstructured"]}
   Fallback: both sources if JSON parsing fails
  │
  ├─► structured             (orchestrator.py → _retrieve_structured)
  │    └─ text_to_sql.py → answer_with_sql()
  │         ├─ Build context from ChromaDB (db_structure + taxonomy)
  │         ├─ LLM generates SQL
  │         └─ Execute SQL against SQLite → results
  │
  ├─► unstructured           (orchestrator.py → _retrieve_unstructured)
  │    └─ vectordb.py → IngestVectorStore.query()
  │         └─ Sentence-transformer embed query → cosine search
  │            across ingest_text / ingest_tables / ingest_figures
  │            → top-5 chunks with citations
  │
  ▼
3. Synthesise answer         (orchestrator.py → answer)
   Merge structured + unstructured context
   LLM produces final answer (executive-summary style)
  │
  ▼
4. Return response           {"answer": "…", "sources_used": […], "sql": "…"}
```

---

## Main Modules

### `app/` — Application core

| File | Purpose | Key entrypoints |
|---|---|---|
| `main.py` | FastAPI app factory; lifespan hook runs data ingestion on startup | `lifespan()`, `app` |
| `config.py` | Central settings from `.env` (Ollama URL, paths, model name) | `Settings`, `settings` |
| `api/routes.py` | REST endpoints | `ask_question()`, `health_check()` |
| `services/orchestrator.py` | Query classification → retrieval → LLM synthesis | `answer()`, `_classify()` |
| `services/text_to_sql.py` | NL → SQL via LLM with ChromaDB-augmented context | `answer_with_sql()`, `_build_context()` |
| `services/csv_loader.py` | Load CSV/Excel → SQLite; generate column & taxonomy embeddings | `load_csv_to_sqlite()`, `run_sql()` |
| `services/llm.py` | Thin wrapper around Ollama's OpenAI-compatible API | `chat()` |
| `services/pdf_loader.py` | CLI (`ingest`, `stats`, `query` subcommands) | `main()` |

### `app/ingestion/` — PDF ingestion pipeline

| File | Purpose | Key entrypoints |
|---|---|---|
| `pipeline.py` | Orchestrates the full per-file ingestion flow | `ingest_folder()`, `ingest_file()` |
| `pdf_parser.py` | PyMuPDF page rendering + text/image extraction | `parse_pdf()`, `PageData`, `compute_file_hash()` |
| `layout.py` | Heuristic layout segmentation (font size + position) | `analyse_layout()`, `group_paragraphs()`, `LayoutLabel` |
| `ocr.py` | EasyOCR engine for page-level and region-level OCR | `ocr_image_bytes()`, `ocr_to_text()` |
| `tables.py` | Table extraction (pdfplumber + OCR fallback) | `extract_tables_pdfplumber()`, `extract_table_ocr()` |
| `figures.py` | Figure detection, cropping, saving, caption linking | `extract_figures()`, `ExtractedFigure` |
| `charts.py` | Chart classification, DePlot → Flan-T5 chart-to-text, digitisation | `classify_chart_type()`, `chart_to_text()`, `digitise_chart()` |
| `chunker.py` | Type-aware chunking (text, table, figure, page-summary) | `chunk_text_blocks()`, `chunk_tables()`, `chunk_figures()` |
| `embeddings.py` | Sentence-transformer embedding (lazy-loaded, MPS-aware) | `embed_texts()`, `embed_query()` |
| `vectordb.py` | ChromaDB multi-collection store with cross-collection retrieval | `IngestVectorStore` |
| `quality.py` | Quality gates (length, noise, repetition) | `filter_chunks()` |
| `schemas.py` | Pydantic models: `DocumentChunk`, `TableChunk`, `FigureChunk`, `ChunkMetadata` | `chunk_to_text()`, `chunk_to_metadata_dict()` |
| `config.py` | Pipeline settings with `INGEST_` env-var prefix | `IngestSettings`, `ingest_settings` |

---

## Setup

### 7.1 Prerequisites

- **Python 3.11+**
- **Ollama** running locally (default: `http://localhost:11434`) with a pulled model (default: `llama3.2`)
- **Docker & Docker Compose** (optional — for containerised deployment)
- macOS (Apple Silicon) or Linux

> No OpenAI API key is required. The project uses Ollama's OpenAI-compatible API exclusively (`app/services/llm.py`).

### 7.2 Clone & create virtual environment

```bash
git clone https://github.com/oscartejedorgarcia-ops/PICTET_INVESTMENT_ASSISTANT && cd Pictet_investment_assistant
python -m venv .venv
source .venv/bin/activate
```

### 7.3 Install dependencies

```bash
pip install -r requirements.txt
```

### 7.4 Configure environment

Create a `.env` file in the project root. The following variables are read by the application:

```dotenv
# ── Ollama LLM ───────────────────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434/v1   # app/config.py → Settings.ollama_base_url
LLM_MODEL=llama3.2                          # app/config.py → Settings.llm_model

# ── Ingestion overrides (optional) ───────────────────────
# INGEST_DPI=100
# INGEST_OCR_GPU=false
# INGEST_TEXT_CHUNK_SIZE=450
# INGEST_TEXT_CHUNK_OVERLAP=50
# INGEST_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# INGEST_OCR_CONFIDENCE_THRESHOLD=0.40
# INGEST_MIN_CHUNK_LENGTH=30
```

### 7.5 Data locations

| Data | Location | Notes |
|---|---|---|
| Macro / strategy PDFs | `data/unstructured/` | Any `.pdf` files |
| Stock CSV or Excel | `data/structured/equities.xlsx` | Default path in `app/config.py` |
| ChromaDB persistence | `storage/chroma_db/` | Created automatically |
| SQLite database | `storage/sqlite/stocks.db` | Created automatically |
| Cropped figure images | `storage/resources/<doc_hash>/` | Created during ingestion |

### 7.6 Run the server

**Option A — Local (requires Ollama running separately):**

```bash
ollama pull llama3.2
uvicorn app.main:app --reload
```

**Option B (WIP) — Docker Compose (Ollama + FastAPI):**

```bash
docker-compose up
```

The API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

On startup, the server automatically ingests CSV/Excel → SQLite and PDFs → ChromaDB if the stores are empty.

### 7.7 Force re-ingestion

```bash
# Ingest all PDFs (skips already-processed by file hash)
python -m app.services.pdf_loader ingest

# Force re-ingestion of all PDFs
python -m app.services.pdf_loader ingest --force

# Ingest a single file
python -m app.services.pdf_loader ingest --file-path data/unstructured/report.pdf

# Ingest a structured file (CSV/Excel)
python -m app.services.csv_loader data/structured/equities.xlsx

# Check vector store stats
python -m app.services.pdf_loader stats

# Test retrieval
python -m app.services.pdf_loader query "inflation outlook 2025"
```

---

## API Endpoints

### `POST /api/ask`

Submit a research question. The assistant determines which data sources to query.

**Request:**

```json
{
  "question": "What is the price of Accenture?"
}
```

**Response:**

```json
{
  "answer": "The current price of Accenture PLC is $266.50.",
  "sources_used": ["structured"],
  "sql": "SELECT price FROM stocks WHERE company = 'Accenture PLC'"
}  
```

### `GET /api/health`

Returns service health and data-readiness status.

```json
{
  "status": "ok",
  "vector_store_docs": 342,
  "sqlite_loaded": true
}
```

### Example queries

| Question | Expected sources |
|---|---|
| "What is the price of Accenture?" | structured |
| "Tell me company that has the highest price?" | structured |
| "What are the current macroeconomic trends affecting markets?" | unstructured |
| "What is the price of Tesla, and how does it compare to current macro trend?" | structured + unstructured |

---

## Toolkit Choices (Validated in Code)

| Capability | Tool / Library | Where used | Why |
|---|---|---|---|
| PDF parsing & rendering | PyMuPDF (fitz) | `app/ingestion/pdf_parser.py` | Fast native text extraction + page-to-PNG rendering |
| Layout detection | Font-size + position heuristics | `app/ingestion/layout.py` | Zero-dependency, works well on digitally-born financial PDFs |
| OCR | EasyOCR | `app/ingestion/ocr.py` | Apache-2.0; MPS-compatible; good on clean financial fonts |
| Table extraction | pdfplumber + EasyOCR fallback | `app/ingestion/tables.py` | pdfplumber for vector tables, OCR row-clustering for scanned |
| Figure extraction | PyMuPDF image xrefs + layout crop | `app/ingestion/figures.py` | No extra dependency beyond PyMuPDF + Pillow |
| Chart classification | Keyword regex heuristic | `app/ingestion/charts.py` | Lightweight; covers common chart types |
| Chart-to-text | google/deplot → google/flan-t5-base | `app/ingestion/charts.py` | Apache-2.0; fully local; linearises chart data then summarises |
| Chunking | Type-aware sliding window | `app/ingestion/chunker.py` | Separate strategies for text, tables, figures, page summaries |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | `app/ingestion/embeddings.py` | 384-dim, fast, Apache-2.0, MPS-accelerated |
| Vector DB | ChromaDB (5 collections, cosine) | `app/ingestion/vectordb.py` | Persistent, embedded, no external server needed |
| Structured store | SQLite via Pandas `to_sql` | `app/services/csv_loader.py` | Zero-config; read-only queries via `run_sql()` |
| LLM | Ollama (OpenAI-compatible API) | `app/services/llm.py` | Local inference; default model `llama3.2` |
| API framework | FastAPI + Uvicorn | `app/main.py`, `app/api/routes.py` | Async, auto-docs, Pydantic validation |

---

## Testing

The project includes unit tests in `tests/`:

```bash
python -m pytest tests/ -v
```

| Test file | Coverage |
|---|---|
| `tests/test_ingestion.py` | Schema creation, chunk-to-text helpers, layout classification, chunking logic, quality gates, OCR mocking, table extraction, pipeline orchestration (395 lines) |
| `tests/test_chromadb_report.py` | Standalone script to list ChromaDB collections and document counts |
| `tests/test_sqlite_report.py` | Standalone script to list SQLite tables and row counts |

**Smoke test with curl:**

```bash
# Health check
curl -s http://localhost:8000/api/health | python -m json.tool

# Ask a question
curl -s -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the top 3 stocks by target price?"}' | python -m json.tool
```

---

## Implementation Validation Map

| Claim | Code reference | Status |
|---|---|---|
| Uses ChromaDB for vector storage | `app/ingestion/vectordb.py` → `IngestVectorStore` | ✅ Verified |
| PDF parsed via PyMuPDF | `app/ingestion/pdf_parser.py` → `parse_pdf()`, `import fitz` | ✅ Verified |
| Layout analysis via heuristics | `app/ingestion/layout.py` → `analyse_layout()`, `_classify_block()` | ✅ Verified |
| OCR via EasyOCR | `app/ingestion/ocr.py` → `_get_reader()`, `import easyocr` | ✅ Verified |
| Tables extracted with pdfplumber | `app/ingestion/tables.py` → `extract_tables_pdfplumber()`, `import pdfplumber` | ✅ Verified |
| OCR fallback for scanned tables | `app/ingestion/tables.py` → `extract_table_ocr()` | ✅ Verified |
| Figure cropping and saving | `app/ingestion/figures.py` → `extract_figures()`, saves to `storage/resources/` | ✅ Verified |
| Caption linking (nearest spatial) | `app/ingestion/figures.py` → `_find_nearest_caption()` | ✅ Verified |
| Chart classification (keyword) | `app/ingestion/charts.py` → `classify_chart_type()`, `_KEYWORD_MAP` | ✅ Verified |
| Chart-to-text via DePlot + Flan-T5 | `app/ingestion/charts.py` → `chart_to_text()`, `_load_deplot()`, `_load_summariser()` | ✅ Verified |
| Chart digitisation (DePlot → JSON) | `app/ingestion/charts.py` → `digitise_chart()`, `_parse_linearised()` | ✅ Verified |
| Type-aware chunking | `app/ingestion/chunker.py` → `chunk_text_blocks()`, `chunk_tables()`, `chunk_figures()` | ✅ Verified |
| Quality gates | `app/ingestion/quality.py` → `filter_chunks()`, `validate_text_chunk()` | ✅ Verified |
| Embeddings via all-MiniLM-L6-v2 | `app/ingestion/embeddings.py` → `_get_model()`, `embed_texts()` | ✅ Verified |
| Content-hash deduplication | `app/ingestion/vectordb.py` → `upsert_chunks()` uses `content_hash` as ID | ✅ Verified |
| Incremental ingestion (file hash) | `app/ingestion/pipeline.py` → `_ProcessedDocHashes`, `compute_file_hash()` | ✅ Verified |
| CSV/Excel → SQLite | `app/services/csv_loader.py` → `load_csv_to_sqlite()` | ✅ Verified |
| LLM-generated column descriptions | `app/services/csv_loader.py` → `_generate_column_descriptions_from_db()` | ✅ Verified |
| Column embeddings in ChromaDB | `app/services/csv_loader.py` → `_store_embeddings_in_chromadb()` → collection `stocks_db_structure` | ✅ Verified |
| Taxonomy embeddings in ChromaDB | `app/services/csv_loader.py` → `_store_taxonomy_embeddings_in_chromadb()` → collection `stocks_taxonomy` | ✅ Verified |
| Text-to-SQL via LLM | `app/services/text_to_sql.py` → `answer_with_sql()` | ✅ Verified |
| Query classification via LLM | `app/services/orchestrator.py` → `_classify()` | ✅ Verified |
| Synthesis via LLM | `app/services/orchestrator.py` → `answer()` | ✅ Verified |
| Ollama as LLM backend | `app/services/llm.py` → `_get_client()` uses `settings.ollama_base_url`, `api_key="unused"` | ✅ Verified |
| Docker Compose with Ollama sidecar | `docker-compose.yml` → services `ollama` + `app` | ✅ Verified |
| Lifespan-based auto-ingestion | `app/main.py` → `lifespan()` | ✅ Verified |
| MPS acceleration for embeddings | `app/ingestion/embeddings.py` → `_mps_available()`, `device="mps"` | ✅ Verified |
| Entity extraction via LLM at query time | Not implemented as a separate module | ⚠️ Not found — entities extracted implicitly via LLM prompt context |

---

## Refresh Assumptions

### Assumptions made

1. **`.env.example` is empty** — the file exists but contains no content. The variables documented above are inferred from `app/config.py` and `app/ingestion/config.py`.
2. **Default structured data file is `equities.xlsx`** — `app/config.py` sets `csv_path` to `data/structured/equities.xlsx`, not `data/stocks.csv` as the original README stated.
3. **Ollama, not OpenAI** — `app/config.py` sets `ollama_base_url` and `app/services/llm.py` uses `api_key="unused"`. The `openai` Python package is used only as an HTTP client for the Ollama-compatible API.
4. **Five ChromaDB collections total** — `ingest_text`, `ingest_tables`, `ingest_figures` (PDF pipeline) + `stocks_db_structure`, `stocks_taxonomy` (CSV pipeline). The original README mentioned 3.
5. **Default DPI is 100**, not 200 as previously documented (`app/ingestion/config.py`).
6. **Default chunk size is 450 chars / 50 overlap**, not 800/200 as previously documented.

### Recommended documentation improvements

- Populate `.env.example` with the variables listed in section 7.4.
- Add an `ARCHITECTURE.md` with Mermaid diagrams for the ingestion and query flows.
- Document the ChromaDB collection schemas (what metadata fields each collection stores).
- Add integration tests that spin up Ollama and exercise the full `/api/ask` round-trip.
- Document the expected CSV/Excel column names (currently inferred dynamically — any schema works, but the taxonomy embedding step expects specific columns like `company`, `sector_-_level_1`, etc.).
