"""Application entry-point – creates the FastAPI app and runs startup tasks."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import settings
from app.services.csv_loader import load_csv_to_sqlite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _sqlite_has_data() -> bool:
    """Return True if the SQLite DB file exists and the stocks table has rows."""
    if not settings.sqlite_path.exists():
        return False
    import sqlite3
    try:
        conn = sqlite3.connect(str(settings.sqlite_path))
        cursor = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='stocks'"
        )
        table_exists = cursor.fetchone()[0] > 0
        if not table_exists:
            conn.close()
            return False
        cursor = conn.execute("SELECT COUNT(*) FROM stocks")
        row_count = cursor.fetchone()[0]
        conn.close()
        return row_count > 0
    except Exception:
        return False


def _chroma_has_data() -> bool:
    """Return True if ChromaDB already contains ingested documents."""
    from app.ingestion.config import ingest_settings
    chroma_db_file = ingest_settings.chroma_dir / "chroma.sqlite3"
    if not chroma_db_file.exists():
        return False
    try:
        from app.ingestion.vectordb import IngestVectorStore
        store = IngestVectorStore()
        return store.total_count > 0
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ingest data sources only if the databases don't already exist."""
    logger.info("=== Checking data stores ===")

    # Structured data → SQLite
    if _sqlite_has_data():
        logger.info("SQLite already populated – skipping CSV/Excel load.")
    else:
        logger.info("SQLite empty or missing – loading structured data…")
        csv_rows = load_csv_to_sqlite()
        logger.info("CSV → SQLite: %d rows loaded.", csv_rows)

    # Unstructured data → ChromaDB
    if _chroma_has_data():
        logger.info("ChromaDB already populated – skipping PDF ingestion.")
    else:
        logger.info("ChromaDB empty or missing – ingesting PDFs…")
        from app.ingestion.pipeline import ingest_folder
        stats = ingest_folder()
        logger.info(
            "PDFs → ChromaDB: %d files, %d chunks stored in %.1fs.",
            stats.files_processed,
            stats.total_stored,
            stats.elapsed_seconds,
        )

    logger.info("=== Startup complete ===")
    yield


app = FastAPI(
    title="Stock Investment Research Assistant",
    description=(
        "GenAI-powered assistant combining macroeconomic documents (PDFs) "
        "with structured stock data (CSV) to answer investment research queries."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api")
