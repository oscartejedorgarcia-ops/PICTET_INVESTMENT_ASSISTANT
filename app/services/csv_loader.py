"""Load stock data (CSV or Excel) into a SQLite database for Text-to-SQL queries."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import pandas as pd

from app.config import settings
from app.ingestion.vectordb import IngestVectorStore

logger = logging.getLogger(__name__)

TABLE_NAME = "stocks"
METADATA_TABLE = "column_metadata"


def _get_connection() -> sqlite3.Connection:
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(settings.sqlite_path))


def _generate_column_descriptions(df: pd.DataFrame, sample_n: int = 10) -> dict[str, str]:
    """Use the local LLM to produce a concise description for every column.

    Reads *sample_n* rows, sends column names + sample values to llama3.2,
    and returns a mapping ``{normalised_column_name: description}``.
    """
    from app.services import llm  # local import to avoid circular deps

    sample = df.head(sample_n)

    # Build a compact representation: column -> list of sample values
    col_samples: dict[str, list] = {}
    for col in sample.columns:
        values = sample[col].dropna().astype(str).tolist()
        col_samples[col] = values[:sample_n]

    system_prompt = (
        "You are a financial data analyst. You will receive a JSON object where each key "
        "is a column name from a stock/equity dataset and each value is a list of sample "
        "values from that column.\n\n"
        "For EVERY column, produce a clear, concise, one-sentence description of what the "
        "column stores, its data type (e.g., text, number, date, boolean), and its meaning "
        "in a financial/investment context.\n\n"
        "Return ONLY a valid JSON object mapping each column name to its description. "
        "Do NOT include any explanations, code examples, or additional text. "
        "Ensure the JSON is properly formatted and parsable."
    )

    user_prompt = json.dumps(col_samples, indent=2, default=str)

    logger.info("Requesting column descriptions from LLM for %d columns…", len(col_samples))
    raw = llm.chat(system_prompt, user_prompt, max_tokens=4096)

    # Parse the LLM response – strip possible code fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    try:
        descriptions: dict[str, str] = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("LLM returned invalid JSON for column descriptions. Raw output:\n%s", raw)
        # Fallback: empty descriptions so the load still succeeds
        descriptions = {col: "" for col in df.columns}

    # Iterate over columns and ensure every column in the DataFrame has a description
    for col in df.columns:
        if col not in descriptions:
            descriptions[col] = f"Description for column '{col}' could not be generated."

    return descriptions


def _generate_column_descriptions_from_db(conn: sqlite3.Connection, table_name: str = TABLE_NAME, sample_n: int = 10) -> dict[str, str]:
    """Generate column descriptions by retrieving sample data from the database."""
    from app.services import llm  # local import to avoid circular deps

    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]

    descriptions = {}

    for col in columns:
        # Ensure column names are quoted to handle special characters
        quoted_col = f'"{col}"'

        # Retrieve sample data for the column
        cursor = conn.execute(f"SELECT {quoted_col} FROM {table_name} LIMIT {sample_n}")
        sample_values = [row[0] for row in cursor.fetchall() if row[0] is not None]

        # Skip if no sample data is available
        if not sample_values:
            descriptions[col] = f"No sample data available for column '{col}'."
            continue

        # Prepare the prompt for the LLM
        system_prompt = (
            "You are a financial data analyst. You will receive a column name and a list of sample "
            "values from a stock/equity dataset.\n\n"
            "Produce a clear, concise, one-sentence description of what the column stores including the minimun information needed, do "
            "not include information like this column stores \n\n"
            "Return ONLY the description as plain text. Do NOT include any additional text."
        )

        user_prompt = json.dumps({"column": col, "samples": sample_values}, indent=2, default=str)

        logger.info("Requesting description for column '%s' with %d samples…", col, len(sample_values))
        raw = llm.chat(system_prompt, user_prompt, max_tokens=512)

        # Log the raw output from the LLM for debugging purposes
        print(f"Raw LLM output for column '{col}': {raw}")  # Debugging output
        logger.debug("Raw LLM output for column '%s': %s", col, raw)

        # Clean and store the description
        description = raw.strip()
        descriptions[col] = description

    return descriptions


def _store_column_metadata(
    conn: sqlite3.Connection,
    descriptions: dict[str, str],
    table_name: str = TABLE_NAME,
) -> None:
    """Persist column descriptions in a ``column_metadata`` table."""
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {METADATA_TABLE} ("
        "  table_name TEXT NOT NULL,"
        "  column_name TEXT NOT NULL,"
        "  description TEXT,"
        "  PRIMARY KEY (table_name, column_name)"
        ")"
    )
    conn.execute(
        f"DELETE FROM {METADATA_TABLE} WHERE table_name = ?", (table_name,)
    )
    conn.executemany(
        f"INSERT INTO {METADATA_TABLE} (table_name, column_name, description) VALUES (?, ?, ?)",
        [(table_name, col, desc) for col, desc in descriptions.items()],
    )
    conn.commit()
    logger.info(
        "Stored %d column descriptions in '%s'.", len(descriptions), METADATA_TABLE
    )


def _store_embeddings_in_chromadb(
    table_name: str, descriptions: dict[str, str]
) -> None:
    """Generate embeddings for column descriptions and store them in ChromaDB."""
    store = IngestVectorStore(collection_name="stocks_db_structure")

    # Prepare data for embedding
    embeddings_data = [
        {
            "table": table_name,
            "column": column,
            "description": description,
        }
        for column, description in descriptions.items()
    ]

    # Add embeddings to ChromaDB
    ids = [f"{data['table']}_{data['column']}" for data in embeddings_data]
    documents = [
        f"Table: {data['table']}, Column: {data['column']}, Description: {data['description']}"
        for data in embeddings_data
    ]
    metadatas = [
        {
            "table": data["table"],
            "column": data["column"],
        }
        for data in embeddings_data
    ]

    store.add_documents(ids=ids, documents=documents, metadatas=metadatas)

    logger.info("Stored embeddings for table '%s' in ChromaDB.", table_name)


def _store_taxonomy_embeddings_in_chromadb(df: pd.DataFrame) -> None:
    """Generate embeddings for taxonomy fields and store them in ChromaDB."""
    store = IngestVectorStore(collection_name="stocks_taxonomy")

    # Define the fields to extract for embeddings
    taxonomy_fields = [
        "company",
        "sector_-_level_1",
        "industry_group_-_level_2",
        "industry_-_level_3",
        "sub-industry_-_level_4",
        "region",
    ]

 
    # Ensure the fields exist in the DataFrame
    missing_fields = [field for field in taxonomy_fields if field not in df.columns]
    if missing_fields:
        logger.warning("Missing fields in the DataFrame: %s", missing_fields)
        print(f"Missing fields in the DataFrame: {missing_fields}")
        return

    # Iterate over rows in the DataFrame to create embeddings
    embeddings_data = df[taxonomy_fields].dropna().to_dict(orient="records")

    ids = [f"row_{index}" for index in range(len(embeddings_data))]
    documents = [
        ", ".join([f"{key}: {value}" for key, value in data.items()])
        for data in embeddings_data
    ]
    metadatas = [data for data in embeddings_data]

    # Debugging logs 
    print("before store taxonomy embeddings in ChromaDB.")


    # Add embeddings to ChromaDB
    store.add_documents(ids=ids, documents=documents, metadatas=metadatas)

    logger.info("Stored taxonomy embeddings in ChromaDB.")
    print("Stored taxonomy embeddings in ChromaDB.")


def load_csv_to_sqlite(csv_path: Path | None = None) -> int:
    """Read a CSV or Excel file and write it into a SQLite table. Returns row count."""
    csv_path = csv_path or settings.csv_path
    if not csv_path.exists():
        logger.warning("Data file %s does not exist – skipping load.", csv_path)
        return 0

    suffix = csv_path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(csv_path)
    else:
        df = pd.read_csv(csv_path)

    # Normalize column names: strip whitespace, lowercase, replace spaces with underscores.
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    conn = _get_connection()
    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    row_count = len(df)

    # Generate & store column descriptions from database
    descriptions = _generate_column_descriptions_from_db(conn, TABLE_NAME)
    _store_column_metadata(conn, descriptions)

    # Store embeddings in ChromaDB
    _store_embeddings_in_chromadb(TABLE_NAME, descriptions)

    # Store taxonomy embeddings in ChromaDB
    # Log the size of the DataFrame before storing taxonomy embeddings
    logger.info("DataFrame size before storing taxonomy embeddings: %d rows", len(df))
    _store_taxonomy_embeddings_in_chromadb(df)

    conn.close()

    logger.info("Loaded %d rows from %s into SQLite table '%s'.", row_count, csv_path.name, TABLE_NAME)
    return row_count


def get_column_descriptions(table_name: str = TABLE_NAME) -> dict[str, str]:
    """Return stored column descriptions as ``{column_name: description}``."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            f"SELECT column_name, description FROM {METADATA_TABLE} WHERE table_name = ?",
            (table_name,),
        )
        descriptions = {row[0]: row[1] for row in cursor.fetchall()}
    except sqlite3.OperationalError:
        descriptions = {}
    conn.close()
    return descriptions


def get_table_schema() -> str:
    """Return the CREATE TABLE statement for the stocks table (used in prompts)."""
    conn = _get_connection()
    cursor = conn.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{TABLE_NAME}'")
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return ""
    return row[0]


def get_full_ddl() -> str:
    """Return the complete DDL of the SQLite database (all tables, indexes, views)."""
    conn = _get_connection()
    cursor = conn.execute(
        "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name"
    )
    statements = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return "\n\n".join(statements) if statements else ""


def get_sample_rows(n: int = 3) -> list[dict]:
    """Return *n* sample rows from the stocks table as list of dicts."""
    conn = _get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(f"SELECT * FROM {TABLE_NAME} LIMIT {n}")
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def run_sql(sql: str) -> list[dict]:
    """Execute a read-only SQL query and return results as list of dicts."""
    conn = _get_connection()
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql)
        rows = [dict(r) for r in cursor.fetchall()]
    except Exception:
        conn.close()
        raise
    conn.close()
    return rows



if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)

    if len(sys.argv) < 2:
        print("Usage: python -m app.services.csv_loader <path_to_csv_or_excel>")
        sys.exit(1)

    csv_path = sys.argv[1]
    rows = load_csv_to_sqlite(csv_path=Path(csv_path))
    print(f"Loaded {rows} rows into SQLite")
