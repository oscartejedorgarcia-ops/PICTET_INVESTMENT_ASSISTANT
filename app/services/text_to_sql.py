"""Generate and execute SQL from natural-language questions about stocks."""

from __future__ import annotations

import json
import logging
import re

from app.services import csv_loader, llm
from app.ingestion.vectordb import IngestVectorStore

logger = logging.getLogger(__name__)

#SQL_MODEL = "pxlksr/defog_sqlcoder-7b-2:Q2_KS"
SQL_MODEL = "llama3.2"  # Use Llama 3.2 for SQL generation


SYSTEM_PROMPT = """\
You are an expert SQL analyst. Given the column names that you need to use and a few examples of companies,
generate a valid SQLite SELECT query that answers the user's question.

Rules:
- Return ONLY the SQL query, nothing else.
- THE Company name is stored on the "company" column.
- The query must be a SELECT statement (read-only).
- Do NOT wrap the query in markdown code fences.
- Use LIKE with % wildcards for partial string matching.
- when searching for indsutry or sector the anwser could be aggregared  
- Column names are lowercase with underscores.
- take into consideration that all columns belong to the same table called "stocks"
- keywords like company name, sector, industry use the data extracted from the context and keep sensitive case.
"""


def _build_context(question: str) -> str:
    """Build context for SQL generation by querying ChromaDB."""
    # Initialize vector stores
    db_structure_store = IngestVectorStore(collection_name="stocks_db_structure")
    taxonomy_store = IngestVectorStore(collection_name="stocks_taxonomy")

    # Call Llama 3.2 LLM to refine the query for better embedding search

    refined_question_fields = llm.chat("need to understand what fields from a economic or finance database that contains stock information are needed to use to extract the requested information, I just need the data fields that would be needed, do not add addtional comments:", question, model="llama3.2")
    
    refined_question_stocks = llm.chat("need to get just the company, industry or sector from the the question, do not add addtional comments", question, model="llama3.2")
    

    # Query ChromaDB for database structure information using refined question
    
    db_structure_results = db_structure_store.query(refined_question_fields, n_results=15)
    

    # Extract fields to pull from the database based on the refined question
    fields_to_pull = [res['metadata'] for res in db_structure_results]
   

    # Query ChromaDB for taxonomy information using refined question
    taxonomy_results = taxonomy_store.query(refined_question_stocks, n_results=5)
    # Extract the complete embedding for taxonomy values
    taxonomy_values = [res for res in taxonomy_results]


    # Combine extracted information into the context
    db_structure_context = "\n".join(
        [f"Column: {field}" for field in fields_to_pull]
    )

    # Adjust taxonomy context to handle list of embeddings
    taxonomy_context = "\n".join(
        [
            ", ".join([f"{key}: {value}" for key, value in res['metadata'].items()])
            for res in taxonomy_values
        ]
    )
    logger.info("#####################111:")
    # Combine contexts
    ctx = "Database Structure:\n" + db_structure_context + "\n\n"
    ctx += "additional context information:\n" + taxonomy_context + "\n"

    #logger.debug("Combined Context: %s", ctx)

    # Log the combined context before returning
    logger.info("#####################Combined Context: %s", ctx)
    return ctx


def _sanitize_sql(raw: str) -> str:
    """Strip markdown fences and whitespace from LLM output."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:sql)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def answer_with_sql(question: str) -> dict:
    """Translate *question* to SQL, execute it, and return results + SQL used.

    Returns a dict with keys: ``sql``, ``results``, ``error``.
    """
    logger.info("Building context for the question...")
    context = _build_context(question)
    
    logger.info("Context Built: %s", context)

    if not context.strip():
        return {"sql": "", "results": [], "error": "Stock data not loaded."}

    user_prompt = f"{context}\n\nQuestion: {question}"

    

    logger.info("Sending prompt to LLM...")
    raw_sql = llm.chat(SYSTEM_PROMPT, user_prompt, model=SQL_MODEL)
    logger.info("Raw SQL Response: %s", raw_sql)

    logger.info("Sanitizing the SQL response...")
    sql = _sanitize_sql(raw_sql)
    
    logger.info("Executing the SQL query...")
    try:
        results = csv_loader.run_sql(sql)
    except Exception as exc:
        logger.error("SQL execution failed: %s", exc)
        return {"sql": sql, "results": [], "error": str(exc)}

    return {"sql": sql, "results": results, "error": None}
