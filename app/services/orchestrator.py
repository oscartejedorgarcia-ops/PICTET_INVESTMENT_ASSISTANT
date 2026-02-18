"""Orchestrator: classifies the user query, retrieves from the right sources,
and synthesises a final answer using the LLM."""

from __future__ import annotations

import json
import logging

from app.services import llm, text_to_sql

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

CLASSIFIER_SYSTEM = """\
You are a query classifier for an investment research assistant.
Given the user query, decide which data sources are needed.

Reply with a JSON object (no markdown fences) with exactly one key "sources"
whose value is a list with one or more of:
  - "structured"   → the query asks about specific stock data (prices, ratios, sectors…)
  - "unstructured"  → the query asks about macroeconomic trends, strategic insights, or PDF reports

Examples:
  "What is the target price of Tesla?" → {"sources": ["structured"]}
  "What are the current macro trends?" → {"sources": ["unstructured"]}
  "How does Apple's valuation compare to macro trends?" → {"sources": ["structured", "unstructured"]}
"""


def _classify(question: str) -> list[str]:
    raw = llm.chat(CLASSIFIER_SYSTEM, question)
    try:
        parsed = json.loads(raw)
        sources = parsed.get("sources", ["structured", "unstructured"])
        if "structured" in sources:
            logger.info("Entering structured condition.")
        if "unstructured" in sources:
            logger.info("Entering unstructured condition.")
    except (json.JSONDecodeError, AttributeError):
        # Fallback: use both sources.
        logger.warning("Classification failed – defaulting to both sources.")
        sources = ["structured", "unstructured"]
    logger.info("Query classified → sources=%s", sources)
    return sources


# ---------------------------------------------------------------------------
# Context retrieval helpers
# ---------------------------------------------------------------------------

def _retrieve_structured(question: str) -> str:
    logger.info("Question received in _retrieve_structured: %s", question)
    
    result = text_to_sql.answer_with_sql(question)
    if result["error"]:
        return f"[Structured DB error: {result['error']}]"
    if not result["results"]:
        return "[No matching stock data found.]"
    return (
        f"SQL used: {result['sql']}\n"
        f"Results:\n{json.dumps(result['results'], indent=2, default=str)}"
    )


def _retrieve_unstructured(question: str, n_results: int = 5) -> str:
    from app.ingestion.vectordb import IngestVectorStore

    store = IngestVectorStore()
    docs = store.query(question, n_results=n_results)
    if not docs:
        return "[No relevant macroeconomic documents found.]"
    parts: list[str] = []
    for i, d in enumerate(docs, 1):
        meta = d["metadata"]
        source = meta.get("source_file", "unknown")
        citation = meta.get("citation", "")
        parts.append(f"--- Excerpt {i} (from {source} – {citation}) ---\n{d['text']}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Final synthesis
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM = """\
You are an expert investment research assistant at a wealth-management firm.
Using the provided context from structured stock data and/or macroeconomic
documents, answer the user's question in clear, professional English, using executive summary style extremely concise answers.
Do not invent any data – if specific numbers or insights are not present in the context, say so.

Guidelines:
- Cite specific numbers (prices, ratios, yields) when available.
- Reference document sources when summarising macroeconomic insights.
- If the data is insufficient, say so honestly.
- Keep the answer concise but comprehensive.
"""


def answer(question: str) -> dict:
    """End-to-end pipeline: classify → retrieve → synthesise.

    Returns a dict with ``answer``, ``sources_used``, and optional ``sql``.
    """
    sources = _classify(question)

    context_parts: list[str] = []
    sql_used: str | None = None

    if "structured" in sources:
        logger.info("Entering structured condition.")
        structured_ctx = _retrieve_structured(question)
        context_parts.append(f"### Structured Stock Data\n{structured_ctx}")
        # Try to extract SQL
        if structured_ctx.startswith("SQL used:"):
            sql_used = structured_ctx.split("\n")[0].replace("SQL used: ", "")

    if "unstructured" in sources:
        logger.info("Entering unstructured condition.")
        unstructured_ctx = _retrieve_unstructured(question)
        context_parts.append(f"### Macroeconomic / Strategic Documents\n{unstructured_ctx}")

    full_context = "\n\n".join(context_parts)
    user_prompt = f"Context:\n{full_context}\n\nUser question: {question}"

    final_answer = llm.chat(SYNTHESIS_SYSTEM, user_prompt, max_tokens=2048)

    return {
        "answer": final_answer,
        "sources_used": sources,
        "sql": sql_used,
    }
