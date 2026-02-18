"""REST API routes for the Investment Research Assistant."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.orchestrator import answer as orchestrator_answer

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's investment research question.")


class QuestionResponse(BaseModel):
    answer: str
    sources_used: list[str]
    sql: str | None = None


class HealthResponse(BaseModel):
    status: str
    vector_store_docs: int
    sqlite_loaded: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Return service health and data-readiness status."""
    from app.ingestion.vectordb import IngestVectorStore
    from app.services.csv_loader import get_table_schema

    store = IngestVectorStore()
    return HealthResponse(
        status="ok",
        vector_store_docs=store.total_count,
        sqlite_loaded=bool(get_table_schema()),
    )


@router.post("/ask", response_model=QuestionResponse, tags=["research"])
async def ask_question(body: QuestionRequest):
    """Submit a research question and receive an AI-generated answer.

    The assistant will automatically determine whether to query
    structured stock data, unstructured macro documents, or both.
    """
    try:
        result = orchestrator_answer(body.question)
    except Exception as exc:
        logger.exception("Error processing question.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QuestionResponse(**result)
