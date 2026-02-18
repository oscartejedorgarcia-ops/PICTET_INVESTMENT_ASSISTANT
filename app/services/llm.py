"""Thin wrapper around the Ollama local LLM chat-completion API."""

from __future__ import annotations

import logging

from openai import OpenAI as _HTTPClient

from app.config import settings

logger = logging.getLogger(__name__)

_client: _HTTPClient | None = None


def _get_client() -> _HTTPClient:
    global _client
    if _client is None:
        _client = _HTTPClient(
            base_url=settings.ollama_base_url,
            api_key="unused",  # Ollama does not require an API key
        )
    return _client


def chat(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """Send a chat completion request and return the assistant's reply."""
    client = _get_client()
    response = client.chat.completions.create(
        model=model or settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    logger.debug("LLM response (%d chars): %s…", len(content), content[:120])
    return content.strip()
