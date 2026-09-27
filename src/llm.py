"""Factory for the Groq-hosted chat model used across SpeakReady."""

from __future__ import annotations

from langchain_groq import ChatGroq

from src import config


def get_llm(
    temperature: float | None = None,
    model: str | None = None,
) -> ChatGroq:
    """Build a ChatGroq client using settings from ``src.config``.

    Args:
        temperature: Optional override for the sampling temperature.
        model: Optional override for the model name.

    Returns:
        A configured ``ChatGroq`` instance.

    Raises:
        config.ConfigError: If GROQ_API_KEY is missing.
    """
    return ChatGroq(
        api_key=config.get_groq_api_key(),
        model=model or config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
        max_retries=config.LLM_MAX_RETRIES,
    )
