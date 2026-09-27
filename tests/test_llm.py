"""Smoke test for the Groq LLM client. Skipped when no API key is configured."""

import pytest

from src import config

pytestmark = pytest.mark.skipif(
    not config.GROQ_API_KEY, reason="GROQ_API_KEY not set; skipping live LLM test"
)


def test_get_llm_returns_response() -> None:
    """The LLM should answer a trivial prompt with non-empty text."""
    from src.llm import get_llm

    response = get_llm(temperature=0).invoke("Reply with the single word: ready")
    assert isinstance(response.content, str)
    assert response.content.strip()
