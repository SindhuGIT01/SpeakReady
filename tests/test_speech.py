"""Tests for Whisper speech-to-text.

The Groq API is mocked throughout so these tests run without a GROQ_API_KEY.
One live test transcribes the real sample fixture and is skipped otherwise.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from groq import RateLimitError

from src import config
from src.speech import (
    AudioValidationError,
    Segment,
    Transcript,
    TranscriptionError,
    Word,
    transcribe,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_ANSWER_MP3 = FIXTURES_DIR / "sample_answer.mp3"


def _fake_rate_limit_error() -> RateLimitError:
    """Build a RateLimitError without needing a real HTTP response."""
    response = MagicMock()
    response.status_code = 429
    response.headers = {}
    return RateLimitError("rate limited", response=response, body=None)


def _mock_client(response) -> MagicMock:
    """A mock Groq client whose transcriptions.create returns ``response``."""
    client = MagicMock()
    client.audio.transcriptions.create.return_value = response
    return client


def _verbose_json_response() -> MagicMock:
    """A mock response shaped like Groq's verbose_json transcription output."""
    response = MagicMock()
    response.text = "Hi, my name is Aditi."
    response.words = [
        {"word": "Hi,", "start": 0.0, "end": 0.3},
        {"word": "my", "start": 0.3, "end": 0.5},
        {"word": "name", "start": 0.5, "end": 0.8},
        {"word": "is", "start": 0.8, "end": 0.95},
        {"word": "Aditi.", "start": 0.95, "end": 1.4},
    ]
    response.duration = 1.4
    response.language = "en"
    response.segments = [
        {"start": 0.0, "end": 1.4, "avg_logprob": -0.2},
    ]
    return response


# --- validation ----------------------------------------------------------------


def test_transcribe_rejects_unsupported_format(tmp_path) -> None:
    """An unsupported extension should raise before any API call."""
    bad_file = tmp_path / "clip.ogg"
    bad_file.write_bytes(b"not real audio")

    with pytest.raises(AudioValidationError, match="Unsupported audio format"):
        transcribe(bad_file, client=_mock_client(_verbose_json_response()))


def test_transcribe_rejects_oversized_file(monkeypatch) -> None:
    """A file larger than the configured limit should be rejected."""
    monkeypatch.setattr(config, "MAX_AUDIO_SIZE_MB", 0.001)
    audio_bytes = SAMPLE_ANSWER_MP3.read_bytes()

    with pytest.raises(AudioValidationError, match="exceeds"):
        transcribe(audio_bytes, client=_mock_client(_verbose_json_response()))


def test_transcribe_rejects_too_short_clip(monkeypatch) -> None:
    """A clip shorter than the configured minimum duration should be rejected."""
    monkeypatch.setattr(config, "MIN_AUDIO_DURATION_SECONDS", 999.0)
    audio_bytes = SAMPLE_ANSWER_MP3.read_bytes()

    with pytest.raises(AudioValidationError, match="only"):
        transcribe(audio_bytes, client=_mock_client(_verbose_json_response()))


# --- successful transcription ---------------------------------------------------


def test_transcribe_returns_parsed_transcript() -> None:
    """A successful call should parse into a Transcript with word timestamps."""
    audio_bytes = SAMPLE_ANSWER_MP3.read_bytes()
    client = _mock_client(_verbose_json_response())

    result = transcribe(audio_bytes, client=client)

    assert isinstance(result, Transcript)
    assert result.text == "Hi, my name is Aditi."
    assert result.words[0] == Word(word="Hi,", start=0.0, end=0.3)
    assert result.duration_seconds == 1.4
    assert result.language == "en"
    assert result.segments == [Segment(start=0.0, end=1.4, avg_logprob=-0.2)]

    client.audio.transcriptions.create.assert_called_once()
    _, kwargs = client.audio.transcriptions.create.call_args
    assert kwargs["model"] == config.WHISPER_MODEL
    assert kwargs["response_format"] == "verbose_json"
    assert kwargs["timestamp_granularities"] == ["word", "segment"]


def test_transcribe_defaults_segments_to_empty_when_absent() -> None:
    """A response with no segments field should parse to an empty list, not raise."""
    response = _verbose_json_response()
    response.segments = None
    client = _mock_client(response)

    result = transcribe(SAMPLE_ANSWER_MP3.read_bytes(), client=client)

    assert result.segments == []


def test_transcribe_accepts_path() -> None:
    """Passing a file path should read it and infer the format from its suffix."""
    client = _mock_client(_verbose_json_response())

    result = transcribe(SAMPLE_ANSWER_MP3, client=client)

    assert result.text == "Hi, my name is Aditi."
    _, kwargs = client.audio.transcriptions.create.call_args
    assert kwargs["file"][0] == "sample_answer.mp3"


# --- rate limit retries ----------------------------------------------------------


def test_transcribe_retries_then_succeeds_on_rate_limit(monkeypatch) -> None:
    """A 429 followed by success should return the successful transcript."""
    monkeypatch.setattr(config, "WHISPER_RETRY_BASE_SECONDS", 0.0)
    audio_bytes = SAMPLE_ANSWER_MP3.read_bytes()

    client = MagicMock()
    client.audio.transcriptions.create.side_effect = [
        _fake_rate_limit_error(),
        _verbose_json_response(),
    ]

    result = transcribe(audio_bytes, client=client)

    assert result.text == "Hi, my name is Aditi."
    assert client.audio.transcriptions.create.call_count == 2


def test_transcribe_raises_after_exhausting_retries(monkeypatch) -> None:
    """Persistent rate limiting should surface as a TranscriptionError."""
    monkeypatch.setattr(config, "WHISPER_RETRY_BASE_SECONDS", 0.0)
    monkeypatch.setattr(config, "WHISPER_MAX_RETRIES", 1)
    audio_bytes = SAMPLE_ANSWER_MP3.read_bytes()

    client = MagicMock()
    client.audio.transcriptions.create.side_effect = _fake_rate_limit_error()

    with pytest.raises(TranscriptionError, match="rate limit"):
        transcribe(audio_bytes, client=client)

    assert client.audio.transcriptions.create.call_count == 2


# --- live test (skipped without an API key) --------------------------------------

pytestmark_live = pytest.mark.skipif(
    not config.GROQ_API_KEY, reason="GROQ_API_KEY not set; skipping live Whisper test"
)


@pytestmark_live
def test_transcribe_live_sample_answer() -> None:
    """The real Whisper API should transcribe the sample fixture successfully."""
    result = transcribe(SAMPLE_ANSWER_MP3)

    assert "aditi" in result.text.lower()
    assert result.words
    assert result.duration_seconds > 0
