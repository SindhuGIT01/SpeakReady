"""Tests for gTTS text-to-speech with disk caching.

``gTTS`` itself is mocked throughout so these tests never make a network call.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src import config
from src.tts import _cache_path, speak


@pytest.fixture(autouse=True)
def _isolated_audio_dir(tmp_path, monkeypatch):
    """Redirect the TTS cache to a temp directory for every test."""
    monkeypatch.setattr(config, "AUDIO_DIR", tmp_path)


def _mock_gtts(audio_bytes: bytes) -> MagicMock:
    """A mock gTTS instance whose write_to_fp writes fixed bytes."""
    instance = MagicMock()
    instance.write_to_fp.side_effect = lambda buffer: buffer.write(audio_bytes)
    return instance


def test_speak_rejects_empty_text() -> None:
    """Empty or whitespace-only text should raise before calling gTTS."""
    with pytest.raises(ValueError, match="empty"):
        speak("   ")


def test_speak_synthesizes_and_caches() -> None:
    """A first call should synthesize audio and write it to the cache file."""
    with patch("src.tts.gTTS", return_value=_mock_gtts(b"fake-mp3-bytes")) as gtts_cls:
        result = speak("Tell me about yourself.", language="en")

    assert result == b"fake-mp3-bytes"
    gtts_cls.assert_called_once_with(text="Tell me about yourself.", lang="en")
    cache_file = _cache_path("Tell me about yourself.", "en")
    assert cache_file.exists()
    assert cache_file.read_bytes() == b"fake-mp3-bytes"


def test_speak_uses_cache_on_second_call() -> None:
    """A second call with the same text/language should skip gTTS entirely."""
    with patch("src.tts.gTTS", return_value=_mock_gtts(b"fake-mp3-bytes")):
        speak("Repeated question", language="en")

    with patch("src.tts.gTTS") as gtts_cls_second:
        result = speak("Repeated question", language="en")

    gtts_cls_second.assert_not_called()
    assert result == b"fake-mp3-bytes"


def test_speak_defaults_to_configured_language(monkeypatch) -> None:
    """Omitting language should fall back to config.TTS_LANGUAGE."""
    monkeypatch.setattr(config, "TTS_LANGUAGE", "fr")

    with patch("src.tts.gTTS", return_value=_mock_gtts(b"bytes")) as gtts_cls:
        speak("Bonjour")

    gtts_cls.assert_called_once_with(text="Bonjour", lang="fr")


def test_different_languages_use_different_cache_entries() -> None:
    """The same text in two languages should not collide in the cache."""
    assert _cache_path("Hello", "en") != _cache_path("Hello", "fr")
