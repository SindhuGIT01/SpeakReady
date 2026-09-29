"""Text-to-speech via gTTS, with a disk cache so repeated questions are free.

Interview questions repeat often (same question bank, same follow-ups), so
generated audio is cached under ``config.AUDIO_DIR`` keyed by text + language.
"""

from __future__ import annotations

import hashlib
import io

from gtts import gTTS

from src import config


def _cache_path(text: str, language: str):
    """Build the cache file path for a given text and language.

    Args:
        text: The text that would be spoken.
        language: The gTTS language code.

    Returns:
        The path where this clip's cached mp3 bytes are (or would be) stored.
    """
    key = hashlib.sha256(f"{language}:{text}".encode()).hexdigest()
    return config.AUDIO_DIR / f"tts_{key}.mp3"


def speak(text: str, language: str | None = None) -> bytes:
    """Synthesize speech for the given text, using a cached clip if available.

    Args:
        text: The text to speak. Must be non-empty.
        language: Optional gTTS language code override. Defaults to
            ``config.TTS_LANGUAGE``.

    Returns:
        MP3-encoded audio bytes.

    Raises:
        ValueError: If ``text`` is empty or whitespace-only.
    """
    if not text.strip():
        raise ValueError("text must not be empty")

    lang = language or config.TTS_LANGUAGE
    cache_file = _cache_path(text, lang)
    if cache_file.exists():
        return cache_file.read_bytes()

    buffer = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buffer)
    audio_bytes = buffer.getvalue()

    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(audio_bytes)
    return audio_bytes
