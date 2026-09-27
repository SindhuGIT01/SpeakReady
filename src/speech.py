"""Speech-to-text via the Groq-hosted Whisper API.

Validates uploaded audio (format, size, minimum duration), transcribes it with
word-level timestamps, and retries automatically on rate limits.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from groq import Groq, RateLimitError

from src import config


class AudioValidationError(ValueError):
    """Raised when uploaded audio fails format, size, or duration checks."""


class TranscriptionError(RuntimeError):
    """Raised when the Groq Whisper API call fails after retries."""


@dataclass(frozen=True)
class Word:
    """A single word with its timing within the audio.

    Attributes:
        word: The transcribed word text.
        start: Start time in seconds from the beginning of the audio.
        end: End time in seconds from the beginning of the audio.
    """

    word: str
    start: float
    end: float


@dataclass(frozen=True)
class Transcript:
    """The result of transcribing a spoken answer.

    Attributes:
        text: The full transcribed text.
        words: Word-level timestamps, in order.
        duration_seconds: Length of the audio, in seconds.
        language: Detected or requested language code (e.g. ``"en"``).
    """

    text: str
    words: list[Word] = field(default_factory=list)
    duration_seconds: float = 0.0
    language: str = ""


@lru_cache(maxsize=1)
def _get_client() -> Groq:
    """Return a cached Groq API client.

    Returns:
        A configured ``Groq`` client.

    Raises:
        config.ConfigError: If GROQ_API_KEY is missing.
    """
    return Groq(api_key=config.get_groq_api_key())


def _read_audio_bytes(audio: bytes | str | Path) -> tuple[bytes, str]:
    """Load raw audio bytes and a filename from a path or in-memory bytes.

    Args:
        audio: Raw audio bytes, or a path to an audio file on disk.

    Returns:
        A tuple of (audio_bytes, filename). The filename is used only to
        infer the format extension and to label the upload.
    """
    if isinstance(audio, (str, Path)):
        path = Path(audio)
        return path.read_bytes(), path.name
    return audio, "audio.wav"


def _probe_duration_seconds(audio_bytes: bytes) -> float | None:
    """Best-effort read of an audio clip's duration.

    Only WAV/FLAC/OGG are reliably decodable this way without extra
    dependencies; other supported formats (mp3, m4a, webm) simply skip this
    pre-check and rely on Whisper's own reported duration instead.

    Args:
        audio_bytes: Raw audio file bytes.

    Returns:
        The duration in seconds, or ``None`` if it could not be determined.
    """
    try:
        import soundfile as sf
    except ImportError:
        return None

    try:
        info = sf.info(io.BytesIO(audio_bytes))
    except Exception:
        return None
    return info.frames / info.samplerate if info.samplerate else None


def _validate_audio(audio_bytes: bytes, filename: str) -> None:
    """Validate an audio upload before sending it to the API.

    Args:
        audio_bytes: Raw audio file bytes.
        filename: Filename used to infer the format extension.

    Raises:
        AudioValidationError: If the format is unsupported, the file is too
            large, or the clip is shorter than the configured minimum.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in config.SUPPORTED_AUDIO_FORMATS:
        supported = ", ".join(sorted(config.SUPPORTED_AUDIO_FORMATS))
        raise AudioValidationError(
            f"Unsupported audio format '{suffix}'. Supported formats: {supported}."
        )

    size_mb = len(audio_bytes) / (1024 * 1024)
    if size_mb > config.MAX_AUDIO_SIZE_MB:
        raise AudioValidationError(
            f"Audio file is {size_mb:.1f} MB, which exceeds the "
            f"{config.MAX_AUDIO_SIZE_MB:.0f} MB limit. Please record a shorter answer."
        )

    duration = _probe_duration_seconds(audio_bytes)
    if duration is not None and duration < config.MIN_AUDIO_DURATION_SECONDS:
        raise AudioValidationError(
            f"Audio clip is only {duration:.1f}s long; please speak for at least "
            f"{config.MIN_AUDIO_DURATION_SECONDS:.0f} seconds."
        )


def _transcribe_with_retry(client: Groq, filename: str, audio_bytes: bytes):
    """Call the Whisper transcription endpoint, retrying on rate limits.

    Args:
        client: The Groq API client.
        filename: Filename to label the upload with.
        audio_bytes: Raw audio file bytes.

    Returns:
        The raw Groq ``Transcription`` response.

    Raises:
        TranscriptionError: If the rate limit persists past the configured
            number of retries.
    """
    last_error: RateLimitError | None = None
    for attempt in range(config.WHISPER_MAX_RETRIES + 1):
        try:
            return client.audio.transcriptions.create(
                model=config.WHISPER_MODEL,
                file=(filename, audio_bytes),
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )
        except RateLimitError as exc:
            last_error = exc
            if attempt == config.WHISPER_MAX_RETRIES:
                break
            time.sleep(config.WHISPER_RETRY_BASE_SECONDS * (2**attempt))

    raise TranscriptionError(
        f"Groq Whisper rate limit exceeded after {config.WHISPER_MAX_RETRIES} retries. "
        "Please wait a moment and try again."
    ) from last_error


def transcribe(audio: bytes | str | Path, client: Groq | None = None) -> Transcript:
    """Transcribe a spoken answer with word-level timestamps.

    Args:
        audio: Raw audio bytes, or a path to an audio file on disk. Supported
            formats are wav, mp3, m4a, and webm.
        client: Optional Groq client to use (mainly for testing). Defaults to
            a cached client built from ``config.get_groq_api_key()``.

    Returns:
        A :class:`Transcript` with the full text, word timestamps, duration,
        and detected language.

    Raises:
        AudioValidationError: If the audio fails format, size, or duration
            validation.
        TranscriptionError: If the API call fails after retrying rate limits.
        config.ConfigError: If no ``client`` is given and GROQ_API_KEY is
            missing.
    """
    audio_bytes, filename = _read_audio_bytes(audio)
    _validate_audio(audio_bytes, filename)

    api_client = client if client is not None else _get_client()
    response = _transcribe_with_retry(api_client, filename, audio_bytes)

    words = [
        Word(word=w["word"], start=w["start"], end=w["end"])
        for w in (getattr(response, "words", None) or [])
    ]
    return Transcript(
        text=response.text,
        words=words,
        duration_seconds=float(getattr(response, "duration", 0.0) or 0.0),
        language=getattr(response, "language", "") or "",
    )
