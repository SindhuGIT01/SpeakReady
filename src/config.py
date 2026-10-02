"""Central configuration for SpeakReady.

All settings (model names, paths, chunk sizes) live here. Values are loaded
from a ``.env`` file at the project root via python-dotenv; secrets must never
be hardcoded anywhere in the codebase.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR: Path = PROJECT_ROOT / "data"
QUESTIONS_DIR: Path = DATA_DIR / "questions"
RESUMES_DIR: Path = DATA_DIR / "resumes"
MODELS_DIR: Path = PROJECT_ROOT / "models"
CHROMA_DIR: Path = PROJECT_ROOT / "chroma_db"
AUDIO_DIR: Path = PROJECT_ROOT / "audio"
DB_PATH: Path = PROJECT_ROOT / "speakready.db"
FLUENCY_MODEL_PATH: Path = MODELS_DIR / "fluency_model.joblib"

# --- API keys ----------------------------------------------------------------
_GROQ_KEY_PLACEHOLDER = "your_groq_api_key_here"
_raw_groq_key = os.getenv("GROQ_API_KEY", "").strip()
GROQ_API_KEY: str | None = (
    _raw_groq_key if _raw_groq_key and _raw_groq_key != _GROQ_KEY_PLACEHOLDER else None
)

# --- LLM ---------------------------------------------------------------------
LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))

# --- Speech ------------------------------------------------------------------
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo")
TTS_LANGUAGE: str = os.getenv("TTS_LANGUAGE", "en")
SUPPORTED_AUDIO_FORMATS: frozenset[str] = frozenset({".wav", ".mp3", ".m4a", ".webm"})
MAX_AUDIO_SIZE_MB: float = float(os.getenv("MAX_AUDIO_SIZE_MB", "25"))
MIN_AUDIO_DURATION_SECONDS: float = float(os.getenv("MIN_AUDIO_DURATION_SECONDS", "2.0"))
WHISPER_MAX_RETRIES: int = int(os.getenv("WHISPER_MAX_RETRIES", "3"))
WHISPER_RETRY_BASE_SECONDS: float = float(os.getenv("WHISPER_RETRY_BASE_SECONDS", "1.0"))

# --- RAG ---------------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
RETRIEVER_TOP_K: int = int(os.getenv("RETRIEVER_TOP_K", "4"))
QUESTIONS_COLLECTION: str = "question_bank"
RESUME_COLLECTION: str = "resumes"

# --- Fluency features ---------------------------------------------------------
PAUSE_THRESHOLD_SECONDS: float = float(os.getenv("PAUSE_THRESHOLD_SECONDS", "0.5"))
LONG_PAUSE_THRESHOLD_SECONDS: float = float(os.getenv("LONG_PAUSE_THRESHOLD_SECONDS", "1.5"))
# Single words and short phrases counted as filler. "like" is handled specially
# (see src/features.py) to avoid flagging its use as an ordinary verb.
FILLER_WORDS: tuple[str, ...] = (
    "um",
    "uh",
    "umm",
    "like",
    "you know",
    "basically",
    "actually",
    "so",
    "i mean",
)
IDEAL_WPM_MIN: float = float(os.getenv("IDEAL_WPM_MIN", "130"))
IDEAL_WPM_MAX: float = float(os.getenv("IDEAL_WPM_MAX", "160"))
FILLER_RATE_HIGH_PER_100_WORDS: float = float(os.getenv("FILLER_RATE_HIGH_PER_100_WORDS", "5"))
TOTAL_PAUSE_RATIO_HIGH: float = float(os.getenv("TOTAL_PAUSE_RATIO_HIGH", "0.3"))
LOW_TYPE_TOKEN_RATIO: float = float(os.getenv("LOW_TYPE_TOKEN_RATIO", "0.4"))
LONG_SENTENCE_WORD_COUNT: float = float(os.getenv("LONG_SENTENCE_WORD_COUNT", "30"))
SHORT_SENTENCE_WORD_COUNT: float = float(os.getenv("SHORT_SENTENCE_WORD_COUNT", "5"))

# --- Feedback engine -----------------------------------------------------------
# Weights for combining the ML fluency score (0-100) and the LLM content score
# (0-10, scaled to 0-100) into one overall score. Should sum to 1.0.
CONTENT_SCORE_WEIGHT: float = float(os.getenv("CONTENT_SCORE_WEIGHT", "0.6"))
FLUENCY_SCORE_WEIGHT: float = float(os.getenv("FLUENCY_SCORE_WEIGHT", "0.4"))
# Extra attempts if the LLM's structured feedback output fails validation
# (0 = try once, no retry).
FEEDBACK_LLM_MAX_RETRIES: int = int(os.getenv("FEEDBACK_LLM_MAX_RETRIES", "1"))

# --- Body language (webcam coaching) -------------------------------------------
# MediaPipe Tasks model bundles, downloaded once (if missing) and cached under
# MODELS_DIR, the same way the fluency model is loaded from disk.
FACE_LANDMARKER_MODEL_PATH: Path = MODELS_DIR / "face_landmarker.task"
FACE_LANDMARKER_MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
POSE_LANDMARKER_MODEL_PATH: Path = MODELS_DIR / "pose_landmarker_lite.task"
POSE_LANDMARKER_MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# A face is counted as "forward-facing" (good eye contact) when its head pose,
# derived from MediaPipe's per-frame facial transformation matrix, stays
# within these yaw/pitch bounds (degrees) of looking straight at the camera.
EYE_CONTACT_YAW_THRESHOLD_DEGREES: float = float(
    os.getenv("EYE_CONTACT_YAW_THRESHOLD_DEGREES", "25")
)
EYE_CONTACT_PITCH_THRESHOLD_DEGREES: float = float(
    os.getenv("EYE_CONTACT_PITCH_THRESHOLD_DEGREES", "20")
)
# Minimum duration a "looking away" streak must last to count as one distinct event.
LOOKING_AWAY_MIN_SECONDS: float = float(os.getenv("LOOKING_AWAY_MIN_SECONDS", "1.0"))
# Shoulder tilt (degrees from level) at which the posture "level" sub-score hits 0.
POSTURE_TILT_SCALE_DEGREES: float = float(os.getenv("POSTURE_TILT_SCALE_DEGREES", "20"))
# Reference nose-to-shoulder-midpoint distance (as a ratio of shoulder width) for
# a fully upright, non-slouched posture; the "upright" sub-score is scaled against it.
POSTURE_IDEAL_NECK_RATIO: float = float(os.getenv("POSTURE_IDEAL_NECK_RATIO", "0.55"))
# Default frame sampling rate assumed for a raw frame sequence (e.g. webcam
# snapshots) that doesn't carry its own timing metadata.
BODY_LANGUAGE_DEFAULT_FPS: float = float(os.getenv("BODY_LANGUAGE_DEFAULT_FPS", "5.0"))

# --- Answer replay timeline (Task 12) ------------------------------------------
# A "fast_speech" marker is raised over any stretch where a rolling window of
# this many consecutive words is spoken faster than this local WPM.
# LONG_PAUSE_THRESHOLD_SECONDS (above) is reused as-is for "long_pause" markers.
FAST_SPEECH_WINDOW_WORDS: int = int(os.getenv("FAST_SPEECH_WINDOW_WORDS", "5"))
FAST_SPEECH_WPM_THRESHOLD: float = float(os.getenv("FAST_SPEECH_WPM_THRESHOLD", "200"))


class ConfigError(RuntimeError):
    """Raised when a required setting is missing or invalid."""


def get_groq_api_key() -> str:
    """Return the Groq API key, failing loudly if it is not configured.

    Returns:
        The GROQ_API_KEY value from the environment or ``.env``.

    Raises:
        ConfigError: If GROQ_API_KEY is missing or empty.
    """
    if not GROQ_API_KEY:
        raise ConfigError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
            "free key from https://console.groq.com/keys"
        )
    return GROQ_API_KEY
