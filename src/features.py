"""Fluency feature extraction from a transcribed spoken answer.

``extract_features`` is the single source of truth for turning a
:class:`~src.speech.Transcript` into a numeric feature vector. It is used both
to build the training set for the fluency model (Task 6) and at inference
time in the app, so the two never drift out of sync.
"""

from __future__ import annotations

import re
from functools import lru_cache

from src import config
from src.speech import Transcript, Word

_PUNCT_STRIP_RE = re.compile(r"^[^a-zA-Z0-9']+|[^a-zA-Z0-9']+$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_SENTENCE_WORD_RE = re.compile(r"[A-Za-z0-9']+")

# The keys of the dict returned by extract_features(), in a fixed order.
# Single source of truth for the fluency model's feature schema (Task 6):
# both the training notebook and src/scorer.py import this instead of
# hardcoding the column order, so they can never drift out of sync.
FEATURE_NAMES: tuple[str, ...] = (
    "words_per_minute",
    "pause_count",
    "long_pause_count",
    "mean_pause_duration",
    "total_pause_ratio",
    "filler_count",
    "filler_rate",
    "repetition_count",
    "type_token_ratio",
    "mean_sentence_length",
)

# Words that make "like" a normal verb rather than a filler, e.g. "I like
# pizza", "would you like", "do you like". This is a heuristic: it catches the
# common subject-before-verb pattern but will miss or mis-flag edge cases.
_VERB_PRECEDING_LIKE = frozenset(
    {"i", "you", "we", "they", "he", "she", "it", "who", "would", "do", "does", "did", "to"}
)


def _normalize_token(word: str) -> str:
    """Lowercase a word and strip leading/trailing punctuation.

    Args:
        word: A raw word token, possibly with attached punctuation (e.g.
            ``"Aditi."``).

    Returns:
        The normalized token (e.g. ``"aditi"``), or an empty string if the
        word was punctuation-only.
    """
    return _PUNCT_STRIP_RE.sub("", word).lower()


@lru_cache(maxsize=1)
def _filler_phrases() -> tuple[tuple[str, ...], ...]:
    """Multi-word filler phrases from config, as tuples of tokens.

    Returns:
        Filler phrases (e.g. ``("you", "know")``) sorted longest-first so
        longer phrases are matched before shorter ones.
    """
    phrases = [tuple(f.lower().split()) for f in config.FILLER_WORDS if " " in f]
    return tuple(sorted(phrases, key=len, reverse=True))


@lru_cache(maxsize=1)
def _single_word_fillers() -> frozenset[str]:
    """Single-word fillers from config, excluding "like" (handled specially).

    Returns:
        The set of single-word filler tokens.
    """
    return frozenset(
        f.lower() for f in config.FILLER_WORDS if " " not in f and f.lower() != "like"
    )


def _count_fillers(tokens: list[str]) -> int:
    """Count filler words and phrases in a normalized token stream.

    Args:
        tokens: Normalized (lowercased, punctuation-stripped) word tokens.

    Returns:
        The number of filler occurrences, treating each matched multi-word
        phrase as a single occurrence and skipping "like" when it is
        preceded by a word that suggests ordinary verb usage.
    """
    phrases = _filler_phrases()
    singles = _single_word_fillers()

    count = 0
    i = 0
    n = len(tokens)
    while i < n:
        matched_len = 0
        for phrase in phrases:
            plen = len(phrase)
            if tuple(tokens[i : i + plen]) == phrase:
                count += 1
                matched_len = plen
                break
        if matched_len:
            i += matched_len
            continue

        token = tokens[i]
        if token in singles:
            count += 1
        elif token == "like":
            prev_token = tokens[i - 1] if i > 0 else ""
            if prev_token not in _VERB_PRECEDING_LIKE:
                count += 1
        i += 1

    return count


def _count_repetitions(tokens: list[str]) -> int:
    """Count immediately-repeated words in a normalized token stream.

    Args:
        tokens: Normalized (lowercased, punctuation-stripped) word tokens.

    Returns:
        The number of positions where a token is identical to the one right
        before it (e.g. "the the cat" counts as one repetition).
    """
    return sum(1 for prev, cur in zip(tokens, tokens[1:], strict=False) if prev == cur)


def _pause_gaps(words: list[Word]) -> list[float]:
    """Silence durations between consecutive words.

    Args:
        words: Word-level timestamps, in speaking order.

    Returns:
        The gap in seconds before each word (excluding the first), clamped
        to zero for overlapping or out-of-order timestamps.
    """
    return [
        max(0.0, cur.start - prev.end) for prev, cur in zip(words, words[1:], strict=False)
    ]


def _mean_sentence_length(text: str) -> float:
    """Average number of words per sentence in the raw transcript text.

    Args:
        text: The full transcribed text.

    Returns:
        The mean word count across sentences (split on ``.``, ``!``, ``?``),
        or 0.0 if the text has no words.
    """
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    word_counts = [len(_SENTENCE_WORD_RE.findall(s)) for s in sentences]
    word_counts = [c for c in word_counts if c > 0]
    if not word_counts:
        return 0.0
    return sum(word_counts) / len(word_counts)


def extract_features(transcript: Transcript) -> dict[str, float]:
    """Extract fluency features from a transcribed spoken answer.

    Args:
        transcript: The transcript to analyze, including word-level
            timestamps and total duration.

    Returns:
        A dict with the following keys: ``words_per_minute``,
        ``pause_count``, ``long_pause_count``, ``mean_pause_duration``,
        ``total_pause_ratio``, ``filler_count``, ``filler_rate``,
        ``repetition_count``, ``type_token_ratio``, ``mean_sentence_length``.
    """
    words = transcript.words
    word_count = len(words)
    duration = transcript.duration_seconds

    words_per_minute = (word_count / duration * 60) if duration > 0 else 0.0

    gaps = _pause_gaps(words)
    pauses = [g for g in gaps if g > config.PAUSE_THRESHOLD_SECONDS]
    long_pauses = [g for g in pauses if g > config.LONG_PAUSE_THRESHOLD_SECONDS]
    mean_pause_duration = sum(pauses) / len(pauses) if pauses else 0.0
    total_pause_ratio = sum(pauses) / duration if duration > 0 else 0.0

    tokens = [t for t in (_normalize_token(w.word) for w in words) if t]
    filler_count = _count_fillers(tokens)
    filler_rate = (filler_count / word_count * 100) if word_count > 0 else 0.0
    repetition_count = _count_repetitions(tokens)
    type_token_ratio = len(set(tokens)) / len(tokens) if tokens else 0.0

    return {
        "words_per_minute": words_per_minute,
        "pause_count": len(pauses),
        "long_pause_count": len(long_pauses),
        "mean_pause_duration": mean_pause_duration,
        "total_pause_ratio": total_pause_ratio,
        "filler_count": filler_count,
        "filler_rate": filler_rate,
        "repetition_count": repetition_count,
        "type_token_ratio": type_token_ratio,
        "mean_sentence_length": _mean_sentence_length(transcript.text),
    }


def explain_features(features: dict[str, float]) -> list[str]:
    """Turn extracted features into human-readable coaching tips.

    Args:
        features: A feature dict as returned by :func:`extract_features`.

    Returns:
        A list of short, plain-language tips, one per notable metric.
    """
    tips: list[str] = []

    wpm = features["words_per_minute"]
    if wpm == 0:
        tips.append("No speech was detected to estimate your speaking pace.")
    elif wpm > config.IDEAL_WPM_MAX:
        tips.append(
            f"You spoke at {wpm:.0f} WPM — a bit fast. "
            f"Aim for {config.IDEAL_WPM_MIN:.0f}-{config.IDEAL_WPM_MAX:.0f}."
        )
    elif wpm < config.IDEAL_WPM_MIN:
        tips.append(
            f"You spoke at {wpm:.0f} WPM — a bit slow. "
            f"Aim for {config.IDEAL_WPM_MIN:.0f}-{config.IDEAL_WPM_MAX:.0f}."
        )
    else:
        tips.append(f"You spoke at {wpm:.0f} WPM — a comfortable, natural pace.")

    filler_rate = features["filler_rate"]
    if filler_rate > config.FILLER_RATE_HIGH_PER_100_WORDS:
        tips.append(
            f"You used {features['filler_count']:.0f} filler words "
            f"({filler_rate:.1f} per 100 words, e.g. um/uh/like). Try pausing "
            "silently instead."
        )
    else:
        tips.append("Minimal filler word usage — nice job.")

    if features["long_pause_count"] > 0:
        tips.append(
            f"You had {features['long_pause_count']:.0f} long pause(s) over "
            f"{config.LONG_PAUSE_THRESHOLD_SECONDS:.1f}s. Try to keep your "
            "momentum going."
        )
    if features["total_pause_ratio"] > config.TOTAL_PAUSE_RATIO_HIGH:
        tips.append(
            f"You spent {features['total_pause_ratio'] * 100:.0f}% of your "
            "answer in silence. Practicing the material may help it flow more."
        )

    if features["repetition_count"] > 0:
        tips.append(
            f"You repeated a word back-to-back {features['repetition_count']:.0f} "
            "time(s). Slowing down slightly can help avoid this."
        )

    ttr = features["type_token_ratio"]
    if ttr < config.LOW_TYPE_TOKEN_RATIO:
        tips.append(
            f"Your vocabulary variety was on the low side (TTR={ttr:.2f}). Try "
            "varying your word choice instead of repeating the same words."
        )
    else:
        tips.append(f"Good vocabulary variety (TTR={ttr:.2f}).")

    sentence_length = features["mean_sentence_length"]
    if sentence_length > config.LONG_SENTENCE_WORD_COUNT:
        tips.append(
            f"Your sentences ran long on average ({sentence_length:.0f} words). "
            "Consider breaking answers into shorter sentences."
        )
    elif 0 < sentence_length < config.SHORT_SENTENCE_WORD_COUNT:
        tips.append(
            f"Your sentences were quite short on average ({sentence_length:.0f} "
            "words), which can come across as choppy."
        )

    return tips
