"""Per-word pronunciation signal (Task 13).

True phoneme-level pronunciation scoring — the kind speechocean762's
word-level scores represent, used to train the fluency model in Task 6 —
needs a forced-alignment or phoneme-confidence model running locally.
Groq's hosted Whisper API doesn't expose that: requesting
``timestamp_granularities=["word", "segment"]`` (see :mod:`src.speech`)
only returns word text/timing plus a per-*segment* ``avg_logprob`` — never
a per-*word* probability the way a local engine (faster-whisper,
whisper.cpp) would.

So this module is an honest proxy, not pronunciation scoring: it flags the
words spoken inside the least-confident segments as "words to double
check" — the ASR had trouble with that stretch of audio, which could be
pronunciation, mumbling, background noise, or an unusual/rare word. Treat
it as a hint of where to listen back, never a verdict on how a word was
pronounced.
"""

from __future__ import annotations

from dataclasses import dataclass

from src import config
from src.speech import Segment, Transcript, Word


@dataclass(frozen=True)
class WordToDoubleCheck:
    """One word spoken inside a low-confidence ASR segment.

    Attributes:
        word: The transcribed word text.
        start: Start time in seconds.
        end: End time in seconds.
        segment_avg_logprob: The ``avg_logprob`` of the segment this word
            fell in; closer to 0 is more confident, so a more negative
            value here means the ASR was less sure about this stretch.
    """

    word: str
    start: float
    end: float
    segment_avg_logprob: float


@dataclass(frozen=True)
class PronunciationSignal:
    """Proxy "words to double check" signal for one spoken answer.

    Attributes:
        words_to_double_check: Words that fell inside the least-confident
            segments, in transcript order. Empty when every segment was
            comfortably confident, or when no segment data was available.
        explanation: A short, honest explanation of what this signal is
            (and isn't) — see the module docstring.
    """

    words_to_double_check: list[WordToDoubleCheck]
    explanation: str


_PROXY_DISCLAIMER = (
    "Not true pronunciation scoring — Groq's Whisper API only exposes ASR "
    "confidence per segment, not per word, so this highlights words from the "
    "least-confident stretches of audio as worth a second listen."
)


def _word_segment(word: Word, segments: list[Segment]) -> Segment | None:
    """Find the segment a word's midpoint falls inside.

    Args:
        word: The word to locate.
        segments: The transcript's segments, in order.

    Returns:
        The containing :class:`Segment`, or ``None`` if the word's
        timestamp doesn't fall inside any segment.
    """
    midpoint = (word.start + word.end) / 2
    for segment in segments:
        if segment.start <= midpoint <= segment.end:
            return segment
    return None


def flag_low_confidence_words(transcript: Transcript) -> PronunciationSignal:
    """Flag words from the ASR's least-confident segments.

    Args:
        transcript: A transcribed answer (Task 4) with word timestamps and
            segment-level confidence (``transcript.segments``).

    Returns:
        The :class:`PronunciationSignal` for this answer. Both
        ``words_to_double_check`` and the matching explanation reflect
        whatever segment/word data was actually available.
    """
    if not transcript.segments:
        return PronunciationSignal(
            words_to_double_check=[],
            explanation=f"No segment-level confidence data was available for this answer. {_PROXY_DISCLAIMER}",
        )

    flagged: list[WordToDoubleCheck] = []
    for word in transcript.words:
        segment = _word_segment(word, transcript.segments)
        if segment is not None and segment.avg_logprob < config.PRONUNCIATION_LOW_CONFIDENCE_LOGPROB:
            flagged.append(
                WordToDoubleCheck(
                    word=word.word,
                    start=word.start,
                    end=word.end,
                    segment_avg_logprob=segment.avg_logprob,
                )
            )

    if not flagged:
        explanation = (
            f"The ASR was confident throughout this answer — no words to double check. {_PROXY_DISCLAIMER}"
        )
    else:
        words_preview = ", ".join(f'"{w.word}"' for w in flagged[:5])
        explanation = f"Words worth a second listen: {words_preview}. {_PROXY_DISCLAIMER}"

    return PronunciationSignal(words_to_double_check=flagged, explanation=explanation)
