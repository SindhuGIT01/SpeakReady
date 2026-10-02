"""Answer replay timeline: filler/pause/pace markers over a transcript (Task 12).

:func:`build_timeline` turns a :class:`~src.speech.Transcript` (Task 4) into a
flat, time-ordered list of :class:`Marker` — one per filler word, long pause,
or fast-speech stretch — for the Streamlit interview page to render as a
visual timeline under the audio player.

Filler detection reuses :func:`src.features.find_filler_words` (the same
matching logic the ML fluency features are built on, Task 5), so the
timeline and the fluency score never disagree about what counts as a filler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src import config
from src.features import find_filler_words, pause_gaps
from src.speech import Transcript, Word

MarkerType = Literal["filler", "long_pause", "fast_speech"]


@dataclass(frozen=True)
class Marker:
    """One moment of interest on the answer replay timeline.

    Attributes:
        type: ``"filler"``, ``"long_pause"``, or ``"fast_speech"``.
        start: Start time in seconds from the start of the answer.
        end: End time in seconds (equal to ``start`` for a single instant).
        label: A short, human-readable description, e.g.
            ``'filler: "um" at 4.2s'``.
    """

    type: MarkerType
    start: float
    end: float
    label: str


def _filler_markers(transcript: Transcript) -> list[Marker]:
    """Build one "filler" marker per filler word/phrase occurrence.

    Args:
        transcript: The transcript to scan.

    Returns:
        One :class:`Marker` per occurrence, in transcript order.
    """
    markers = []
    for span in find_filler_words(transcript):
        text = " ".join(w.word for w in span)
        markers.append(
            Marker(
                type="filler",
                start=span[0].start,
                end=span[-1].end,
                label=f'filler: "{text}" at {span[0].start:.1f}s',
            )
        )
    return markers


def _long_pause_markers(words: list[Word]) -> list[Marker]:
    """Build one "long_pause" marker per gap over the configured threshold.

    Args:
        words: Word-level timestamps, in speaking order.

    Returns:
        One :class:`Marker` per long pause, in transcript order.
    """
    gaps = pause_gaps(words)
    markers = []
    for i, gap in enumerate(gaps):
        if gap > config.LONG_PAUSE_THRESHOLD_SECONDS:
            before, after = words[i], words[i + 1]
            markers.append(
                Marker(
                    type="long_pause",
                    start=before.end,
                    end=after.start,
                    label=f"long pause: {gap:.1f}s",
                )
            )
    return markers


def _fast_speech_word_runs(words: list[Word]) -> list[tuple[int, int]]:
    """Find contiguous runs of words whose local pace exceeds the threshold.

    Slides a ``config.FAST_SPEECH_WINDOW_WORDS``-word window across ``words``;
    any word inside a window whose local WPM exceeds
    ``config.FAST_SPEECH_WPM_THRESHOLD`` is flagged, and adjacent flagged
    words are merged into a single run so one fast stretch produces one
    marker rather than one per overlapping window.

    Args:
        words: Word-level timestamps, in speaking order.

    Returns:
        ``(start_index, end_index)`` pairs (inclusive) into ``words``, in
        order.
    """
    window = config.FAST_SPEECH_WINDOW_WORDS
    n = len(words)
    if n < window:
        return []

    flagged = [False] * n
    for i in range(n - window + 1):
        window_words = words[i : i + window]
        duration = window_words[-1].end - window_words[0].start
        if duration <= 0:
            continue
        local_wpm = window / duration * 60
        if local_wpm > config.FAST_SPEECH_WPM_THRESHOLD:
            for j in range(i, i + window):
                flagged[j] = True

    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for idx, is_flagged in enumerate(flagged):
        if is_flagged and run_start is None:
            run_start = idx
        elif not is_flagged and run_start is not None:
            runs.append((run_start, idx - 1))
            run_start = None
    if run_start is not None:
        runs.append((run_start, n - 1))
    return runs


def _fast_speech_markers(words: list[Word]) -> list[Marker]:
    """Build one "fast_speech" marker per over-threshold stretch.

    Args:
        words: Word-level timestamps, in speaking order.

    Returns:
        One :class:`Marker` per fast stretch, in transcript order.
    """
    markers = []
    for start_idx, end_idx in _fast_speech_word_runs(words):
        run = words[start_idx : end_idx + 1]
        duration = run[-1].end - run[0].start
        wpm = (len(run) / duration * 60) if duration > 0 else 0.0
        markers.append(
            Marker(
                type="fast_speech",
                start=run[0].start,
                end=run[-1].end,
                label=f"fast speech: ~{wpm:.0f} WPM",
            )
        )
    return markers


def build_timeline(transcript: Transcript) -> list[Marker]:
    """Build the full, time-ordered marker timeline for one answer.

    Args:
        transcript: The transcribed spoken answer (Task 4), with word-level
            timestamps.

    Returns:
        All "filler", "long_pause", and "fast_speech" markers, sorted by
        ``start`` (ties broken by type, for a stable, deterministic order).
    """
    words = transcript.words
    markers = [
        *_filler_markers(transcript),
        *_long_pause_markers(words),
        *_fast_speech_markers(words),
    ]
    markers.sort(key=lambda m: (m.start, m.type))
    return markers
