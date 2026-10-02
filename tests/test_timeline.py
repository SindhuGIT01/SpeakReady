"""Tests for the answer replay timeline (Task 12).

Transcripts are hand-built with known word timings so every expected marker
position below is computed by hand, not just re-derived from the
implementation (same approach as tests/test_features.py).
"""

from __future__ import annotations

import pytest

from src import config
from src.speech import Transcript, Word
from src.timeline import Marker, build_timeline


def _words(*specs: tuple[str, float, float]) -> list[Word]:
    """Build a list of Word objects from (text, start, end) tuples."""
    return [Word(word=w, start=s, end=e) for w, s, e in specs]


def _transcript(words: list[Word], duration: float | None = None) -> Transcript:
    """A Transcript whose text is just its words joined by spaces."""
    text = " ".join(w.word for w in words)
    return Transcript(
        text=text, words=words, duration_seconds=duration or (words[-1].end if words else 0.0)
    )


# --- no markers -------------------------------------------------------------------


def test_build_timeline_empty_for_clean_answer() -> None:
    """Evenly-paced words with no fillers or long pauses should yield nothing."""
    words = _words(("we", 0.0, 0.3), ("are", 0.3, 0.6), ("doing", 0.6, 0.9), ("fine", 0.9, 1.2))

    assert build_timeline(_transcript(words)) == []


# --- filler markers ----------------------------------------------------------------


def test_filler_marker_position_and_label() -> None:
    """A single filler word should produce one marker at its exact timing."""
    words = _words(("I", 0.0, 0.2), ("um", 0.2, 0.5), ("think", 0.5, 0.8))

    markers = build_timeline(_transcript(words))

    assert markers == [Marker(type="filler", start=0.2, end=0.5, label='filler: "um" at 0.2s')]


def test_multi_word_filler_marker_spans_both_words() -> None:
    """A two-word filler phrase should span from the first to the last word."""
    words = _words(
        ("it", 0.0, 0.2), ("was", 0.2, 0.4), ("you", 0.4, 0.55), ("know", 0.55, 0.7)
    )

    markers = build_timeline(_transcript(words))

    assert len(markers) == 1
    marker = markers[0]
    assert marker.type == "filler"
    assert marker.start == pytest.approx(0.4)
    assert marker.end == pytest.approx(0.7)
    assert "you know" in marker.label


def test_like_as_a_verb_is_not_flagged_as_filler() -> None:
    """"I like pizza" should not raise a filler marker for "like"."""
    words = _words(("I", 0.0, 0.2), ("like", 0.2, 0.4), ("pizza", 0.4, 0.7))

    assert build_timeline(_transcript(words)) == []


# --- long pause markers -------------------------------------------------------------


def test_long_pause_marker_position_and_duration() -> None:
    """A gap over the configured threshold should be marked with its duration."""
    words = _words(("we", 0.0, 0.3), ("paused", 2.1, 2.4))  # 1.8s gap

    markers = build_timeline(_transcript(words))

    assert markers == [Marker(type="long_pause", start=0.3, end=2.1, label="long pause: 1.8s")]


def test_short_pause_below_threshold_is_not_marked() -> None:
    """A 1.0s gap is a pause, but not a *long* one at the default 1.5s threshold."""
    words = _words(("we", 0.0, 0.3), ("paused", 1.3, 1.6))  # 1.0s gap

    assert build_timeline(_transcript(words)) == []


def test_long_pause_threshold_is_configurable(monkeypatch) -> None:
    """Lowering the threshold should surface a pause that was previously too short."""
    monkeypatch.setattr(config, "LONG_PAUSE_THRESHOLD_SECONDS", 0.5)
    words = _words(("we", 0.0, 0.3), ("paused", 1.0, 1.3))  # 0.7s gap

    markers = build_timeline(_transcript(words))

    assert markers == [Marker(type="long_pause", start=0.3, end=1.0, label="long pause: 0.7s")]


# --- fast speech markers -------------------------------------------------------------


def test_fast_speech_marker_for_quick_window(monkeypatch) -> None:
    """A 3-word window spoken well above the WPM threshold should be flagged."""
    monkeypatch.setattr(config, "FAST_SPEECH_WINDOW_WORDS", 3)
    monkeypatch.setattr(config, "FAST_SPEECH_WPM_THRESHOLD", 200.0)
    # 3 words across 0.0 -> 0.5s = 360 WPM, well over 200.
    words = _words(("go", 0.0, 0.1), ("go", 0.2, 0.3), ("go", 0.4, 0.5))

    markers = build_timeline(_transcript(words))

    assert len(markers) == 1
    marker = markers[0]
    assert marker.type == "fast_speech"
    assert marker.start == pytest.approx(0.0)
    assert marker.end == pytest.approx(0.5)
    assert "WPM" in marker.label


def test_fast_speech_not_marked_below_threshold(monkeypatch) -> None:
    """A window right at a normal pace should not be flagged."""
    monkeypatch.setattr(config, "FAST_SPEECH_WINDOW_WORDS", 3)
    monkeypatch.setattr(config, "FAST_SPEECH_WPM_THRESHOLD", 200.0)
    # 3 words across 1.0s = 180 WPM, under 200.
    words = _words(("go", 0.0, 0.2), ("go", 0.4, 0.6), ("go", 0.8, 1.0))

    assert build_timeline(_transcript(words)) == []


def test_fast_speech_merges_overlapping_windows_into_one_run(monkeypatch) -> None:
    """A longer fast stretch should produce one merged marker, not one per window."""
    monkeypatch.setattr(config, "FAST_SPEECH_WINDOW_WORDS", 3)
    monkeypatch.setattr(config, "FAST_SPEECH_WPM_THRESHOLD", 200.0)
    # 6 words, each 0.1s long back-to-back -> every 3-word window is 360 WPM.
    words = _words(*[(f"w{i}", i * 0.1, i * 0.1 + 0.1) for i in range(6)])

    markers = build_timeline(_transcript(words))

    assert len(markers) == 1
    assert markers[0].start == pytest.approx(0.0)
    assert markers[0].end == pytest.approx(0.6)


def test_fast_speech_skipped_when_fewer_words_than_window(monkeypatch) -> None:
    """Fewer words than the rolling window should never raise a fast_speech marker."""
    monkeypatch.setattr(config, "FAST_SPEECH_WINDOW_WORDS", 5)
    words = _words(("go", 0.0, 0.05), ("go", 0.05, 0.1))

    assert build_timeline(_transcript(words)) == []


# --- combined / ordering -------------------------------------------------------------


def test_markers_are_sorted_by_start_time() -> None:
    """Fillers, long pauses, and fast speech should come back in time order."""
    words = _words(
        ("um", 0.0, 0.2),  # filler at 0.0
        ("we", 4.0, 4.2),  # gap 3.8s -> long pause at 0.2-4.0
        ("are", 4.2, 4.3),
        ("done", 4.3, 4.4),
    )

    markers = build_timeline(_transcript(words))

    assert [m.type for m in markers] == ["filler", "long_pause"]
    assert markers[0].start <= markers[1].start


def test_build_timeline_handles_no_words() -> None:
    """An empty transcript (e.g. silence) should return no markers, not raise."""
    assert build_timeline(Transcript(text="", words=[], duration_seconds=0.0)) == []


# --- the Task 4/5 sample transcript --------------------------------------------------


def test_sample_answer_transcript_flags_its_brisk_pace() -> None:
    """The Task 4/5 sample_answer.mp3 transcript: 5 words in 1.4s is ~214 WPM.

    No fillers and no pause over 1.5s, but the default
    FAST_SPEECH_WINDOW_WORDS=5 window exceeds the default 200 WPM threshold
    for this clip, so it should raise exactly one fast_speech marker spanning
    the whole answer.
    """
    words = _words(
        ("Hi,", 0.0, 0.3),
        ("my", 0.3, 0.5),
        ("name", 0.5, 0.8),
        ("is", 0.8, 0.95),
        ("Aditi.", 0.95, 1.4),
    )
    transcript = Transcript(text="Hi, my name is Aditi.", words=words, duration_seconds=1.4)

    markers = build_timeline(transcript)

    assert len(markers) == 1
    assert markers[0].type == "fast_speech"
    assert markers[0].start == pytest.approx(0.0)
    assert markers[0].end == pytest.approx(1.4)
