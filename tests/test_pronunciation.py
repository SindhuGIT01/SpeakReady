"""Tests for the "words to double check" pronunciation proxy signal (Task 13).

Pure logic over :class:`~src.speech.Transcript`/:class:`~src.speech.Segment`
objects — no API calls or audio decoding involved.
"""

from __future__ import annotations

from src import config
from src.pronunciation import PronunciationSignal, flag_low_confidence_words
from src.speech import Segment, Transcript, Word


def _word(text: str, start: float, end: float) -> Word:
    return Word(word=text, start=start, end=end)


def test_no_segments_returns_empty_with_explanation() -> None:
    """A transcript with no segment data should return an empty, explained result."""
    transcript = Transcript(
        text="Hello there.",
        words=[_word("Hello", 0.0, 0.5), _word("there.", 0.5, 1.0)],
        duration_seconds=1.0,
        segments=[],
    )

    result = flag_low_confidence_words(transcript)

    assert isinstance(result, PronunciationSignal)
    assert result.words_to_double_check == []
    assert "No segment-level confidence data" in result.explanation


def test_confident_segments_flag_nothing() -> None:
    """Words inside comfortably-confident segments shouldn't be flagged."""
    transcript = Transcript(
        text="Hello there.",
        words=[_word("Hello", 0.0, 0.5), _word("there.", 0.5, 1.0)],
        duration_seconds=1.0,
        segments=[Segment(start=0.0, end=1.0, avg_logprob=-0.1)],
    )

    result = flag_low_confidence_words(transcript)

    assert result.words_to_double_check == []
    assert "confident" in result.explanation.lower()


def test_words_in_low_confidence_segment_are_flagged() -> None:
    """Words whose midpoint falls in a low-avg_logprob segment should be flagged."""
    transcript = Transcript(
        text="Hello mumble there.",
        words=[
            _word("Hello", 0.0, 0.5),
            _word("mumble", 0.5, 1.0),
            _word("there.", 1.0, 1.5),
        ],
        duration_seconds=1.5,
        segments=[
            Segment(start=0.0, end=0.5, avg_logprob=-0.1),
            Segment(start=0.5, end=1.0, avg_logprob=-0.9),
            Segment(start=1.0, end=1.5, avg_logprob=-0.1),
        ],
    )

    result = flag_low_confidence_words(transcript)

    flagged_words = [w.word for w in result.words_to_double_check]
    assert flagged_words == ["mumble"]
    assert result.words_to_double_check[0].segment_avg_logprob == -0.9
    assert '"mumble"' in result.explanation


def test_confidence_threshold_is_configurable(monkeypatch) -> None:
    """A stricter (less negative) threshold should flag more segments."""
    transcript = Transcript(
        text="Hello there.",
        words=[_word("Hello", 0.0, 0.5), _word("there.", 0.5, 1.0)],
        duration_seconds=1.0,
        segments=[Segment(start=0.0, end=1.0, avg_logprob=-0.3)],
    )

    result_default = flag_low_confidence_words(transcript)
    assert result_default.words_to_double_check == []

    monkeypatch.setattr(config, "PRONUNCIATION_LOW_CONFIDENCE_LOGPROB", -0.2)
    result_strict = flag_low_confidence_words(transcript)

    assert len(result_strict.words_to_double_check) == 2


def test_word_outside_any_segment_is_not_flagged() -> None:
    """A word whose timestamp falls in a gap between segments should be skipped."""
    transcript = Transcript(
        text="Hello there.",
        words=[_word("Hello", 0.0, 0.5), _word("there.", 2.0, 2.5)],
        duration_seconds=2.5,
        segments=[Segment(start=0.0, end=0.5, avg_logprob=-0.9)],
    )

    result = flag_low_confidence_words(transcript)

    assert [w.word for w in result.words_to_double_check] == ["Hello"]


def test_explanation_previews_at_most_five_words() -> None:
    """The explanation should preview a handful of flagged words, not the full list."""
    words = [_word(f"word{i}", float(i), float(i) + 0.9) for i in range(8)]
    transcript = Transcript(
        text=" ".join(w.word for w in words),
        words=words,
        duration_seconds=8.0,
        segments=[Segment(start=0.0, end=8.0, avg_logprob=-0.9)],
    )

    result = flag_low_confidence_words(transcript)

    assert len(result.words_to_double_check) == 8
    assert result.explanation.count('"') == 10  # 5 previewed words, quoted
