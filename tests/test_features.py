"""Tests for fluency feature extraction.

Transcripts are hand-built with known word timings and text so every expected
value below is computed by hand, not just re-derived from the implementation.
"""

from __future__ import annotations

import pytest

from src import config
from src.features import explain_features, extract_features
from src.speech import Transcript, Word


def _words(*specs: tuple[str, float, float]) -> list[Word]:
    """Build a list of Word objects from (text, start, end) tuples."""
    return [Word(word=w, start=s, end=e) for w, s, e in specs]


# --- pace and pauses -------------------------------------------------------------


def test_words_per_minute_and_pauses() -> None:
    """5 words over a 6s clip, with one short pause and one long pause."""
    words = _words(
        ("we", 0.0, 0.5),
        ("are", 1.2, 1.5),  # gap 0.7s -> pause
        ("testing", 1.5, 2.0),  # gap 0.0s -> no pause
        ("pauses", 4.0, 4.5),  # gap 2.0s -> pause + long pause
        ("now", 4.5, 5.0),  # gap 0.0s -> no pause
    )
    transcript = Transcript(text="we are testing pauses now", words=words, duration_seconds=6.0)

    features = extract_features(transcript)

    assert features["words_per_minute"] == pytest.approx(50.0)
    assert features["pause_count"] == 2
    assert features["long_pause_count"] == 1
    assert features["mean_pause_duration"] == pytest.approx(1.35)
    assert features["total_pause_ratio"] == pytest.approx(0.45)


def test_no_pauses_when_words_run_back_to_back() -> None:
    """Consecutive words with zero gap should report no pauses at all."""
    words = _words(("a", 0.0, 0.2), ("b", 0.2, 0.4), ("c", 0.4, 0.6))
    transcript = Transcript(text="a b c", words=words, duration_seconds=0.6)

    features = extract_features(transcript)

    assert features["pause_count"] == 0
    assert features["long_pause_count"] == 0
    assert features["mean_pause_duration"] == 0.0
    assert features["total_pause_ratio"] == 0.0


def test_zero_duration_does_not_raise() -> None:
    """A transcript with no duration/words should return safe zero values."""
    transcript = Transcript(text="", words=[], duration_seconds=0.0)

    features = extract_features(transcript)

    assert features["words_per_minute"] == 0.0
    assert features["total_pause_ratio"] == 0.0
    assert features["filler_rate"] == 0.0
    assert features["type_token_ratio"] == 0.0
    assert features["mean_sentence_length"] == 0.0


# --- fillers and repetition -------------------------------------------------------


def _filler_repetition_transcript() -> Transcript:
    """24-word transcript with 5 fillers (including one skipped verb "like")
    and one back-to-back repetition ("the the").
    """
    text = (
        "Um, so, you know, I think this is like a good example. "
        "It was, like, amazing, and I like pizza. The the cat sat."
    )
    tokens = [
        "Um,", "so,", "you", "know,", "I", "think", "this", "is", "like", "a",
        "good", "example.", "It", "was,", "like,", "amazing,", "and", "I",
        "like", "pizza.", "The", "the", "cat", "sat.",
    ]
    words = [Word(word=w, start=i * 0.3, end=i * 0.3 + 0.25) for i, w in enumerate(tokens)]
    return Transcript(text=text, words=words, duration_seconds=len(tokens) * 0.3)


def test_filler_count_skips_verb_usage_of_like() -> None:
    """'um', 'so', 'you know', and two fillerish 'like's should count; the
    verb usage in 'I like pizza' should not.
    """
    features = extract_features(_filler_repetition_transcript())

    assert features["filler_count"] == 5
    assert features["filler_rate"] == pytest.approx(5 / 24 * 100)


def test_repetition_count_counts_back_to_back_duplicates() -> None:
    """'The the cat sat' contains exactly one immediate repetition."""
    features = extract_features(_filler_repetition_transcript())

    assert features["repetition_count"] == 1


def test_type_token_ratio() -> None:
    """20 unique tokens out of 24 total."""
    features = extract_features(_filler_repetition_transcript())

    assert features["type_token_ratio"] == pytest.approx(20 / 24)


def test_mean_sentence_length() -> None:
    """Three sentences of 12, 8, and 4 words average to 8.0."""
    features = extract_features(_filler_repetition_transcript())

    assert features["mean_sentence_length"] == pytest.approx(8.0)


def test_like_as_verb_not_counted_as_filler() -> None:
    """A transcript that only uses 'like' as a verb should have zero fillers."""
    tokens = ["I", "like", "pizza", "and", "you", "like", "tacos"]
    words = [Word(word=w, start=i * 0.3, end=i * 0.3 + 0.25) for i, w in enumerate(tokens)]
    transcript = Transcript(
        text="I like pizza and you like tacos.", words=words, duration_seconds=2.1
    )

    features = extract_features(transcript)

    assert features["filler_count"] == 0


def test_you_know_phrase_counts_once() -> None:
    """The two-word filler phrase 'you know' should count as one filler, not two."""
    tokens = ["you", "know", "it", "was", "great"]
    words = [Word(word=w, start=i * 0.3, end=i * 0.3 + 0.25) for i, w in enumerate(tokens)]
    transcript = Transcript(text="you know it was great", words=words, duration_seconds=1.5)

    features = extract_features(transcript)

    assert features["filler_count"] == 1


# --- explain_features -------------------------------------------------------------


def test_explain_features_flags_fast_pace() -> None:
    """A high WPM should produce a 'bit fast' tip naming the ideal range."""
    features = {
        "words_per_minute": 185.0,
        "pause_count": 0,
        "long_pause_count": 0,
        "mean_pause_duration": 0.0,
        "total_pause_ratio": 0.0,
        "filler_count": 0,
        "filler_rate": 0.0,
        "repetition_count": 0,
        "type_token_ratio": 0.8,
        "mean_sentence_length": 15.0,
    }

    tips = explain_features(features)

    assert any(
        "185 WPM" in t and "a bit fast" in t
        and f"{config.IDEAL_WPM_MIN:.0f}-{config.IDEAL_WPM_MAX:.0f}" in t
        for t in tips
    )


def test_explain_features_flags_high_filler_rate_and_long_pauses() -> None:
    """High filler rate and a long pause should each surface their own tip."""
    features = {
        "words_per_minute": 145.0,
        "pause_count": 3,
        "long_pause_count": 2,
        "mean_pause_duration": 2.0,
        "total_pause_ratio": 0.4,
        "filler_count": 10,
        "filler_rate": 10.0,
        "repetition_count": 2,
        "type_token_ratio": 0.3,
        "mean_sentence_length": 35.0,
    }

    tips = explain_features(features)

    assert any("filler words" in t for t in tips)
    assert any("long pause" in t for t in tips)
    assert any("silence" in t for t in tips)
    assert any("repeated a word" in t for t in tips)
    assert any("vocabulary variety" in t and "low" in t for t in tips)
    assert any("sentences ran long" in t for t in tips)


def test_explain_features_praises_good_metrics() -> None:
    """Metrics well within healthy ranges should yield positive feedback."""
    features = {
        "words_per_minute": 145.0,
        "pause_count": 1,
        "long_pause_count": 0,
        "mean_pause_duration": 0.6,
        "total_pause_ratio": 0.05,
        "filler_count": 1,
        "filler_rate": 1.0,
        "repetition_count": 0,
        "type_token_ratio": 0.7,
        "mean_sentence_length": 14.0,
    }

    tips = explain_features(features)

    assert any("comfortable, natural pace" in t for t in tips)
    assert any("Minimal filler word usage" in t for t in tips)
    assert any("Good vocabulary variety" in t for t in tips)
