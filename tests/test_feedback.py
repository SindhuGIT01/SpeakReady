"""Tests for the LLM feedback engine.

LLM calls are mocked throughout so these tests run without a GROQ_API_KEY.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src import config
from src.feedback import (
    ContentScore,
    Feedback,
    FeedbackError,
    GrammarCorrection,
    _FeedbackLLMOutput,
    _summarize_filler_usage,
    compute_overall_score,
    generate_feedback,
)
from src.scorer import FluencyPrediction

FIXTURE_FEATURES: dict[str, float] = {
    "words_per_minute": 140.0,
    "pause_count": 2,
    "long_pause_count": 0,
    "mean_pause_duration": 0.6,
    "total_pause_ratio": 0.08,
    "filler_count": 1,
    "filler_rate": 1.2,
    "repetition_count": 0,
    "type_token_ratio": 0.75,
    "mean_sentence_length": 12.0,
}

FIXTURE_FLUENCY_RESULT = FluencyPrediction(label="Fluent", confidence=0.8, score=88.0)


def _llm_output(**overrides) -> _FeedbackLLMOutput:
    """A well-formed LLM output, with any fields overridden for a test."""
    defaults = dict(
        content_score=ContentScore(score=7.0, reason="Solid answer, missed one point."),
        key_points_covered=["Relevant experience"],
        key_points_missed=["Specific metrics"],
        grammar_corrections=[
            GrammarCorrection(
                original="I have went",
                corrected="I have gone",
                explanation="Incorrect past participle.",
            )
        ],
        improved_answer="In my previous role, I led a project that...",
        one_tip="Back up your claims with specific numbers.",
    )
    defaults.update(overrides)
    return _FeedbackLLMOutput(**defaults)


def _mock_llm(invoke_return=None, invoke_side_effect=None) -> MagicMock:
    """A mock chat model whose with_structured_output().invoke() is stubbed."""
    mock_structured = MagicMock()
    if invoke_side_effect is not None:
        mock_structured.invoke.side_effect = invoke_side_effect
    else:
        mock_structured.invoke.return_value = invoke_return
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    return mock_llm


# --- _summarize_filler_usage -------------------------------------------------


def test_summarize_filler_usage_no_fillers() -> None:
    """Zero fillers should be praised, not flagged."""
    summary = _summarize_filler_usage({"filler_count": 0, "filler_rate": 0.0})

    assert "No filler words" in summary


def test_summarize_filler_usage_high_rate_flags_it() -> None:
    """A filler rate above the configured threshold should be called out."""
    summary = _summarize_filler_usage(
        {"filler_count": 8, "filler_rate": config.FILLER_RATE_HIGH_PER_100_WORDS + 1}
    )

    assert "above the ideal range" in summary
    assert "8" in summary


def test_summarize_filler_usage_normal_rate() -> None:
    """A filler rate within the normal range should not be flagged as high."""
    summary = _summarize_filler_usage(
        {"filler_count": 1, "filler_rate": config.FILLER_RATE_HIGH_PER_100_WORDS - 1}
    )

    assert "normal range" in summary
    assert "above the ideal range" not in summary


# --- compute_overall_score --------------------------------------------------


def test_compute_overall_score_is_weighted_average() -> None:
    """The overall score should combine content (x10) and fluency by config weights."""
    result = compute_overall_score(content_score=8.0, fluency_score=70.0)

    expected = 8.0 * 10 * config.CONTENT_SCORE_WEIGHT + 70.0 * config.FLUENCY_SCORE_WEIGHT
    assert result == pytest.approx(expected, abs=0.05)


def test_compute_overall_score_perfect_scores_yield_100() -> None:
    """A perfect content score and perfect fluency score should combine to 100."""
    result = compute_overall_score(content_score=10.0, fluency_score=100.0)

    assert result == pytest.approx(100.0)


# --- generate_feedback: happy path ------------------------------------------


def test_generate_feedback_uses_structured_output() -> None:
    """generate_feedback should request structured output shaped like _FeedbackLLMOutput."""
    mock_llm = _mock_llm(invoke_return=_llm_output())

    generate_feedback(
        question="Tell me about yourself.",
        transcript="I am a software engineer with three years of experience.",
        features=FIXTURE_FEATURES,
        fluency_result=FIXTURE_FLUENCY_RESULT,
        llm=mock_llm,
    )

    mock_llm.with_structured_output.assert_called_once_with(_FeedbackLLMOutput)


def test_generate_feedback_returns_full_feedback() -> None:
    """The returned Feedback should carry through LLM fields and computed ones."""
    mock_llm = _mock_llm(invoke_return=_llm_output())

    feedback = generate_feedback(
        question="Tell me about yourself.",
        transcript="I am a software engineer with three years of experience.",
        features=FIXTURE_FEATURES,
        fluency_result=FIXTURE_FLUENCY_RESULT,
        llm=mock_llm,
    )

    assert isinstance(feedback, Feedback)
    assert feedback.content_score.score == 7.0
    assert feedback.key_points_covered == ["Relevant experience"]
    assert feedback.key_points_missed == ["Specific metrics"]
    assert feedback.grammar_corrections[0].corrected == "I have gone"
    assert feedback.one_tip == "Back up your claims with specific numbers."
    assert "1 filler word" in feedback.filler_summary
    assert feedback.overall_score == pytest.approx(
        compute_overall_score(7.0, FIXTURE_FLUENCY_RESULT.score)
    )


def test_generate_feedback_filler_summary_ignores_llm() -> None:
    """filler_summary must come from features, never from the LLM's own text."""
    mock_llm = _mock_llm(invoke_return=_llm_output())

    feedback = generate_feedback(
        question="Tell me about yourself.",
        transcript="...",
        features={"filler_count": 0, "filler_rate": 0.0},
        fluency_result=FIXTURE_FLUENCY_RESULT,
        llm=mock_llm,
    )

    assert feedback.filler_summary == _summarize_filler_usage(
        {"filler_count": 0, "filler_rate": 0.0}
    )


def test_generate_feedback_passes_optional_context_into_prompt() -> None:
    """what_good_answer_covers and resume_context should reach the LLM's prompt."""
    mock_llm = _mock_llm(invoke_return=_llm_output())

    generate_feedback(
        question="Tell me about yourself.",
        transcript="I am a software engineer.",
        features=FIXTURE_FEATURES,
        fluency_result=FIXTURE_FLUENCY_RESULT,
        what_good_answer_covers=["Career goal", "Relevant skills"],
        resume_context="Skills: Python, SQL",
        llm=mock_llm,
    )

    structured_model = mock_llm.with_structured_output.return_value
    messages = structured_model.invoke.call_args[0][0]
    user_message_text = messages[1].content

    assert "Career goal" in user_message_text
    assert "Skills: Python, SQL" in user_message_text


# --- generate_feedback: retry / error handling ------------------------------


def test_generate_feedback_retries_on_invalid_output() -> None:
    """An invalid first response should be retried and recovered from."""
    mock_llm = _mock_llm(invoke_side_effect=[ValueError("bad json"), _llm_output()])

    feedback = generate_feedback(
        question="Tell me about yourself.",
        transcript="I am a software engineer.",
        features=FIXTURE_FEATURES,
        fluency_result=FIXTURE_FLUENCY_RESULT,
        llm=mock_llm,
    )

    structured_model = mock_llm.with_structured_output.return_value
    assert structured_model.invoke.call_count == 2
    assert isinstance(feedback, Feedback)


def test_generate_feedback_raises_after_exhausting_retries(monkeypatch) -> None:
    """Persistent invalid output should raise FeedbackError, not propagate raw."""
    monkeypatch.setattr(config, "FEEDBACK_LLM_MAX_RETRIES", 1)
    mock_llm = _mock_llm(invoke_side_effect=ValueError("always bad"))

    with pytest.raises(FeedbackError):
        generate_feedback(
            question="Tell me about yourself.",
            transcript="I am a software engineer.",
            features=FIXTURE_FEATURES,
            fluency_result=FIXTURE_FLUENCY_RESULT,
            llm=mock_llm,
        )

    structured_model = mock_llm.with_structured_output.return_value
    assert structured_model.invoke.call_count == 2
