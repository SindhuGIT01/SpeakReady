"""LLM-based content feedback for one interview answer.

Combines the LLM's judgment of *what* the candidate said (content_score,
key points, grammar, a rewritten answer, one tip) with the filler-word
summary computed deterministically from :func:`src.features.extract_features`
(never guessed by the LLM), and folds in the ML fluency score from
:func:`src.scorer.predict_fluency` to produce one overall 0-100 score.
"""

from __future__ import annotations

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from src import config, prompts
from src.scorer import FluencyPrediction


class FeedbackError(RuntimeError):
    """Raised when the LLM fails to produce valid structured feedback."""


class ContentScore(BaseModel):
    """The LLM's score for how well an answer covered the question's content."""

    score: float = Field(ge=0, le=10, description="Content quality score, 0-10.")
    reason: str = Field(description="A one-sentence explanation for the score.")


class GrammarCorrection(BaseModel):
    """A single real grammar or phrasing mistake found in the transcript."""

    original: str = Field(description="The exact mistaken phrase from the transcript.")
    corrected: str = Field(description="The corrected version of that phrase.")
    explanation: str = Field(description="A short explanation of the mistake.")


class _FeedbackLLMOutput(BaseModel):
    """The LLM's structured output for one answer.

    Excludes ``filler_summary``, which :func:`generate_feedback` computes
    deterministically from speech features rather than asking the LLM.
    """

    content_score: ContentScore
    key_points_covered: list[str] = Field(default_factory=list)
    key_points_missed: list[str] = Field(default_factory=list)
    grammar_corrections: list[GrammarCorrection] = Field(default_factory=list)
    improved_answer: str
    one_tip: str


class Feedback(BaseModel):
    """Full feedback for one interview answer.

    Attributes:
        content_score: The LLM's 0-10 content score and its reason.
        key_points_covered: Points from the answer that a strong answer
            should cover, which this answer did cover.
        key_points_missed: Points a strong answer should cover, which this
            answer missed.
        grammar_corrections: Real grammar/phrasing mistakes found, each with
            a correction and explanation.
        filler_summary: A plain-language summary of filler word usage,
            computed from speech features (not the LLM).
        improved_answer: A better version of the answer, in the candidate's
            own words and honest to what they actually said/their resume.
        one_tip: The single most important thing to improve.
        overall_score: Combined 0-100 score from fluency and content, per
            :func:`compute_overall_score`.
    """

    content_score: ContentScore
    key_points_covered: list[str]
    key_points_missed: list[str]
    grammar_corrections: list[GrammarCorrection]
    filler_summary: str
    improved_answer: str
    one_tip: str
    overall_score: float


def compute_overall_score(content_score: float, fluency_score: float) -> float:
    """Combine the LLM content score and ML fluency score into one 0-100 score.

    Args:
        content_score: The LLM's content score, on a 0-10 scale.
        fluency_score: The fluency model's score, on a 0-100 scale (see
            :class:`src.scorer.FluencyPrediction`).

    Returns:
        A weighted 0-100 score using ``config.CONTENT_SCORE_WEIGHT`` and
        ``config.FLUENCY_SCORE_WEIGHT``.
    """
    weighted = (
        content_score * 10 * config.CONTENT_SCORE_WEIGHT
        + fluency_score * config.FLUENCY_SCORE_WEIGHT
    )
    return round(weighted, 1)


def _summarize_filler_usage(features: dict[str, float]) -> str:
    """Turn filler-word features into a plain-language summary.

    Computed from :func:`src.features.extract_features` output directly, so
    the filler summary is never guessed or altered by the LLM.

    Args:
        features: A feature dict as returned by
            :func:`src.features.extract_features`.

    Returns:
        A short, human-readable sentence describing filler word usage.
    """
    filler_count = features.get("filler_count", 0)
    filler_rate = features.get("filler_rate", 0.0)

    if filler_count == 0:
        return "No filler words detected — clean delivery."
    if filler_rate > config.FILLER_RATE_HIGH_PER_100_WORDS:
        return (
            f"{filler_count:.0f} filler word(s) detected ({filler_rate:.1f} per "
            "100 words), above the ideal range. Try pausing silently instead of "
            'saying "um"/"uh"/"like".'
        )
    return (
        f"{filler_count:.0f} filler word(s) detected ({filler_rate:.1f} per 100 "
        "words) — within a normal range."
    )


def _invoke_with_retry(structured_model, messages: list) -> _FeedbackLLMOutput:
    """Call the structured-output LLM, retrying on invalid/unparseable JSON.

    Args:
        structured_model: A chat model wrapped with
            ``with_structured_output(_FeedbackLLMOutput)``.
        messages: The messages to send.

    Returns:
        The parsed :class:`_FeedbackLLMOutput`.

    Raises:
        FeedbackError: If every attempt fails to produce valid output.
    """
    last_error: Exception | None = None
    for _attempt in range(config.FEEDBACK_LLM_MAX_RETRIES + 1):
        try:
            result = structured_model.invoke(messages)
            return (
                result
                if isinstance(result, _FeedbackLLMOutput)
                else _FeedbackLLMOutput.model_validate(result)
            )
        except (ValidationError, OutputParserException, ValueError) as exc:
            last_error = exc

    raise FeedbackError(
        "LLM failed to produce valid structured feedback after "
        f"{config.FEEDBACK_LLM_MAX_RETRIES + 1} attempt(s)."
    ) from last_error


def generate_feedback(
    question: str,
    transcript: str,
    features: dict[str, float],
    fluency_result: FluencyPrediction,
    what_good_answer_covers: list[str] | None = None,
    resume_context: str | None = None,
    llm: BaseChatModel | None = None,
) -> Feedback:
    """Generate full feedback for one interview answer.

    Args:
        question: The interview question that was asked.
        transcript: The candidate's transcribed spoken answer.
        features: Fluency features from
            :func:`src.features.extract_features`, used for
            ``filler_summary`` (never passed to the LLM to guess from).
        fluency_result: The ML fluency prediction from
            :func:`src.scorer.predict_fluency`, used for ``overall_score``.
        what_good_answer_covers: Optional list of points a strong answer to
            this question should cover, from the question bank.
        resume_context: Optional resume text/snippets relevant to this
            answer, used to keep ``improved_answer`` honest to the
            candidate's real experience.
        llm: Optional chat model to use (mainly for testing). Defaults to
            :func:`src.llm.get_llm`.

    Returns:
        A :class:`Feedback` with content feedback, a filler summary, and a
        combined overall score.

    Raises:
        FeedbackError: If the LLM fails to produce valid structured output
            after retrying.
        config.ConfigError: If no ``llm`` is given and GROQ_API_KEY is
            missing.
    """
    from src.llm import get_llm

    model = llm if llm is not None else get_llm()
    structured_model = model.with_structured_output(_FeedbackLLMOutput)

    user_prompt = prompts.build_feedback_user_prompt(
        question=question,
        transcript=transcript,
        what_good_answer_covers=what_good_answer_covers,
        resume_context=resume_context,
    )
    messages = [
        SystemMessage(content=prompts.FEEDBACK_SYSTEM_PROMPT_V1),
        HumanMessage(content=user_prompt),
    ]

    llm_output = _invoke_with_retry(structured_model, messages)
    overall_score = compute_overall_score(llm_output.content_score.score, fluency_result.score)

    return Feedback(
        content_score=llm_output.content_score,
        key_points_covered=llm_output.key_points_covered,
        key_points_missed=llm_output.key_points_missed,
        grammar_corrections=llm_output.grammar_corrections,
        filler_summary=_summarize_filler_usage(features),
        improved_answer=llm_output.improved_answer,
        one_tip=llm_output.one_tip,
        overall_score=overall_score,
    )
