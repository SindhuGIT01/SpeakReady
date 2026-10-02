"""Interview agent: runs a full mock interview end to end.

:class:`InterviewSession` ties together the question bank (Task 2), resume
RAG (Task 3), speech-to-text (Task 4), fluency features (Task 5), the ML
fluency model (Task 6), and the LLM feedback engine (Task 7) into one
stateful, testable class: build a question plan, ask each question with TTS
audio, score each spoken answer, decide on the fly whether a follow-up is
worth asking, and end with a session summary and a personalized practice
plan.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from statistics import mean
from typing import Any
from uuid import uuid4

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from src import body_language as body_language_mod
from src import features as features_mod
from src import feedback as feedback_mod
from src import prompts, question_bank, scorer, speech, storage, tts
from src import resume as resume_mod
from src.body_language import BodyLanguageResult
from src.feedback import Feedback
from src.resume import ResumeProfile
from src.scorer import FluencyPrediction
from src.speech import Transcript

# Duration assumed for typed (audio-less) answers in submit_text_answer(),
# used only to estimate words_per_minute for the CLI demo and tests.
_DEMO_ASSUMED_WPM = 150.0


class InterviewError(RuntimeError):
    """Raised for invalid :class:`InterviewSession` state transitions."""


# --- Structured LLM outputs ----------------------------------------------------


class FollowUpDecision(BaseModel):
    """The LLM's decision on whether a follow-up question is worth asking."""

    should_follow_up: bool = Field(
        description="Whether a follow-up would meaningfully improve this interview."
    )
    follow_up_question: str = Field(
        default="",
        description=(
            "The follow-up question, referencing what the candidate actually said. "
            "Empty when should_follow_up is false."
        ),
    )
    reason: str = Field(default="", description="A one-sentence reason for the decision.")


class PracticeDay(BaseModel):
    """One day of a personalized post-interview practice plan."""

    day: int = Field(ge=1, le=7)
    focus: str = Field(description="The specific skill or area this day targets.")
    activity: str = Field(description="A concrete, actionable exercise for this day.")


class _PracticePlanLLMOutput(BaseModel):
    """Wrapper so the LLM's structured output for the practice plan is one object."""

    plan: list[PracticeDay]


class SessionSummary(BaseModel):
    """Summary produced by :meth:`InterviewSession.end`."""

    questions_answered: int
    average_overall_score: float
    average_fluency_score: float
    average_content_score: float
    strongest_area: str
    weakest_area: str
    practice_plan: list[PracticeDay]


# --- Session state dataclasses --------------------------------------------------


@dataclass(frozen=True)
class QuestionItem:
    """One question in an interview plan.

    Attributes:
        id: Unique identifier for this question instance.
        text: The question text.
        source: Where the question came from, ``"bank"`` or ``"resume"``.
        area: Category (question-bank questions) or resume section
            (resume questions); used to group scores for the session
            summary. Follow-ups inherit their parent question's area.
        what_good_answer_covers: Key points a strong answer should cover,
            when known (only set for question-bank questions).
        is_followup: Whether this is a follow-up to another question.
        parent_id: The id of the main question this follows up on, if any.
    """

    id: str
    text: str
    source: str
    area: str
    what_good_answer_covers: list[str] | None = None
    is_followup: bool = False
    parent_id: str | None = None


@dataclass(frozen=True)
class NextQuestion:
    """A question ready to be presented to the candidate."""

    question_id: str
    text: str
    audio: bytes
    is_followup: bool
    question_number: int
    total_questions: int


@dataclass(frozen=True)
class AnswerRecord:
    """Everything recorded for one answered question."""

    question_item: QuestionItem
    transcript: Transcript
    features: dict[str, float]
    fluency_result: FluencyPrediction
    feedback: Feedback
    answered_at: datetime
    body_language_result: BodyLanguageResult | None = None


@dataclass(frozen=True)
class AnswerResult:
    """What :meth:`InterviewSession.submit_answer` returns to the caller."""

    transcript: Transcript
    features: dict[str, float]
    fluency_result: FluencyPrediction
    feedback: Feedback
    follow_up_asked: bool
    body_language_result: BodyLanguageResult | None = None


def _interleave(a: list[QuestionItem], b: list[QuestionItem]) -> list[QuestionItem]:
    """Interleave two question lists so the two sources alternate.

    Args:
        a: First list of questions (e.g. question-bank questions).
        b: Second list of questions (e.g. resume questions).

    Returns:
        A single list alternating between ``a`` and ``b``, with any leftover
        tail from the longer list appended at the end.
    """
    merged: list[QuestionItem] = []
    for left, right in zip_longest(a, b):
        if left is not None:
            merged.append(left)
        if right is not None:
            merged.append(right)
    return merged


def _scores_by_area(answers: list[AnswerRecord]) -> dict[str, list[float]]:
    """Group each answer's overall score by its question's area.

    Args:
        answers: Answers submitted so far this session.

    Returns:
        A dict mapping area name to the list of overall scores recorded for
        questions in that area.
    """
    grouped: dict[str, list[float]] = {}
    for record in answers:
        grouped.setdefault(record.question_item.area, []).append(record.feedback.overall_score)
    return grouped


def _strongest_and_weakest(area_scores: dict[str, list[float]]) -> tuple[str, str]:
    """Pick the highest- and lowest-averaging areas.

    Args:
        area_scores: Areas mapped to their list of overall scores, as
            returned by :func:`_scores_by_area`.

    Returns:
        A ``(strongest_area, weakest_area)`` tuple, or ``("N/A", "N/A")`` if
        ``area_scores`` is empty.
    """
    if not area_scores:
        return "N/A", "N/A"
    averages = {area: mean(scores) for area, scores in area_scores.items()}
    strongest = max(averages, key=lambda area: averages[area])
    weakest = min(averages, key=lambda area: averages[area])
    return strongest, weakest


def _resume_context(profile: ResumeProfile) -> str:
    """Render a resume profile as short plain text for LLM grounding.

    Args:
        profile: The candidate's extracted resume profile.

    Returns:
        A human-readable listing of the profile's fields.
    """
    lines = [f"Name: {profile.name}"]
    if profile.skills:
        lines.append("Skills: " + ", ".join(profile.skills))
    if profile.projects:
        projects = "; ".join(f"{p.name} ({', '.join(p.tech)})" for p in profile.projects)
        lines.append(f"Projects: {projects}")
    if profile.education:
        lines.append("Education: " + "; ".join(profile.education))
    if profile.certifications:
        lines.append("Certifications: " + ", ".join(profile.certifications))
    return "\n".join(lines)


def _analyze_body_language(frames: Sequence[Any] | None) -> BodyLanguageResult | None:
    """Best-effort webcam body language analysis for one answer.

    This feature is optional and off by default (Task 11): no frames means
    the candidate didn't opt in, and any analysis failure (bad frame data, a
    missing model download, etc.) must never take down the interview itself.

    Args:
        frames: Webcam frames captured during the answer, or ``None``/empty
            if the webcam toggle wasn't used.

    Returns:
        The :class:`~src.body_language.BodyLanguageResult`, or ``None`` if no
        frames were given or analysis failed.
    """
    if not frames:
        return None
    try:
        return body_language_mod.analyze_video(frames)
    except (body_language_mod.BodyLanguageError, ValueError, OSError):
        return None


class InterviewSession:
    """A single stateful mock interview, from question plan to summary.

    All state lives on the instance (question queue, answers, follow-up
    counts) — nothing is stored in module-level globals, so multiple
    sessions can run concurrently and the class is easy to unit test.
    """

    MAX_FOLLOW_UPS_PER_QUESTION: int = 2

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Create a new, not-yet-started interview session.

        Args:
            llm: Optional chat model to use for every LLM call this session
                makes (feedback, follow-up decisions, the practice plan).
                Mainly for testing; defaults to a lazily-created
                :func:`src.llm.get_llm` client.
            conn: Optional SQLite connection (see :mod:`src.storage`). When
                given, each answer and the final summary are persisted as
                the session progresses.
        """
        self._llm = llm
        self._conn = conn

        self.session_id: str = ""
        self.role: str = ""
        self.difficulty: str | None = None
        self.resume_profile: ResumeProfile | None = None
        self.question_plan: list[QuestionItem] = []
        self.summary: SessionSummary | None = None

        self._queue: deque[QuestionItem] = deque()
        self._current: QuestionItem | None = None
        self._answers: list[AnswerRecord] = []
        self._follow_up_counts: dict[str, int] = {}
        self._asked_main_count: int = 0

    def _get_llm(self) -> BaseChatModel:
        """Return the session's chat model, creating it lazily if needed.

        Returns:
            The chat model to use for this session's LLM calls.

        Raises:
            config.ConfigError: If no ``llm`` was injected and
                ``GROQ_API_KEY`` is missing.
        """
        if self._llm is None:
            from src.llm import get_llm

            self._llm = get_llm()
        return self._llm

    # --- start ------------------------------------------------------------

    def start(
        self,
        role: str,
        difficulty: str | None = None,
        resume_profile: ResumeProfile | None = None,
        num_questions: int = 6,
    ) -> list[QuestionItem]:
        """Build a question plan and initialize a fresh session.

        Mixes question-bank questions (:func:`src.question_bank.get_questions`)
        with resume-grounded questions
        (:func:`src.resume.generate_resume_questions`) when a resume profile
        is given, split roughly evenly and interleaved.

        Args:
            role: Target job role, e.g. "Java Developer".
            difficulty: Optional difficulty filter ("easy", "medium", "hard").
            resume_profile: Optional extracted resume profile (Task 3). When
                given, roughly half the plan is personalized questions
                grounded in it.
            num_questions: Total number of main questions to plan (follow-ups
                are added on top of this during the interview).

        Returns:
            The planned list of :class:`QuestionItem`, in order.
        """
        self.role = role
        self.difficulty = difficulty
        self.resume_profile = resume_profile
        self.session_id = str(uuid4())
        started_at = datetime.now(timezone.utc)

        resume_count = min(num_questions // 2, num_questions) if resume_profile is not None else 0
        bank_count = num_questions - resume_count

        bank_items: list[QuestionItem] = []
        if bank_count > 0:
            for q in question_bank.get_questions(role, difficulty=difficulty, n=bank_count):
                bank_items.append(
                    QuestionItem(
                        id=str(uuid4()),
                        text=q["question"],
                        source="bank",
                        area=q["category"],
                        what_good_answer_covers=q["what_good_answer_covers"],
                    )
                )

        resume_items: list[QuestionItem] = []
        if resume_count > 0:
            questions = resume_mod.generate_resume_questions(
                resume_profile, n=resume_count, llm=self._get_llm()
            )
            for rq in questions:
                resume_items.append(
                    QuestionItem(
                        id=str(uuid4()),
                        text=rq.question,
                        source="resume",
                        area=rq.resume_section,
                    )
                )

        self.question_plan = _interleave(bank_items, resume_items)
        self._queue = deque(self.question_plan)
        self._current = None
        self._answers = []
        self._follow_up_counts = {}
        self._asked_main_count = 0
        self.summary = None

        if self._conn is not None:
            storage.create_session(
                self._conn,
                session_id=self.session_id,
                role=role,
                difficulty=difficulty,
                started_at=started_at.isoformat(),
            )

        return self.question_plan

    # --- next_question ------------------------------------------------------

    def next_question(self) -> NextQuestion | None:
        """Return the next question in the plan, with its TTS audio.

        Returns:
            The next :class:`NextQuestion`, or ``None`` once the plan (and
            any pending follow-ups) is exhausted.

        Raises:
            InterviewError: If the previously returned question hasn't been
                answered yet.
        """
        if self._current is not None:
            raise InterviewError(
                "The current question has not been answered yet; call "
                "submit_answer() or submit_text_answer() first."
            )
        if not self._queue:
            return None

        item = self._queue.popleft()
        self._current = item
        if not item.is_followup:
            self._asked_main_count += 1

        audio = tts.speak(item.text)
        return NextQuestion(
            question_id=item.id,
            text=item.text,
            audio=audio,
            is_followup=item.is_followup,
            question_number=self._asked_main_count,
            total_questions=len(self.question_plan),
        )

    # --- submit_answer / submit_text_answer ----------------------------------

    def submit_answer(
        self,
        audio: bytes | str | Path,
        body_language_frames: Sequence[Any] | None = None,
    ) -> AnswerResult:
        """Transcribe a spoken answer, score it, and generate feedback.

        Args:
            audio: Raw audio bytes or a path to an audio file, as accepted
                by :func:`src.speech.transcribe`.
            body_language_frames: Optional webcam frames captured during the
                answer (Task 11). When given, they're run through
                :func:`src.body_language.analyze_video`; omit this (or pass
                ``None``) when the webcam toggle is off — the interview works
                the same either way.

        Returns:
            The :class:`AnswerResult` for this answer.

        Raises:
            InterviewError: If no question is currently pending.
        """
        transcript = speech.transcribe(audio)
        return self._process_transcript(transcript, body_language_frames)

    def submit_text_answer(
        self,
        text: str,
        body_language_frames: Sequence[Any] | None = None,
    ) -> AnswerResult:
        """Score a typed answer, skipping real transcription.

        Used by the text-only CLI demo and by tests, where no audio is
        available. Duration is estimated from word count at
        ``_DEMO_ASSUMED_WPM``, so pace/pause features are approximate rather
        than measured.

        Args:
            text: The candidate's typed answer.
            body_language_frames: Optional webcam frames captured during the
                answer (Task 11); see :meth:`submit_answer`.

        Returns:
            The :class:`AnswerResult` for this answer.

        Raises:
            ValueError: If ``text`` is empty or whitespace-only.
            InterviewError: If no question is currently pending.
        """
        if not text.strip():
            raise ValueError("text must not be empty")

        duration = max(len(text.split()) / _DEMO_ASSUMED_WPM * 60, 1.0)
        transcript = Transcript(text=text, words=[], duration_seconds=duration, language="en")
        return self._process_transcript(transcript, body_language_frames)

    def _process_transcript(
        self,
        transcript: Transcript,
        body_language_frames: Sequence[Any] | None = None,
    ) -> AnswerResult:
        """Shared scoring/feedback/follow-up pipeline for one transcript.

        Args:
            transcript: The transcribed (or synthesized, for typed answers)
                answer to the current question.
            body_language_frames: Optional webcam frames captured during the
                answer (Task 11); see :meth:`submit_answer`.

        Returns:
            The :class:`AnswerResult` for this answer.

        Raises:
            InterviewError: If no question is currently pending.
        """
        if self._current is None:
            raise InterviewError("No question is pending; call next_question() first.")
        item = self._current

        feats = features_mod.extract_features(transcript)
        fluency_result = scorer.predict_fluency(feats)
        resume_context = _resume_context(self.resume_profile) if self.resume_profile else None
        fb = feedback_mod.generate_feedback(
            question=item.text,
            transcript=transcript.text,
            features=feats,
            fluency_result=fluency_result,
            what_good_answer_covers=item.what_good_answer_covers,
            resume_context=resume_context,
            llm=self._get_llm(),
        )
        body_language_result = _analyze_body_language(body_language_frames)

        answered_at = datetime.now(timezone.utc)
        self._answers.append(
            AnswerRecord(
                question_item=item,
                transcript=transcript,
                features=feats,
                fluency_result=fluency_result,
                feedback=fb,
                answered_at=answered_at,
                body_language_result=body_language_result,
            )
        )

        if self._conn is not None:
            storage.save_answer(
                self._conn,
                session_id=self.session_id,
                question=item.text,
                is_followup=item.is_followup,
                area=item.area,
                transcript=transcript.text,
                scores={
                    "content_score": fb.content_score.score,
                    "fluency_score": fluency_result.score,
                    "overall_score": fb.overall_score,
                },
                features=feats,
                feedback=fb.model_dump(),
                answered_at=answered_at.isoformat(),
                body_language=asdict(body_language_result) if body_language_result else None,
            )

        follow_up_asked = self._maybe_queue_follow_up(item, transcript.text, fb)
        self._current = None

        return AnswerResult(
            transcript=transcript,
            features=feats,
            fluency_result=fluency_result,
            feedback=fb,
            follow_up_asked=follow_up_asked,
            body_language_result=body_language_result,
        )

    def _maybe_queue_follow_up(
        self, item: QuestionItem, transcript_text: str, fb: Feedback
    ) -> bool:
        """Ask the LLM whether a follow-up is warranted and queue it if so.

        Args:
            item: The question that was just answered.
            transcript_text: The candidate's answer text.
            fb: The feedback generated for this answer.

        Returns:
            ``True`` if a follow-up question was queued, ``False``
            otherwise (including when the per-question follow-up limit has
            already been reached).
        """
        parent_id = item.parent_id or item.id
        count = self._follow_up_counts.get(parent_id, 0)
        if count >= self.MAX_FOLLOW_UPS_PER_QUESTION:
            return False

        decision = self._decide_follow_up(item, transcript_text, fb)
        if not decision.should_follow_up or not decision.follow_up_question.strip():
            return False

        follow_up = QuestionItem(
            id=str(uuid4()),
            text=decision.follow_up_question.strip(),
            source=item.source,
            area=item.area,
            is_followup=True,
            parent_id=parent_id,
        )
        self._queue.appendleft(follow_up)
        self._follow_up_counts[parent_id] = count + 1
        return True

    def _decide_follow_up(
        self, item: QuestionItem, transcript_text: str, fb: Feedback
    ) -> FollowUpDecision:
        """Call the LLM to decide whether a follow-up is worth asking.

        Args:
            item: The question that was just answered.
            transcript_text: The candidate's answer text.
            fb: The feedback generated for this answer.

        Returns:
            The LLM's :class:`FollowUpDecision`. Falls back to "no follow-up"
            if the LLM call fails, since a missed follow-up should never
            abort the interview.
        """
        model = self._get_llm().with_structured_output(FollowUpDecision)
        user_prompt = prompts.build_followup_user_prompt(
            question=item.text,
            transcript=transcript_text,
            key_points_missed=fb.key_points_missed,
            content_score=fb.content_score.score,
        )
        try:
            result = model.invoke(
                [
                    SystemMessage(content=prompts.FOLLOWUP_SYSTEM_PROMPT_V1),
                    HumanMessage(content=user_prompt),
                ]
            )
        except (ValidationError, OutputParserException, ValueError):
            return FollowUpDecision(should_follow_up=False)

        return (
            result
            if isinstance(result, FollowUpDecision)
            else FollowUpDecision.model_validate(result)
        )

    # --- end ----------------------------------------------------------------

    def end(self) -> SessionSummary:
        """End the session and build a summary with a personalized practice plan.

        Returns:
            The session's :class:`SessionSummary`.

        Raises:
            InterviewError: If no answers were submitted, or if the practice
                plan LLM call fails.
        """
        if not self._answers:
            raise InterviewError("Cannot end a session with no submitted answers.")

        overall_scores = [a.feedback.overall_score for a in self._answers]
        fluency_scores = [a.fluency_result.score for a in self._answers]
        content_scores = [a.feedback.content_score.score for a in self._answers]

        strongest_area, weakest_area = _strongest_and_weakest(_scores_by_area(self._answers))
        tips = [a.feedback.one_tip for a in self._answers]
        practice_plan = self._generate_practice_plan(strongest_area, weakest_area, tips)

        summary = SessionSummary(
            questions_answered=len(self._answers),
            average_overall_score=round(mean(overall_scores), 1),
            average_fluency_score=round(mean(fluency_scores), 1),
            average_content_score=round(mean(content_scores), 1),
            strongest_area=strongest_area,
            weakest_area=weakest_area,
            practice_plan=practice_plan,
        )
        self.summary = summary

        if self._conn is not None:
            storage.end_session(
                self._conn,
                session_id=self.session_id,
                ended_at=datetime.now(timezone.utc).isoformat(),
                summary=summary.model_dump(),
            )

        return summary

    def _generate_practice_plan(
        self, strongest_area: str, weakest_area: str, tips: list[str]
    ) -> list[PracticeDay]:
        """Ask the LLM for a personalized 7-day practice plan.

        Args:
            strongest_area: The candidate's strongest-scoring area.
            weakest_area: The candidate's weakest-scoring area.
            tips: The ``one_tip`` from every answer's feedback this session.

        Returns:
            A list of 7 :class:`PracticeDay` entries.

        Raises:
            InterviewError: If the LLM fails to produce valid structured
                output.
        """
        model = self._get_llm().with_structured_output(_PracticePlanLLMOutput)
        user_prompt = prompts.build_practice_plan_user_prompt(
            role=self.role,
            strongest_area=strongest_area,
            weakest_area=weakest_area,
            tips=tips,
        )
        try:
            result = model.invoke(
                [
                    SystemMessage(content=prompts.PRACTICE_PLAN_SYSTEM_PROMPT_V1),
                    HumanMessage(content=user_prompt),
                ]
            )
        except (ValidationError, OutputParserException, ValueError) as exc:
            raise InterviewError("Failed to generate a practice plan.") from exc

        output = (
            result
            if isinstance(result, _PracticePlanLLMOutput)
            else _PracticePlanLLMOutput.model_validate(result)
        )
        return output.plan


# --- CLI demo -------------------------------------------------------------------


def _run_demo() -> None:
    """Run a text-only mock interview in the terminal (no mic/audio needed).

    Requires a configured ``GROQ_API_KEY`` since feedback, follow-up
    decisions, and the practice plan are all real LLM calls.
    """
    print("SpeakReady - text-only demo interview\n")
    role = input("Target role (e.g. 'Python Developer'): ").strip() or "Software Engineer"
    difficulty = input("Difficulty [easy/medium/hard] (blank = any): ").strip() or None
    num_raw = input("Number of questions [default 4]: ").strip()
    num_questions = int(num_raw) if num_raw else 4

    session = InterviewSession()
    session.start(role=role, difficulty=difficulty, num_questions=num_questions)

    while (question := session.next_question()) is not None:
        label = (
            "Follow-up"
            if question.is_followup
            else f"Question {question.question_number}/{question.total_questions}"
        )
        print(f"\n[{label}] {question.text}")
        answer = input("Your answer: ").strip() or "I don't have an answer for this one."

        result = session.submit_text_answer(answer)
        print(f"  -> Overall score: {result.feedback.overall_score}/100")
        print(f"  -> Tip: {result.feedback.one_tip}")

    summary = session.end()
    print("\n=== Session summary ===")
    print(f"Questions answered: {summary.questions_answered}")
    print(f"Average overall score: {summary.average_overall_score}/100")
    print(f"Strongest area: {summary.strongest_area}")
    print(f"Weakest area: {summary.weakest_area}")
    print("7-day practice plan:")
    for day in summary.practice_plan:
        print(f"  Day {day.day} - {day.focus}: {day.activity}")


def _main() -> None:
    """CLI entry point: ``python -m src.interview_agent --demo``."""
    parser = argparse.ArgumentParser(description="SpeakReady interview agent.")
    parser.add_argument("--demo", action="store_true", help="Run a text-only mock interview.")
    args = parser.parse_args()

    if args.demo:
        _run_demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    _main()
