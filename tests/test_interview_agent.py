"""Tests for the interview agent.

LLM and speech calls are mocked throughout so these tests run without a
GROQ_API_KEY. The fluency model is also mocked so tests don't depend on
``models/fluency_model.joblib`` existing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.feedback import ContentScore, Feedback, GrammarCorrection
from src.interview_agent import (
    FollowUpDecision,
    InterviewError,
    InterviewSession,
    PracticeDay,
    QuestionItem,
    _interleave,
    _PracticePlanLLMOutput,
    _strongest_and_weakest,
)
from src.resume import Project, ResumeProfile, ResumeQuestion, _ResumeQuestionSet
from src.scorer import FluencyPrediction

FIXTURE_BANK_QUESTIONS = [
    {
        "id": "q001",
        "question": "Tell me about yourself.",
        "category": "HR",
        "difficulty": "easy",
        "what_good_answer_covers": ["Background", "Career goal"],
    },
    {
        "id": "q010",
        "question": "What is a Python decorator?",
        "category": "Python",
        "difficulty": "easy",
        "what_good_answer_covers": ["Definition", "Example"],
    },
]

FIXTURE_RESUME_PROFILE = ResumeProfile(
    name="Aditi Sharma",
    skills=["Python", "SQL"],
    projects=[
        Project(
            name="CustomerIQ",
            tech=["Python", "XGBoost"],
            description="Churn prediction system.",
        )
    ],
    education=["B.Tech in Computer Science"],
    certifications=[],
)


def _feedback(overall_score: float = 75.0, one_tip: str = "Speak with more detail.") -> Feedback:
    """A well-formed Feedback object for tests."""
    return Feedback(
        content_score=ContentScore(score=7.0, reason="Solid answer."),
        key_points_covered=["Background"],
        key_points_missed=["Specific metrics"],
        grammar_corrections=[
            GrammarCorrection(original="I has", corrected="I have", explanation="Verb agreement.")
        ],
        filler_summary="No filler words detected.",
        improved_answer="In my previous role, I...",
        one_tip=one_tip,
        overall_score=overall_score,
    )


class _FakeLLM:
    """A fake chat model dispatching with_structured_output() by schema class.

    Real LLM mocks in this codebase stub a single with_structured_output()
    return value, but InterviewSession asks for several different schemas
    (feedback, follow-up decisions, the practice plan) from the same model,
    so this fake routes each schema to its own configurable mock.
    """

    def __init__(self) -> None:
        self.mocks: dict[type, MagicMock] = {}
        self.calls: list[type] = []

    def stub(self, schema: type, return_value=None, side_effect=None) -> MagicMock:
        mock_structured = MagicMock()
        if side_effect is not None:
            mock_structured.invoke.side_effect = side_effect
        else:
            mock_structured.invoke.return_value = return_value
        self.mocks[schema] = mock_structured
        return mock_structured

    def with_structured_output(self, schema: type):
        self.calls.append(schema)
        if schema not in self.mocks:
            self.stub(schema, return_value=MagicMock())
        return self.mocks[schema]


def _mock_fluency_model(label: str = "Fluent", proba: list[float] | None = None) -> MagicMock:
    """A mock fluency classifier for src.scorer.predict_fluency."""
    labels = ["Beginner", "Intermediate", "Fluent"]
    model = MagicMock()
    model.classes_ = labels
    model.predict_proba.return_value = np.array([proba or [0.1, 0.2, 0.7]])
    return model


def _started_session(
    llm: _FakeLLM,
    monkeypatch,
    resume_profile: ResumeProfile | None = None,
    num_questions: int = 2,
) -> InterviewSession:
    """An InterviewSession past start(), with question_bank/scorer mocked."""
    monkeypatch.setattr(
        "src.interview_agent.question_bank.get_questions",
        lambda role, difficulty=None, n=5: FIXTURE_BANK_QUESTIONS[:n],
    )
    monkeypatch.setattr(
        "src.interview_agent.scorer.predict_fluency",
        lambda features: FluencyPrediction(label="Fluent", confidence=0.8, score=80.0),
    )
    monkeypatch.setattr("src.interview_agent.tts.speak", lambda text, language=None: b"audio-bytes")

    session = InterviewSession(llm=llm)
    session.start(
        role="Python Developer",
        num_questions=num_questions,
        resume_profile=resume_profile,
    )
    return session


# --- _interleave / _scores_by_area / _strongest_and_weakest -------------------


def test_interleave_alternates_sources() -> None:
    """Bank and resume questions should alternate in the merged plan."""
    bank = [QuestionItem(id="b1", text="b1", source="bank", area="HR")]
    resume = [
        QuestionItem(id="r1", text="r1", source="resume", area="Skills"),
        QuestionItem(id="r2", text="r2", source="resume", area="Skills"),
    ]

    merged = _interleave(bank, resume)

    assert [q.id for q in merged] == ["b1", "r1", "r2"]


def test_strongest_and_weakest_picks_highest_and_lowest_average() -> None:
    """The area with the highest/lowest mean overall_score should be identified."""
    area_scores = {"HR": [90.0, 80.0], "Python": [40.0, 50.0]}

    strongest, weakest = _strongest_and_weakest(area_scores)

    assert strongest == "HR"
    assert weakest == "Python"


def test_strongest_and_weakest_empty_returns_na() -> None:
    """No answers yet should not raise, just report N/A."""
    assert _strongest_and_weakest({}) == ("N/A", "N/A")


# --- start ---------------------------------------------------------------------


def test_start_builds_plan_from_bank_only(monkeypatch) -> None:
    """With no resume profile, the whole plan should come from the question bank."""
    llm = _FakeLLM()
    session = _started_session(llm, monkeypatch, resume_profile=None, num_questions=2)

    assert len(session.question_plan) == 2
    assert all(q.source == "bank" for q in session.question_plan)
    assert session.session_id


def test_start_uses_llm_generation_for_custom_role(monkeypatch) -> None:
    """use_custom_role=True should generate questions instead of using the bank."""
    llm = _FakeLLM()
    generated = [
        {
            "id": "custom-1",
            "question": "How do you prioritize a backlog with competing deadlines?",
            "category": "Product Management",
            "difficulty": "medium",
            "what_good_answer_covers": ["A prioritization framework"],
        },
        {
            "id": "custom-2",
            "question": "Tell me about a stakeholder conflict you resolved.",
            "category": "Behavioral",
            "difficulty": "medium",
            "what_good_answer_covers": ["Situation, action, outcome"],
        },
    ]
    calls: list[tuple[str, str | None, int]] = []

    def fake_generate(role, difficulty=None, n=5, llm=None):
        calls.append((role, difficulty, n))
        return generated[:n]

    monkeypatch.setattr(
        "src.interview_agent.question_bank.generate_custom_role_questions", fake_generate
    )
    monkeypatch.setattr(
        "src.interview_agent.question_bank.get_questions",
        lambda *a, **k: pytest.fail("get_questions() should not be called for a custom role"),
    )
    monkeypatch.setattr(
        "src.interview_agent.scorer.predict_fluency",
        lambda features: FluencyPrediction(label="Fluent", confidence=0.8, score=80.0),
    )
    monkeypatch.setattr("src.interview_agent.tts.speak", lambda text, language=None: b"audio-bytes")

    session = InterviewSession(llm=llm)
    session.start(role="Business Analyst", num_questions=2, use_custom_role=True)

    assert calls == [("Business Analyst", None, 2)]
    assert len(session.question_plan) == 2
    assert all(q.source == "custom_role" for q in session.question_plan)
    assert {q.area for q in session.question_plan} == {"Product Management", "Behavioral"}


def test_start_mixes_in_resume_questions(monkeypatch) -> None:
    """With a resume profile, roughly half the plan should be resume questions."""
    llm = _FakeLLM()
    llm.stub(
        _ResumeQuestionSet,
        return_value=_ResumeQuestionSet(
            questions=[
                ResumeQuestion(
                    question="Tell me about CustomerIQ.",
                    resume_section="Projects: CustomerIQ",
                )
            ]
        ),
    )

    session = _started_session(
        llm, monkeypatch, resume_profile=FIXTURE_RESUME_PROFILE, num_questions=2
    )

    sources = {q.source for q in session.question_plan}
    assert sources == {"bank", "resume"}


# --- next_question ---------------------------------------------------------------


def test_next_question_returns_text_and_audio(monkeypatch) -> None:
    """next_question() should surface the question text and TTS audio."""
    llm = _FakeLLM()
    session = _started_session(llm, monkeypatch, num_questions=2)

    question = session.next_question()

    assert question is not None
    assert question.text == FIXTURE_BANK_QUESTIONS[0]["question"]
    assert question.audio == b"audio-bytes"
    assert question.question_number == 1
    assert question.total_questions == 2


def test_next_question_raises_if_current_unanswered(monkeypatch) -> None:
    """Calling next_question() twice without answering should raise."""
    llm = _FakeLLM()
    session = _started_session(llm, monkeypatch, num_questions=2)
    session.next_question()

    with pytest.raises(InterviewError):
        session.next_question()


def test_next_question_returns_none_when_exhausted(monkeypatch) -> None:
    """Once every question (and follow-up) is answered, it should return None."""
    llm = _FakeLLM()
    llm.stub(FollowUpDecision, return_value=FollowUpDecision(should_follow_up=False))
    session = _started_session(llm, monkeypatch, num_questions=1)

    monkeypatch.setattr(
        "src.interview_agent.feedback_mod.generate_feedback", lambda **kwargs: _feedback()
    )

    session.next_question()
    session.submit_text_answer("My answer.")

    assert session.next_question() is None


# --- submit_answer / submit_text_answer ------------------------------------------


def test_submit_text_answer_raises_without_pending_question(monkeypatch) -> None:
    """Submitting before calling next_question() should raise."""
    llm = _FakeLLM()
    session = _started_session(llm, monkeypatch, num_questions=1)

    with pytest.raises(InterviewError):
        session.submit_text_answer("An answer.")


def test_submit_text_answer_rejects_empty_text(monkeypatch) -> None:
    """An empty/whitespace-only answer should raise ValueError."""
    llm = _FakeLLM()
    session = _started_session(llm, monkeypatch, num_questions=1)
    session.next_question()

    with pytest.raises(ValueError):
        session.submit_text_answer("   ")


def test_submit_text_answer_scores_and_returns_feedback(monkeypatch) -> None:
    """A typed answer should be scored and feedback returned, without follow-up."""
    llm = _FakeLLM()
    expected_feedback = _feedback(overall_score=82.0)
    monkeypatch.setattr(
        "src.interview_agent.feedback_mod.generate_feedback",
        lambda **kwargs: expected_feedback,
    )
    llm.stub(FollowUpDecision, return_value=FollowUpDecision(should_follow_up=False))

    session = _started_session(llm, monkeypatch, num_questions=1)
    session.next_question()
    result = session.submit_text_answer("I have three years of experience in backend systems.")

    assert result.feedback.overall_score == 82.0
    assert result.follow_up_asked is False
    assert session._current is None


def test_submit_answer_uses_transcribe(monkeypatch) -> None:
    """submit_answer(audio) should route through src.speech.transcribe."""
    from src.speech import Transcript

    llm = _FakeLLM()
    monkeypatch.setattr(
        "src.interview_agent.speech.transcribe",
        lambda audio, client=None: Transcript(
            text="A transcribed spoken answer.", words=[], duration_seconds=5.0, language="en"
        ),
    )
    monkeypatch.setattr(
        "src.interview_agent.feedback_mod.generate_feedback",
        lambda **kwargs: _feedback(),
    )
    llm.stub(FollowUpDecision, return_value=FollowUpDecision(should_follow_up=False))

    session = _started_session(llm, monkeypatch, num_questions=1)
    session.next_question()
    result = session.submit_answer(b"fake-audio-bytes")

    assert result.transcript.text == "A transcribed spoken answer."


def test_submit_transcript_does_not_call_transcribe(monkeypatch) -> None:
    """submit_transcript() (Task 12's quick-signal path) must not re-transcribe."""
    from unittest.mock import MagicMock

    from src.speech import Transcript

    llm = _FakeLLM()
    transcribe_mock = MagicMock()
    monkeypatch.setattr("src.interview_agent.speech.transcribe", transcribe_mock)
    monkeypatch.setattr(
        "src.interview_agent.feedback_mod.generate_feedback",
        lambda **kwargs: _feedback(),
    )
    llm.stub(FollowUpDecision, return_value=FollowUpDecision(should_follow_up=False))

    session = _started_session(llm, monkeypatch, num_questions=1)
    session.next_question()
    transcript = Transcript(text="Already transcribed.", words=[], duration_seconds=3.0)
    result = session.submit_transcript(transcript)

    transcribe_mock.assert_not_called()
    assert result.transcript.text == "Already transcribed."
    assert result.feedback.overall_score == 75.0


# --- body language (Task 11) --------------------------------------------------


def test_submit_text_answer_without_frames_has_no_body_language_result(monkeypatch) -> None:
    """Not passing body_language_frames (the default) should leave it None."""
    llm = _FakeLLM()
    monkeypatch.setattr(
        "src.interview_agent.feedback_mod.generate_feedback", lambda **kwargs: _feedback()
    )
    llm.stub(FollowUpDecision, return_value=FollowUpDecision(should_follow_up=False))

    session = _started_session(llm, monkeypatch, num_questions=1)
    session.next_question()
    result = session.submit_text_answer("An answer.")

    assert result.body_language_result is None


def test_submit_text_answer_with_frames_attaches_body_language_result(monkeypatch) -> None:
    """Frames should be run through body_language.analyze_video and attached."""
    from src.body_language import BodyLanguageResult

    expected = BodyLanguageResult(
        eye_contact_ratio=0.9, posture_score=85.0, looking_away_count=0, summary="Great job."
    )
    llm = _FakeLLM()
    monkeypatch.setattr(
        "src.interview_agent.feedback_mod.generate_feedback", lambda **kwargs: _feedback()
    )
    monkeypatch.setattr(
        "src.interview_agent.body_language_mod.analyze_video", lambda frames: expected
    )
    llm.stub(FollowUpDecision, return_value=FollowUpDecision(should_follow_up=False))

    session = _started_session(llm, monkeypatch, num_questions=1)
    session.next_question()
    result = session.submit_text_answer("An answer.", body_language_frames=["frame"])

    assert result.body_language_result == expected
    assert session._answers[0].body_language_result == expected


def test_body_language_failure_does_not_break_the_interview(monkeypatch) -> None:
    """A webcam-analysis failure must never take down the rest of the interview."""
    from src.body_language import BodyLanguageError

    llm = _FakeLLM()
    monkeypatch.setattr(
        "src.interview_agent.feedback_mod.generate_feedback", lambda **kwargs: _feedback()
    )

    def _boom(frames):
        raise BodyLanguageError("no model available")

    monkeypatch.setattr("src.interview_agent.body_language_mod.analyze_video", _boom)
    llm.stub(FollowUpDecision, return_value=FollowUpDecision(should_follow_up=False))

    session = _started_session(llm, monkeypatch, num_questions=1)
    session.next_question()
    result = session.submit_text_answer("An answer.", body_language_frames=["frame"])

    assert result.body_language_result is None
    assert result.feedback.overall_score == 75.0


# --- voice confidence / pronunciation (Task 13) --------------------------------


def test_submit_text_answer_has_no_voice_confidence_or_pronunciation(monkeypatch) -> None:
    """A typed answer has no real audio/ASR words, so both signals stay None."""
    llm = _FakeLLM()
    monkeypatch.setattr(
        "src.interview_agent.feedback_mod.generate_feedback", lambda **kwargs: _feedback()
    )
    llm.stub(FollowUpDecision, return_value=FollowUpDecision(should_follow_up=False))

    session = _started_session(llm, monkeypatch, num_questions=1)
    session.next_question()
    result = session.submit_text_answer("An answer.")

    assert result.voice_confidence_result is None
    assert result.pronunciation_result is None


def test_submit_answer_attaches_voice_confidence_result(monkeypatch) -> None:
    """submit_answer() should run the raw audio through analyze_voice_confidence."""
    from src.confidence import VoiceConfidenceResult
    from src.speech import Transcript

    expected = VoiceConfidenceResult(
        pitch_variability=80.0,
        volume_steadiness=90.0,
        speaking_energy=70.0,
        confidence_score=80.0,
        summary="Confident and clear.",
    )
    llm = _FakeLLM()
    monkeypatch.setattr(
        "src.interview_agent.speech.transcribe",
        lambda audio, client=None: Transcript(
            text="A transcribed spoken answer.", words=[], duration_seconds=5.0, language="en"
        ),
    )
    monkeypatch.setattr(
        "src.interview_agent.feedback_mod.generate_feedback", lambda **kwargs: _feedback()
    )
    monkeypatch.setattr(
        "src.interview_agent.confidence_mod.analyze_voice_confidence", lambda audio: expected
    )
    llm.stub(FollowUpDecision, return_value=FollowUpDecision(should_follow_up=False))

    session = _started_session(llm, monkeypatch, num_questions=1)
    session.next_question()
    result = session.submit_answer(b"fake-audio-bytes")

    assert result.voice_confidence_result == expected
    assert session._answers[0].voice_confidence_result == expected


def test_voice_confidence_failure_does_not_break_the_interview(monkeypatch) -> None:
    """A confidence-analysis failure must never take down the rest of the interview."""
    from src.confidence import VoiceConfidenceError
    from src.speech import Transcript

    llm = _FakeLLM()
    monkeypatch.setattr(
        "src.interview_agent.speech.transcribe",
        lambda audio, client=None: Transcript(
            text="A transcribed spoken answer.", words=[], duration_seconds=5.0, language="en"
        ),
    )
    monkeypatch.setattr(
        "src.interview_agent.feedback_mod.generate_feedback", lambda **kwargs: _feedback()
    )

    def _boom(audio):
        raise VoiceConfidenceError("no voice detected")

    monkeypatch.setattr("src.interview_agent.confidence_mod.analyze_voice_confidence", _boom)
    llm.stub(FollowUpDecision, return_value=FollowUpDecision(should_follow_up=False))

    session = _started_session(llm, monkeypatch, num_questions=1)
    session.next_question()
    result = session.submit_answer(b"fake-audio-bytes")

    assert result.voice_confidence_result is None
    assert result.feedback.overall_score == 75.0


def test_submit_transcript_with_words_attaches_pronunciation_result(monkeypatch) -> None:
    """A transcript with real words should get a pronunciation proxy signal."""
    from src.speech import Transcript, Word

    llm = _FakeLLM()
    monkeypatch.setattr(
        "src.interview_agent.feedback_mod.generate_feedback", lambda **kwargs: _feedback()
    )
    llm.stub(FollowUpDecision, return_value=FollowUpDecision(should_follow_up=False))

    session = _started_session(llm, monkeypatch, num_questions=1)
    session.next_question()
    transcript = Transcript(
        text="Hello there.",
        words=[Word(word="Hello", start=0.0, end=0.5), Word(word="there.", start=0.5, end=1.0)],
        duration_seconds=1.0,
    )
    result = session.submit_transcript(transcript)

    assert result.pronunciation_result is not None
    assert result.pronunciation_result.words_to_double_check == []


# --- follow-up logic ---------------------------------------------------------


def test_follow_up_is_queued_and_asked_next(monkeypatch) -> None:
    """A should_follow_up=True decision should insert a follow-up right after."""
    llm = _FakeLLM()
    monkeypatch.setattr(
        "src.interview_agent.feedback_mod.generate_feedback",
        lambda **kwargs: _feedback(),
    )
    llm.stub(
        FollowUpDecision,
        return_value=FollowUpDecision(
            should_follow_up=True,
            follow_up_question="You mentioned backend systems - which ones specifically?",
            reason="The answer was vague about specific technologies.",
        ),
    )

    session = _started_session(llm, monkeypatch, num_questions=1)
    session.next_question()
    result = session.submit_text_answer("I worked on backend systems.")

    assert result.follow_up_asked is True

    follow_up = session.next_question()
    assert follow_up is not None
    assert follow_up.is_followup is True
    assert "backend systems" in follow_up.text


def test_follow_up_respects_max_limit(monkeypatch) -> None:
    """No more than MAX_FOLLOW_UPS_PER_QUESTION follow-ups per main question."""
    llm = _FakeLLM()
    monkeypatch.setattr(
        "src.interview_agent.feedback_mod.generate_feedback",
        lambda **kwargs: _feedback(),
    )
    llm.stub(
        FollowUpDecision,
        return_value=FollowUpDecision(should_follow_up=True, follow_up_question="Can you clarify?"),
    )

    session = _started_session(llm, monkeypatch, num_questions=1)
    session.MAX_FOLLOW_UPS_PER_QUESTION = 2

    follow_ups_asked = 0
    while session.next_question() is not None:
        result = session.submit_text_answer("An answer that stays vague.")
        if result.follow_up_asked:
            follow_ups_asked += 1
        if follow_ups_asked > 2:
            pytest.fail("More than MAX_FOLLOW_UPS_PER_QUESTION follow-ups were asked.")

    assert follow_ups_asked == 2


def test_follow_up_llm_failure_does_not_raise(monkeypatch) -> None:
    """If the follow-up LLM call fails, the interview should continue quietly."""
    llm = _FakeLLM()
    monkeypatch.setattr(
        "src.interview_agent.feedback_mod.generate_feedback",
        lambda **kwargs: _feedback(),
    )
    llm.stub(FollowUpDecision, side_effect=ValueError("bad json"))

    session = _started_session(llm, monkeypatch, num_questions=1)
    session.next_question()
    result = session.submit_text_answer("An answer.")

    assert result.follow_up_asked is False
    assert session.next_question() is None


# --- end -----------------------------------------------------------------------


def test_end_raises_without_any_answers(monkeypatch) -> None:
    """Ending a session with zero submitted answers should raise."""
    llm = _FakeLLM()
    session = _started_session(llm, monkeypatch, num_questions=1)

    with pytest.raises(InterviewError):
        session.end()


def test_end_builds_summary_with_practice_plan(monkeypatch) -> None:
    """end() should average scores, pick strongest/weakest area, and get a plan."""
    llm = _FakeLLM()
    monkeypatch.setattr(
        "src.interview_agent.feedback_mod.generate_feedback",
        lambda **kwargs: _feedback(overall_score=70.0),
    )
    llm.stub(FollowUpDecision, return_value=FollowUpDecision(should_follow_up=False))
    llm.stub(
        _PracticePlanLLMOutput,
        return_value=_PracticePlanLLMOutput(
            plan=[
                PracticeDay(day=i, focus=f"Focus {i}", activity=f"Activity {i}")
                for i in range(1, 8)
            ]
        ),
    )

    session = _started_session(llm, monkeypatch, num_questions=2)
    session.next_question()
    session.submit_text_answer("First answer.")
    session.next_question()
    session.submit_text_answer("Second answer.")

    summary = session.end()

    assert summary.questions_answered == 2
    assert summary.average_overall_score == pytest.approx(70.0)
    assert len(summary.practice_plan) == 7
    assert summary.practice_plan[0].day == 1
    assert session.summary == summary


def test_end_practice_plan_failure_raises_interview_error(monkeypatch) -> None:
    """A broken practice-plan LLM call should surface as InterviewError."""
    llm = _FakeLLM()
    monkeypatch.setattr(
        "src.interview_agent.feedback_mod.generate_feedback",
        lambda **kwargs: _feedback(),
    )
    llm.stub(FollowUpDecision, return_value=FollowUpDecision(should_follow_up=False))
    llm.stub(_PracticePlanLLMOutput, side_effect=ValueError("bad json"))

    session = _started_session(llm, monkeypatch, num_questions=1)
    session.next_question()
    session.submit_text_answer("An answer.")

    with pytest.raises(InterviewError):
        session.end()


# --- storage integration ----------------------------------------------------


def test_session_persists_to_storage_when_conn_given(monkeypatch) -> None:
    """When a sqlite connection is given, sessions/answers/summary should persist."""
    from src import storage

    llm = _FakeLLM()
    monkeypatch.setattr(
        "src.interview_agent.feedback_mod.generate_feedback",
        lambda **kwargs: _feedback(overall_score=88.0),
    )
    llm.stub(FollowUpDecision, return_value=FollowUpDecision(should_follow_up=False))
    llm.stub(
        _PracticePlanLLMOutput,
        return_value=_PracticePlanLLMOutput(
            plan=[
                PracticeDay(day=i, focus=f"Focus {i}", activity=f"Activity {i}")
                for i in range(1, 8)
            ]
        ),
    )

    conn = storage.get_connection(":memory:")
    monkeypatch.setattr(
        "src.interview_agent.question_bank.get_questions",
        lambda role, difficulty=None, n=5: FIXTURE_BANK_QUESTIONS[:n],
    )
    monkeypatch.setattr(
        "src.interview_agent.scorer.predict_fluency",
        lambda features: FluencyPrediction(label="Fluent", confidence=0.8, score=80.0),
    )
    monkeypatch.setattr("src.interview_agent.tts.speak", lambda text, language=None: b"audio")

    session = InterviewSession(llm=llm, conn=conn)
    session.start(role="Python Developer", num_questions=1)
    session.next_question()
    session.submit_text_answer("An answer with enough detail.")
    session.end()

    stored_session = storage.get_session(conn, session.session_id)
    stored_answers = storage.list_answers(conn, session.session_id)

    assert stored_session is not None
    assert stored_session["ended_at"] is not None
    assert len(stored_answers) == 1
    assert stored_answers[0]["question"] == FIXTURE_BANK_QUESTIONS[0]["question"]
