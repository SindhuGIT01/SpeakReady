"""Interview page: plays each question, records an answer, and shows feedback."""

from __future__ import annotations

import streamlit as st

from src import config
from src.feedback import FeedbackError
from src.interview_agent import InterviewError
from src.scorer import ScorerError
from src.speech import AudioValidationError, TranscriptionError

st.title("🎙️ Interview")

session = st.session_state.get("interview_session")
if session is None:
    st.info("No interview in progress yet.")
    if st.button("Go to Setup", type="primary"):
        st.switch_page("app_pages/setup.py")
    st.stop()

stage = st.session_state.get("interview_stage", "idle")

# Pull the next question (or end the session) whenever we're ready for one.
if stage == "asking" and st.session_state.get("current_question") is None:
    try:
        next_question = session.next_question()
    except InterviewError:
        next_question = None

    if next_question is None:
        try:
            with st.spinner("Wrapping up your session..."):
                summary = session.end()
            st.session_state["session_summary"] = summary
            st.session_state["interview_stage"] = "finished"
        except InterviewError:
            st.error("Couldn't finish the session — please try answering one more question.")
    else:
        st.session_state["current_question"] = next_question
    st.rerun()

stage = st.session_state.get("interview_stage", "idle")

if stage == "finished":
    st.success("Interview complete! 🎉")
    st.write("Head over to your report for the full breakdown.")
    if st.button("View Report", type="primary"):
        st.switch_page("app_pages/report.py")
    st.stop()

if stage == "asking":
    question = st.session_state["current_question"]

    progress = min(question.question_number / max(question.total_questions, 1), 1.0)
    st.progress(progress)
    st.caption(
        "Follow-up question"
        if question.is_followup
        else f"Question {question.question_number} of {question.total_questions}"
    )
    st.markdown(f"### {question.text}")
    st.audio(question.audio, format="audio/mp3", autoplay=True)

    st.write("")
    audio_value = st.audio_input("Record your answer")

    if audio_value is not None and st.button("Submit Answer", type="primary"):
        try:
            with st.spinner("Analyzing your answer..."):
                result = session.submit_answer(audio_value.getvalue())
            st.session_state["last_answer_result"] = result
            st.session_state["interview_stage"] = "reviewing"
            st.rerun()
        except AudioValidationError as exc:
            st.error(str(exc))
        except TranscriptionError as exc:
            st.error(str(exc))
        except (FeedbackError, ScorerError):
            st.error("We had trouble scoring that answer. Please try again.")
        except config.ConfigError:
            st.error("The AI service isn't configured. Please check the app's setup.")
        except Exception:
            st.error("Something went wrong while analyzing your answer. Please try again.")

elif stage == "reviewing":
    result = st.session_state["last_answer_result"]
    fb = result.feedback
    features = result.features
    fluency = result.fluency_result

    st.success("Here's your feedback")

    score_col, fluency_col, wpm_col, filler_col = st.columns(4)
    score_col.metric("Overall score", f"{fb.overall_score:.0f}/100")
    fluency_col.metric("Fluency", fluency.label, f"{fluency.confidence * 100:.0f}% confidence")
    wpm_col.metric("Pace", f"{features['words_per_minute']:.0f} WPM")
    filler_col.metric("Filler words", f"{features['filler_count']:.0f}")

    st.caption(
        f"Pauses: {features['pause_count']:.0f} total "
        f"({features['long_pause_count']:.0f} long)"
    )
    st.info(fb.filler_summary)

    with st.container(border=True):
        st.markdown(f"**Content score:** {fb.content_score.score:.1f}/10")
        st.caption(fb.content_score.reason)
        if fb.key_points_covered:
            st.markdown("✅ **Covered:** " + "; ".join(fb.key_points_covered))
        if fb.key_points_missed:
            st.markdown("❌ **Missed:** " + "; ".join(fb.key_points_missed))

    if fb.grammar_corrections:
        with st.expander(f"Grammar corrections ({len(fb.grammar_corrections)})"):
            for correction in fb.grammar_corrections:
                st.markdown(f"~~{correction.original}~~ → **{correction.corrected}**")
                st.caption(correction.explanation)

    with st.expander("Improved answer"):
        st.write(fb.improved_answer)

    st.markdown(f"💡 **Tip:** {fb.one_tip}")

    if result.follow_up_asked:
        st.caption("A follow-up question is coming up next.")

    if st.button("Continue", type="primary"):
        st.session_state["current_question"] = None
        st.session_state["last_answer_result"] = None
        st.session_state["interview_stage"] = "asking"
        st.rerun()
