"""Interview page: plays each question, records an answer, and shows feedback."""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from app_common import render_timeline_chart
from src import config
from src import speech as speech_mod
from src.features import extract_features
from src.feedback import FeedbackError
from src.interview_agent import InterviewError
from src.scorer import ScorerError
from src.speech import AudioValidationError, TranscriptionError
from src.timeline import build_timeline

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
        st.session_state["body_language_frames"] = []
        st.session_state["_last_webcam_snapshot_bytes"] = None
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

    st.divider()
    webcam_enabled = st.toggle(
        "📷 Track eye contact & posture (optional)",
        value=st.session_state.get("webcam_enabled", False),
        help=(
            "Interviews aren't just about what you say — this practices your "
            "presence too. Take a snapshot or two of yourself while you answer "
            "out loud and we'll add heuristic eye-contact and posture feedback "
            "alongside your fluency and content scores. Nothing is required: "
            "the interview works exactly the same without it, and without "
            "webcam/camera permission."
        ),
    )
    st.session_state["webcam_enabled"] = webcam_enabled

    if webcam_enabled:
        snapshot = st.camera_input(
            "Take a snapshot or two while you answer",
            key=f"webcam_snapshot_{question.question_id}",
        )
        if snapshot is not None:
            snapshot_bytes = snapshot.getvalue()
            if snapshot_bytes != st.session_state.get("_last_webcam_snapshot_bytes"):
                st.session_state["_last_webcam_snapshot_bytes"] = snapshot_bytes
                frame = cv2.imdecode(np.frombuffer(snapshot_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    st.session_state["body_language_frames"].append(frame)

        frame_count = len(st.session_state.get("body_language_frames", []))
        if frame_count:
            count_col, clear_col = st.columns([3, 1])
            count_col.caption(f"📸 {frame_count} snapshot(s) captured for this answer.")
            if clear_col.button("Clear"):
                st.session_state["body_language_frames"] = []
                st.session_state["_last_webcam_snapshot_bytes"] = None
                st.rerun()

    if audio_value is not None and st.button("Submit Answer", type="primary"):
        try:
            audio_bytes = audio_value.getvalue()
            with st.spinner("Transcribing your answer..."):
                transcript = speech_mod.transcribe(audio_bytes)

            # "Live" filler alert (Task 12): Streamlit can't stream mic audio
            # word-by-word, so this is the fastest honest substitute — a quick
            # count computed locally (no LLM call) and shown immediately,
            # before the slower full-feedback call below even starts.
            markers = build_timeline(transcript)
            quick_features = extract_features(transcript)
            fast_speech_count = sum(1 for m in markers if m.type == "fast_speech")
            st.success(
                f"⚡ Quick signal: {quick_features['filler_count']:.0f} filler word(s), "
                f"{quick_features['long_pause_count']:.0f} long pause(s), "
                f"{fast_speech_count} fast-speech stretch(es)"
            )

            body_language_frames = st.session_state.get("body_language_frames") or None
            with st.spinner("Scoring fluency and generating full feedback..."):
                result = session.submit_transcript(
                    transcript,
                    body_language_frames=body_language_frames,
                    audio_for_confidence=audio_bytes,
                )

            st.session_state["last_answer_audio"] = audio_bytes
            st.session_state["last_answer_markers"] = markers
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

    last_answer_audio = st.session_state.get("last_answer_audio")
    if last_answer_audio:
        st.caption("Replay your answer:")
        st.audio(last_answer_audio, format="audio/wav")

    markers = st.session_state.get("last_answer_markers") or []
    if markers:
        st.plotly_chart(
            render_timeline_chart(markers, result.transcript.duration_seconds),
            use_container_width=True,
        )
        st.caption(
            "🟧 filler word · 🟦 long pause · 🟥 fast speech — hover a bar for details."
        )

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

    if result.body_language_result is not None:
        bl = result.body_language_result
        with st.container(border=True):
            st.markdown("**📷 Body language (heuristic)**")
            bl_col1, bl_col2, bl_col3 = st.columns(3)
            bl_col1.metric("Eye contact", f"{bl.eye_contact_ratio * 100:.0f}%")
            bl_col2.metric("Posture", f"{bl.posture_score:.0f}/100")
            bl_col3.metric("Looked away", f"{bl.looking_away_count}x")
            st.caption(bl.summary)

    if result.voice_confidence_result is not None:
        vc = result.voice_confidence_result
        with st.container(border=True):
            st.markdown("**🎚️ Voice confidence (signal-processing heuristic)**")
            vc_col1, vc_col2, vc_col3, vc_col4 = st.columns(4)
            vc_col1.metric("Confidence", f"{vc.confidence_score:.0f}/100")
            vc_col2.metric("Pitch variation", f"{vc.pitch_variability:.0f}/100")
            vc_col3.metric("Volume steadiness", f"{vc.volume_steadiness:.0f}/100")
            vc_col4.metric("Speaking energy", f"{vc.speaking_energy:.0f}/100")
            st.caption(vc.summary)

    if result.pronunciation_result is not None:
        pr = result.pronunciation_result
        with st.container(border=True):
            st.markdown("**🗣️ Words to double check (ASR confidence proxy)**")
            st.caption(pr.explanation)
            if pr.words_to_double_check:
                st.markdown(
                    ", ".join(
                        f'"{w.word}" at {w.start:.1f}s' for w in pr.words_to_double_check
                    )
                )

    st.markdown(f"💡 **Tip:** {fb.one_tip}")

    if result.follow_up_asked:
        st.caption("A follow-up question is coming up next.")

    if st.button("Continue", type="primary"):
        st.session_state["current_question"] = None
        st.session_state["last_answer_result"] = None
        st.session_state["last_answer_audio"] = None
        st.session_state["last_answer_markers"] = []
        st.session_state["interview_stage"] = "asking"
        st.rerun()
