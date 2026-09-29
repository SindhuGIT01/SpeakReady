"""Setup page: choose role and difficulty, optionally upload a resume."""

from __future__ import annotations

import uuid

import streamlit as st

from app_common import DIFFICULTY_OPTIONS, ROLE_OPTIONS, get_db_connection, get_llm_cached
from src import config
from src.interview_agent import InterviewSession
from src.resume import ResumeParseError, extract_profile, load_resume

st.title("⚙️ Interview Setup")
st.caption("Tell us who you're interviewing for, and optionally add your resume.")

role = st.selectbox("Target role", ROLE_OPTIONS, index=0)
difficulty = st.select_slider("Difficulty", options=DIFFICULTY_OPTIONS, value="medium")
num_questions = st.slider("Number of questions", min_value=3, max_value=10, value=6)

st.divider()
st.subheader("Resume (optional)")
st.caption("Upload your resume to get a few personalized questions about your real experience.")

uploaded = st.file_uploader("Resume (PDF)", type=["pdf"])

if uploaded is not None:
    if st.session_state.get("_resume_uploaded_name") != uploaded.name:
        st.session_state["_resume_uploaded_name"] = uploaded.name
        st.session_state["resume_profile"] = None
        st.session_state["resume_use"] = False

    if st.session_state.get("resume_profile") is None:
        with st.spinner("Reading your resume..."):
            try:
                config.RESUMES_DIR.mkdir(parents=True, exist_ok=True)
                dest = config.RESUMES_DIR / f"{uuid.uuid4()}.pdf"
                dest.write_bytes(uploaded.getvalue())
                text = load_resume(dest)
                profile = extract_profile(text, llm=get_llm_cached())
                st.session_state["resume_profile"] = profile
                st.session_state["resume_use"] = True
            except ResumeParseError as exc:
                st.error(f"Couldn't read that resume: {exc}")
            except config.ConfigError:
                st.error(
                    "The AI service isn't configured yet, so resumes can't be "
                    "analyzed right now."
                )
            except Exception:
                st.error(
                    "Something went wrong while reading your resume. Please try "
                    "a different file."
                )

profile = st.session_state.get("resume_profile")
if profile is not None:
    st.success("Resume parsed. Here's what we found — confirm it looks right:")
    with st.container(border=True):
        st.markdown(f"**{profile.name or 'Candidate'}**")
        if profile.skills:
            st.markdown("**Skills:** " + ", ".join(profile.skills))
        if profile.education:
            st.markdown("**Education:** " + "; ".join(profile.education))
        if profile.certifications:
            st.markdown("**Certifications:** " + ", ".join(profile.certifications))
        if profile.projects:
            st.markdown("**Projects:**")
            for project in profile.projects:
                tech = ", ".join(project.tech)
                st.markdown(f"- **{project.name}** ({tech}): {project.description}")
        if not (profile.skills or profile.education or profile.certifications or profile.projects):
            st.caption("No structured details were found in this resume.")

    st.session_state["resume_use"] = st.checkbox(
        "Use my resume to personalize some interview questions",
        value=st.session_state.get("resume_use", True),
    )

st.divider()

if st.button("Begin Interview", type="primary"):
    try:
        with st.spinner("Preparing your interview..."):
            llm = get_llm_cached()
            conn = get_db_connection()
            session = InterviewSession(llm=llm, conn=conn)
            use_resume = profile is not None and st.session_state.get("resume_use")
            session.start(
                role=role,
                difficulty=difficulty,
                resume_profile=profile if use_resume else None,
                num_questions=num_questions,
            )
        st.session_state["interview_session"] = session
        st.session_state["interview_stage"] = "asking"
        st.session_state["current_question"] = None
        st.session_state["last_answer_result"] = None
        st.session_state["session_summary"] = None
        st.session_state["session_role"] = role
        st.session_state["session_difficulty"] = difficulty
        st.switch_page("app_pages/interview.py")
    except config.ConfigError:
        st.error(
            "The AI service isn't configured yet. Please check the app's API "
            "key setup and try again."
        )
    except Exception:
        st.error("Something went wrong while preparing your interview. Please try again.")
