"""Home page: what SpeakReady does, and a call to action to start an interview."""

from __future__ import annotations

import streamlit as st

st.title("🎤 SpeakReady")
st.subheader("Your AI voice interview coach")

st.markdown(
    "SpeakReady runs a real mock interview with you — **by voice**. Upload your "
    "resume, pick a target role, and start talking."
)

st.divider()

col1, col2 = st.columns(2, gap="large")
with col1:
    with st.container(border=True):
        st.markdown("#### 🎧 Listens & scores")
        st.write("Transcribes every spoken answer and scores your fluency with a trained ML model.")
    with st.container(border=True):
        st.markdown("#### 🧠 Real feedback")
        st.write("LLM-based feedback on content, grammar, and a rewritten improved answer.")
    with st.container(border=True):
        st.markdown("#### 💬 Natural follow-ups")
        st.write("Asks follow-up questions on the fly, just like a real interviewer would.")
with col2:
    with st.container(border=True):
        st.markdown("#### 📝 A practice plan")
        st.write("Ends every session with a personalized 7-day plan built from your weak spots.")
    with st.container(border=True):
        st.markdown("#### 📊 A full report")
        st.write("Session summary, per-question scores, and a downloadable PDF report.")
    with st.container(border=True):
        st.markdown("#### 📈 Progress over time")
        st.write("Tracks your score, filler-word rate, and speaking pace across sessions.")

st.divider()

if st.button("Start Interview", type="primary"):
    st.switch_page("app_pages/setup.py")

st.caption("Runs entirely on your machine — your resume and answers never leave it.")
