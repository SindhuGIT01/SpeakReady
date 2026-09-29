"""Report page: session summary, per-question chart, practice plan, and PDF download."""

from __future__ import annotations

import json

import streamlit as st

from app_common import get_db_connection
from src import storage
from src.report import build_report_pdf

st.title("📊 Interview Report")

session = st.session_state.get("interview_session")
summary = st.session_state.get("session_summary")

if session is None or summary is None:
    st.info("No completed interview yet. Finish an interview to see your report here.")
    if st.button("Go to Setup", type="primary"):
        st.switch_page("app_pages/setup.py")
    st.stop()

role = st.session_state.get("session_role") or session.role
difficulty = st.session_state.get("session_difficulty") or session.difficulty

st.subheader(role + (f" · {difficulty.title()} difficulty" if difficulty else ""))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Questions answered", summary.questions_answered)
c2.metric("Overall score", f"{summary.average_overall_score:.0f}/100")
c3.metric("Fluency score", f"{summary.average_fluency_score:.0f}/100")
c4.metric("Content score", f"{summary.average_content_score:.1f}/10")

col_a, col_b = st.columns(2)
col_a.success(f"💪 Strongest area: **{summary.strongest_area}**")
col_b.warning(f"🎯 Focus area: **{summary.weakest_area}**")

st.divider()

conn = get_db_connection()
answers = storage.list_answers(conn, session.session_id)

if answers:
    st.subheader("Score by question")
    chart_data = {
        "Question": [f"Q{i + 1}" for i in range(len(answers))],
        "Overall score": [json.loads(a["scores_json"])["overall_score"] for a in answers],
    }
    st.bar_chart(chart_data, x="Question", y="Overall score")
    st.divider()

st.subheader("7-day practice plan")
for day in summary.practice_plan:
    with st.container(border=True):
        st.markdown(f"**Day {day.day} — {day.focus}**")
        st.write(day.activity)

st.divider()

pdf_bytes = build_report_pdf(role=role, difficulty=difficulty, summary=summary, answers=answers)
st.download_button(
    "Download report (PDF)",
    data=pdf_bytes,
    file_name="speakready_report.pdf",
    mime="application/pdf",
    type="primary",
)
