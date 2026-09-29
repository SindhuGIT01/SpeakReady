"""Progress page: history and trends across all past sessions, from SQLite."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from app_common import get_db_connection
from src import storage

st.title("📈 Progress")

conn = get_db_connection()
sessions = storage.list_sessions(conn)
completed = [s for s in sessions if s["ended_at"] and s["summary_json"]]

if not completed:
    st.info("No completed interviews yet. Finish an interview to start tracking your progress.")
    if st.button("Start an interview", type="primary"):
        st.switch_page("app_pages/setup.py")
    st.stop()

rows: list[dict[str, object]] = []
area_scores: dict[str, list[float]] = {}

for sess in reversed(completed):  # oldest first, for left-to-right trend charts
    summary = json.loads(sess["summary_json"])
    answers = storage.list_answers(conn, sess["id"])
    features_list = [json.loads(a["features_json"]) for a in answers]

    avg_wpm = (
        sum(f["words_per_minute"] for f in features_list) / len(features_list)
        if features_list
        else 0.0
    )
    avg_filler_rate = (
        sum(f["filler_rate"] for f in features_list) / len(features_list) if features_list else 0.0
    )

    rows.append(
        {
            "Session": sess["started_at"][:10],
            "Role": sess["role"],
            "Overall score": summary["average_overall_score"],
            "Fluency score": summary["average_fluency_score"],
            "WPM": avg_wpm,
            "Filler rate": avg_filler_rate,
        }
    )

    for answer in answers:
        area = answer["area"] or "General"
        overall = json.loads(answer["scores_json"])["overall_score"]
        area_scores.setdefault(area, []).append(overall)

df = pd.DataFrame(rows)
df["Session #"] = range(1, len(df) + 1)

st.subheader("Score trend")
st.caption("Overall score, averaged per session.")
st.plotly_chart(
    px.line(df, x="Session #", y="Overall score", markers=True, hover_data=["Session", "Role"]),
    use_container_width=True,
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Filler word rate")
    st.caption("Filler words per 100 words, averaged per session.")
    st.plotly_chart(
        px.line(df, x="Session #", y="Filler rate", markers=True, hover_data=["Session"]),
        use_container_width=True,
    )
with col2:
    st.subheader("Speaking pace")
    st.caption("Words per minute, averaged per session.")
    st.plotly_chart(
        px.line(df, x="Session #", y="WPM", markers=True, hover_data=["Session"]),
        use_container_width=True,
    )

st.subheader("Weakest categories")
st.caption("Average overall score per question category or resume section, across all sessions.")
area_df = pd.DataFrame(
    [{"Area": area, "Average score": sum(scores) / len(scores)} for area, scores in area_scores.items()]
).sort_values("Average score")
st.plotly_chart(
    px.bar(
        area_df,
        x="Area",
        y="Average score",
        color="Average score",
        color_continuous_scale="RdYlGn",
        range_color=[0, 100],
    ),
    use_container_width=True,
)
