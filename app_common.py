"""Shared Streamlit session-state and resource helpers for the SpeakReady app.

Kept separate from ``src/`` because it is UI-runtime glue (Streamlit caching
and session state) rather than framework-independent business logic.
"""

from __future__ import annotations

import sqlite3

import plotly.graph_objects as go
import streamlit as st

from src import config, question_bank, storage
from src.llm import get_llm
from src.timeline import Marker

# Role choices offered on the Setup page. Passed straight through as the
# `role` argument to InterviewSession.start(), which uses it both as the
# semantic search query against the question bank and as the human-readable
# role name in prompts (e.g. the practice plan).
ROLE_OPTIONS: list[str] = [
    "Python Developer",
    "Java Developer",
    "SQL / Data Analyst",
    "Machine Learning Engineer",
    "AWS / Cloud Engineer",
    "Web Developer",
    "HR / Behavioral",
]

DIFFICULTY_OPTIONS: list[str] = ["easy", "medium", "hard"]

_SESSION_STATE_DEFAULTS: dict[str, object] = {
    "resume_profile": None,
    "resume_use": False,
    "interview_session": None,
    "interview_stage": "idle",  # idle | asking | reviewing | finished
    "current_question": None,
    "last_answer_result": None,
    "session_summary": None,
    "session_role": None,
    "session_difficulty": None,
    "webcam_enabled": False,  # Task 11: optional eye contact / posture coaching
    "body_language_frames": [],
    "_last_webcam_snapshot_bytes": None,
    "last_answer_audio": None,  # Task 12: replay timeline
    "last_answer_markers": [],
}

# Timeline marker colors/labels for the Task 12 replay chart, keyed by
# src.timeline.Marker.type.
_MARKER_STYLE: dict[str, tuple[str, str]] = {
    "filler": ("#f5a623", "Filler word"),  # amber
    "long_pause": ("#1f77b4", "Long pause"),  # blue
    "fast_speech": ("#e74c3c", "Fast speech"),  # red
}
# A near-instant marker (e.g. a single short filler word) still needs a
# visible width on the chart.
_MIN_MARKER_SECONDS = 0.15


def init_session_state() -> None:
    """Populate ``st.session_state`` with the default keys the app relies on."""
    for key, value in _SESSION_STATE_DEFAULTS.items():
        st.session_state.setdefault(key, value)


@st.cache_resource(show_spinner=False)
def get_llm_cached():
    """Return a process-wide cached ChatGroq client.

    Raises:
        config.ConfigError: If GROQ_API_KEY is not configured.
    """
    return get_llm()


def get_db_connection() -> sqlite3.Connection:
    """Return this browser session's SQLite connection, opening it if needed.

    ``check_same_thread=False`` because Streamlit can rerun a session's
    script on a different worker thread than the one that first opened the
    connection; the connection is only ever touched by one rerun at a time.
    """
    if "db_conn" not in st.session_state:
        conn = sqlite3.connect(str(config.DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        storage.init_db(conn)
        st.session_state.db_conn = conn
    return st.session_state.db_conn


@st.cache_resource(show_spinner=False)
def ensure_question_bank_ingested() -> int:
    """Embed the question bank into Chroma once per server process.

    Returns:
        The number of newly added questions (0 if already ingested).
    """
    return question_bank.ingest_questions()


def render_timeline_chart(markers: list[Marker], total_duration: float) -> go.Figure:
    """Build the Task 12 answer replay timeline as a Plotly horizontal bar chart.

    One trace per marker type (filler/long_pause/fast_speech) so the legend
    doubles as a color key; hovering (or tapping, on mobile) a bar shows its
    label, e.g. ``'filler: "um" at 4.2s'``.

    Args:
        markers: Markers from :func:`src.timeline.build_timeline`, in any order.
        total_duration: The answer's total duration in seconds, used to size
            the time axis.

    Returns:
        A Plotly figure ready for ``st.plotly_chart``.
    """
    fig = go.Figure()
    for marker_type, (color, legend_label) in _MARKER_STYLE.items():
        type_markers = [m for m in markers if m.type == marker_type]
        if not type_markers:
            continue
        fig.add_trace(
            go.Bar(
                x=[max(m.end - m.start, _MIN_MARKER_SECONDS) for m in type_markers],
                y=["Answer"] * len(type_markers),
                base=[m.start for m in type_markers],
                orientation="h",
                name=legend_label,
                marker_color=color,
                hovertext=[m.label for m in type_markers],
                hoverinfo="text",
                width=0.6,
            )
        )

    fig.update_layout(
        barmode="overlay",
        height=160,
        margin={"l": 10, "r": 10, "t": 10, "b": 40},
        xaxis={"title": "Seconds into your answer", "range": [0, max(total_duration, 0.1)]},
        yaxis={"visible": False},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig
