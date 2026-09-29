"""Shared Streamlit session-state and resource helpers for the SpeakReady app.

Kept separate from ``src/`` because it is UI-runtime glue (Streamlit caching
and session state) rather than framework-independent business logic.
"""

from __future__ import annotations

import sqlite3

import streamlit as st

from src import config, question_bank, storage
from src.llm import get_llm

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
}


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
