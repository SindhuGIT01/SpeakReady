"""SQLite storage for interview sessions and answers.

Persists what :class:`src.interview_agent.InterviewSession` produces so a
candidate's history survives past one process, and so a future progress
tracker (Task 9) can be built directly on top of these tables.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src import config

_SESSIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    difficulty TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    summary_json TEXT
)
"""

_ANSWERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    question TEXT NOT NULL,
    is_followup INTEGER NOT NULL,
    area TEXT,
    transcript TEXT NOT NULL,
    scores_json TEXT NOT NULL,
    features_json TEXT NOT NULL,
    feedback_json TEXT NOT NULL,
    answered_at TEXT NOT NULL
)
"""


def init_db(conn: sqlite3.Connection) -> None:
    """Create the sessions and answers tables if they don't already exist.

    Args:
        conn: An open SQLite connection.
    """
    conn.execute(_SESSIONS_SCHEMA)
    conn.execute(_ANSWERS_SCHEMA)
    conn.commit()


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a SQLite connection with the sessions/answers tables ensured.

    Args:
        db_path: Path to the database file. Defaults to ``config.DB_PATH``.
            Pass ``":memory:"`` for an ephemeral database (mainly for
            testing).

    Returns:
        A ``sqlite3.Connection`` with ``row_factory`` set to
        ``sqlite3.Row`` so rows can be read like dicts.
    """
    path = ":memory:" if db_path == ":memory:" else str(db_path or config.DB_PATH)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def create_session(
    conn: sqlite3.Connection,
    session_id: str,
    role: str,
    difficulty: str | None,
    started_at: str,
) -> None:
    """Insert a new session row.

    Args:
        conn: An open SQLite connection.
        session_id: Unique identifier for the session.
        role: Target job role the session was run for.
        difficulty: Optional difficulty filter used for the session.
        started_at: ISO-8601 timestamp of when the session started.
    """
    conn.execute(
        "INSERT INTO sessions (id, role, difficulty, started_at) VALUES (?, ?, ?, ?)",
        (session_id, role, difficulty, started_at),
    )
    conn.commit()


def save_answer(
    conn: sqlite3.Connection,
    session_id: str,
    question: str,
    is_followup: bool,
    area: str,
    transcript: str,
    scores: dict[str, float],
    features: dict[str, float],
    feedback: dict[str, Any],
    answered_at: str,
) -> int:
    """Insert one answer row.

    Args:
        conn: An open SQLite connection.
        session_id: The session this answer belongs to.
        question: The question text that was answered.
        is_followup: Whether this was a follow-up question.
        area: The category (question bank) or resume section this question
            belongs to.
        transcript: The candidate's transcribed answer text.
        scores: Score summary, e.g. ``{"content_score": .., "fluency_score":
            .., "overall_score": ..}``.
        features: Fluency features from
            :func:`src.features.extract_features`.
        feedback: The full feedback payload, e.g. ``Feedback.model_dump()``.
        answered_at: ISO-8601 timestamp of when the answer was submitted.

    Returns:
        The autoincremented row id of the inserted answer.
    """
    cursor = conn.execute(
        """INSERT INTO answers
           (session_id, question, is_followup, area, transcript, scores_json,
            features_json, feedback_json, answered_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            question,
            int(is_followup),
            area,
            transcript,
            json.dumps(scores),
            json.dumps(features),
            json.dumps(feedback),
            answered_at,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def end_session(
    conn: sqlite3.Connection,
    session_id: str,
    ended_at: str,
    summary: dict[str, Any],
) -> None:
    """Record a session's end time and summary.

    Args:
        conn: An open SQLite connection.
        session_id: The session to update.
        ended_at: ISO-8601 timestamp of when the session ended.
        summary: The session summary payload, e.g.
            ``SessionSummary.model_dump()``.
    """
    conn.execute(
        "UPDATE sessions SET ended_at = ?, summary_json = ? WHERE id = ?",
        (ended_at, json.dumps(summary), session_id),
    )
    conn.commit()


def get_session(conn: sqlite3.Connection, session_id: str) -> dict[str, Any] | None:
    """Fetch one session row by id.

    Args:
        conn: An open SQLite connection.
        session_id: The session to fetch.

    Returns:
        A dict of the session row, or ``None`` if not found.
    """
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def list_answers(conn: sqlite3.Connection, session_id: str) -> list[dict[str, Any]]:
    """Fetch all answers for a session, in submission order.

    Args:
        conn: An open SQLite connection.
        session_id: The session whose answers to fetch.

    Returns:
        A list of answer row dicts, ordered by ``id``.
    """
    rows = conn.execute(
        "SELECT * FROM answers WHERE session_id = ? ORDER BY id", (session_id,)
    ).fetchall()
    return [dict(row) for row in rows]


def list_sessions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Fetch all sessions, most recently started first.

    Args:
        conn: An open SQLite connection.

    Returns:
        A list of session row dicts, ordered by ``started_at`` descending.
    """
    rows = conn.execute("SELECT * FROM sessions ORDER BY started_at DESC").fetchall()
    return [dict(row) for row in rows]
