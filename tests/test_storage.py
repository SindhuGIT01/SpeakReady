"""Tests for SQLite session/answer storage.

All tests use an in-memory database (``get_connection(":memory:")``) so they
never touch the real ``speakready.db`` file.
"""

from __future__ import annotations

import sqlite3

import pytest

from src.storage import (
    create_session,
    end_session,
    get_connection,
    get_session,
    list_answers,
    list_sessions,
    save_answer,
)


@pytest.fixture
def conn() -> sqlite3.Connection:
    """An in-memory database connection with tables already created."""
    return get_connection(":memory:")


def test_get_connection_creates_tables(conn: sqlite3.Connection) -> None:
    """A fresh connection should already have the sessions/answers tables."""
    tables = {
        row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"sessions", "answers"} <= tables


def test_create_session_and_get_session(conn: sqlite3.Connection) -> None:
    """A created session should be fetchable with the same field values."""
    create_session(conn, "sess-1", "Backend Engineer", "medium", "2026-01-01T00:00:00")

    row = get_session(conn, "sess-1")

    assert row is not None
    assert row["id"] == "sess-1"
    assert row["role"] == "Backend Engineer"
    assert row["difficulty"] == "medium"
    assert row["started_at"] == "2026-01-01T00:00:00"
    assert row["ended_at"] is None
    assert row["summary_json"] is None


def test_get_session_returns_none_when_missing(conn: sqlite3.Connection) -> None:
    """Fetching an unknown session id should return None, not raise."""
    assert get_session(conn, "does-not-exist") is None


def test_save_answer_returns_incrementing_row_ids(conn: sqlite3.Connection) -> None:
    """Each saved answer should get its own autoincremented id."""
    create_session(conn, "sess-1", "Backend Engineer", None, "2026-01-01T00:00:00")

    first_id = save_answer(
        conn,
        "sess-1",
        "Tell me about yourself.",
        False,
        "HR",
        "I am a software engineer.",
        {"overall_score": 80.0},
        {"words_per_minute": 140.0},
        {"content_score": 8},
        "2026-01-01T00:01:00",
    )
    second_id = save_answer(
        conn,
        "sess-1",
        "Why this role?",
        True,
        "HR",
        "Because I like building things.",
        {"overall_score": 70.0},
        {"words_per_minute": 130.0},
        {"content_score": 7},
        "2026-01-01T00:02:00",
    )

    assert second_id == first_id + 1


def test_list_answers_returns_rows_in_submission_order(conn: sqlite3.Connection) -> None:
    """Answers should come back ordered by insertion (question order)."""
    create_session(conn, "sess-1", "Backend Engineer", None, "2026-01-01T00:00:00")
    save_answer(
        conn,
        "sess-1",
        "Q1",
        False,
        "HR",
        "A1",
        {"overall_score": 80.0},
        {"words_per_minute": 140.0},
        {"content_score": 8},
        "2026-01-01T00:01:00",
    )
    save_answer(
        conn,
        "sess-1",
        "Q2",
        False,
        "HR",
        "A2",
        {"overall_score": 90.0},
        {"words_per_minute": 150.0},
        {"content_score": 9},
        "2026-01-01T00:02:00",
    )

    rows = list_answers(conn, "sess-1")

    assert [row["question"] for row in rows] == ["Q1", "Q2"]
    assert rows[0]["is_followup"] == 0
    assert rows[0]["scores_json"] == '{"overall_score": 80.0}'


def test_list_answers_empty_for_unknown_session(conn: sqlite3.Connection) -> None:
    """A session with no answers should return an empty list, not raise."""
    assert list_answers(conn, "does-not-exist") == []


def test_end_session_sets_ended_at_and_summary(conn: sqlite3.Connection) -> None:
    """Ending a session should persist its end time and summary payload."""
    create_session(conn, "sess-1", "Backend Engineer", None, "2026-01-01T00:00:00")

    end_session(conn, "sess-1", "2026-01-01T00:10:00", {"questions_answered": 2})

    row = get_session(conn, "sess-1")
    assert row["ended_at"] == "2026-01-01T00:10:00"
    assert row["summary_json"] == '{"questions_answered": 2}'


def test_list_sessions_orders_most_recent_first(conn: sqlite3.Connection) -> None:
    """Sessions should be listed newest-started first."""
    create_session(conn, "sess-old", "Backend Engineer", None, "2026-01-01T00:00:00")
    create_session(conn, "sess-new", "Frontend Engineer", None, "2026-02-01T00:00:00")

    rows = list_sessions(conn)

    assert [row["id"] for row in rows] == ["sess-new", "sess-old"]
