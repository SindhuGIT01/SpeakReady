"""Tests for PDF report generation."""

from __future__ import annotations

import json

from src.interview_agent import PracticeDay, SessionSummary
from src.report import _pdf_safe, build_report_pdf


def _sample_summary() -> SessionSummary:
    """A minimal SessionSummary for report tests."""
    return SessionSummary(
        questions_answered=2,
        average_overall_score=82.5,
        average_fluency_score=75.0,
        average_content_score=8.5,
        strongest_area="Python",
        weakest_area="SQL",
        practice_plan=[
            PracticeDay(day=1, focus="Filler words", activity="Record a 2-minute answer daily."),
            PracticeDay(day=2, focus="Pacing", activity="Practice with a metronome at 140 WPM."),
        ],
    )


def _sample_answers() -> list[dict]:
    """Answer rows shaped like src.storage.list_answers output."""
    return [
        {
            "question": "Tell me about a challenging project.",
            "area": "Behavioral",
            "scores_json": json.dumps({"overall_score": 88.0}),
        },
        {
            "question": "Explain normalization in SQL.",
            "area": None,
            "scores_json": json.dumps({"overall_score": 60.0}),
        },
    ]


def test_pdf_safe_maps_smart_punctuation_to_ascii() -> None:
    """Curly quotes and dashes should become their plain-ASCII equivalents."""
    assert _pdf_safe("‘hi’ — “test”…") == "'hi' - \"test\"..."


def test_pdf_safe_replaces_unsupported_characters() -> None:
    """Characters with no ASCII mapping should be replaced, not raise."""
    assert _pdf_safe("café 你好") == "café ??"


def test_build_report_pdf_returns_valid_pdf_bytes() -> None:
    """The output should be non-empty bytes starting with the PDF file signature."""
    pdf_bytes = build_report_pdf("Backend Engineer", "medium", _sample_summary(), _sample_answers())

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 0


def test_build_report_pdf_handles_no_difficulty_and_no_answers() -> None:
    """Optional difficulty and an empty answers list should not raise."""
    pdf_bytes = build_report_pdf("Backend Engineer", None, _sample_summary(), [])

    assert pdf_bytes.startswith(b"%PDF")
