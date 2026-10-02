"""PDF report generation for a completed interview session.

Renders a :class:`~src.interview_agent.SessionSummary` and its per-question
answers as a downloadable PDF for the Streamlit report page.
"""

from __future__ import annotations

import json
from typing import Any

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from src.interview_agent import SessionSummary

# The core Helvetica font only supports latin-1. LLM-generated text commonly
# includes "smart" typographic punctuation outside that range, so common
# characters are mapped to their plain-ASCII equivalents before rendering;
# anything else falls back to "?" instead of raising.
_UNICODE_TO_ASCII = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "…": "...",
        " ": " ",
    }
)


def _pdf_safe(text: str) -> str:
    """Make text safe for the core (latin-1-only) Helvetica font.

    Args:
        text: Arbitrary text, possibly containing Unicode punctuation.

    Returns:
        The text with common smart punctuation mapped to ASCII, and any
        remaining unsupported characters replaced with ``"?"``.
    """
    translated = text.translate(_UNICODE_TO_ASCII)
    return translated.encode("latin-1", errors="replace").decode("latin-1")


def _add_wrapped(pdf: FPDF, text: str, size: int = 11, style: str = "") -> None:
    """Write a left-aligned, word-wrapped paragraph.

    Args:
        pdf: The PDF document being built.
        text: The paragraph text.
        size: Font size in points.
        style: FPDF font style flags, e.g. ``"B"`` for bold.
    """
    pdf.set_font("Helvetica", style, size)
    # fpdf2's multi_cell defaults to new_x=RIGHT, which leaves the cursor
    # wherever the last line ended instead of the left margin; without
    # resetting it, the next multi_cell(w=0, ...) call can compute almost no
    # remaining width and raise "Not enough horizontal space".
    pdf.multi_cell(0, 6, _pdf_safe(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def build_report_pdf(
    role: str,
    difficulty: str | None,
    summary: SessionSummary,
    answers: list[dict[str, Any]],
) -> bytes:
    """Render a session report as a PDF.

    Args:
        role: The target role the interview was run for.
        difficulty: The difficulty filter used, if any.
        summary: The session's summary from
            :meth:`~src.interview_agent.InterviewSession.end`.
        answers: Answer rows from :func:`src.storage.list_answers`.

    Returns:
        The PDF file contents as bytes.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    _add_wrapped(pdf, "SpeakReady Interview Report", size=18, style="B")
    subtitle = role + (f" - {difficulty.title()} difficulty" if difficulty else "")
    _add_wrapped(pdf, subtitle, size=12)
    pdf.ln(4)

    _add_wrapped(pdf, "Summary", size=14, style="B")
    _add_wrapped(pdf, f"Questions answered: {summary.questions_answered}")
    _add_wrapped(pdf, f"Overall score: {summary.average_overall_score:.0f}/100")
    _add_wrapped(pdf, f"Fluency score: {summary.average_fluency_score:.0f}/100")
    _add_wrapped(pdf, f"Content score: {summary.average_content_score:.1f}/10")
    _add_wrapped(pdf, f"Strongest area: {summary.strongest_area}")
    _add_wrapped(pdf, f"Focus area: {summary.weakest_area}")
    pdf.ln(4)

    body_language_rows = [
        json.loads(row["body_language_json"]) for row in answers if row.get("body_language_json")
    ]
    if body_language_rows:
        _add_wrapped(pdf, "Body Language (heuristic)", size=14, style="B")
        avg_eye_contact = sum(r["eye_contact_ratio"] for r in body_language_rows) / len(
            body_language_rows
        )
        avg_posture = sum(r["posture_score"] for r in body_language_rows) / len(
            body_language_rows
        )
        total_looking_away = sum(r["looking_away_count"] for r in body_language_rows)
        _add_wrapped(
            pdf,
            f"Based on {len(body_language_rows)} of {len(answers)} answer(s) with the "
            "optional webcam toggle on.",
        )
        _add_wrapped(
            pdf,
            f"Avg. eye contact: {avg_eye_contact * 100:.0f}%  -  "
            f"Avg. posture: {avg_posture:.0f}/100  -  Looked away: {total_looking_away}x",
        )
        pdf.ln(4)

    voice_confidence_rows = [
        json.loads(row["voice_confidence_json"])
        for row in answers
        if row.get("voice_confidence_json")
    ]
    if voice_confidence_rows:
        _add_wrapped(pdf, "Voice Confidence (signal-processing heuristic)", size=14, style="B")
        avg_confidence = sum(r["confidence_score"] for r in voice_confidence_rows) / len(
            voice_confidence_rows
        )
        avg_pitch = sum(r["pitch_variability"] for r in voice_confidence_rows) / len(
            voice_confidence_rows
        )
        avg_volume = sum(r["volume_steadiness"] for r in voice_confidence_rows) / len(
            voice_confidence_rows
        )
        avg_energy = sum(r["speaking_energy"] for r in voice_confidence_rows) / len(
            voice_confidence_rows
        )
        _add_wrapped(
            pdf,
            f"Based on {len(voice_confidence_rows)} of {len(answers)} answer(s) with audio.",
        )
        _add_wrapped(
            pdf,
            f"Avg. confidence: {avg_confidence:.0f}/100  -  Pitch variation: "
            f"{avg_pitch:.0f}/100  -  Volume steadiness: {avg_volume:.0f}/100  -  "
            f"Speaking energy: {avg_energy:.0f}/100",
        )
        pdf.ln(4)

    pronunciation_rows = [
        json.loads(row["pronunciation_json"])
        for row in answers
        if row.get("pronunciation_json")
    ]
    total_flagged_words = sum(
        len(r["words_to_double_check"]) for r in pronunciation_rows
    )
    if pronunciation_rows and total_flagged_words:
        _add_wrapped(pdf, "Words To Double Check (ASR confidence proxy)", size=14, style="B")
        _add_wrapped(
            pdf,
            f"{total_flagged_words} word(s) across the session fell in the ASR's "
            "least-confident stretches of audio - not a pronunciation verdict, just "
            "worth a second listen.",
        )
        pdf.ln(4)

    _add_wrapped(pdf, "7-Day Practice Plan", size=14, style="B")
    for day in summary.practice_plan:
        _add_wrapped(pdf, f"Day {day.day} - {day.focus}", style="B")
        _add_wrapped(pdf, day.activity)
    pdf.ln(4)

    if answers:
        _add_wrapped(pdf, "Question by Question", size=14, style="B")
        for i, row in enumerate(answers, start=1):
            scores = json.loads(row["scores_json"])
            _add_wrapped(pdf, f"Q{i}. {row['question']}", style="B")
            area = row["area"] or "General"
            _add_wrapped(pdf, f"Score: {scores['overall_score']:.0f}/100  -  Area: {area}")
            pdf.ln(2)

    return bytes(pdf.output())
