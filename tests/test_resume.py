"""Tests for resume parsing, per-session resume RAG, and question generation.

LLM calls are mocked throughout so these tests run without a GROQ_API_KEY.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_chroma import Chroma

from src.resume import (
    Project,
    ResumeParseError,
    ResumeProfile,
    ResumeQuestion,
    _ResumeQuestionSet,
    _get_embeddings,
    extract_profile,
    generate_resume_questions,
    index_resume,
    load_resume,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_RESUME_PDF = FIXTURES_DIR / "sample_resume.pdf"


@pytest.fixture
def vectorstore(tmp_path) -> Chroma:
    """A throwaway Chroma collection so tests never touch the real chroma_db/."""
    return Chroma(
        collection_name="test_resume_session",
        embedding_function=_get_embeddings(),
        persist_directory=str(tmp_path),
    )


@pytest.fixture
def profile() -> ResumeProfile:
    """A hand-built profile matching tests/fixtures/sample_resume.pdf."""
    return ResumeProfile(
        name="Aditi Sharma",
        skills=["Python", "SQL", "XGBoost", "scikit-learn", "Docker", "AWS"],
        projects=[
            Project(
                name="CustomerIQ",
                tech=["Python", "XGBoost", "Flask", "PostgreSQL"],
                description="Churn prediction system for a retail client.",
            ),
            Project(
                name="PriceWise",
                tech=["React", "Node.js", "PostgreSQL"],
                description="Dynamic pricing dashboard.",
            ),
        ],
        education=["B.Tech in Computer Science, Vellore Institute of Technology, 2024"],
        certifications=["AWS Certified Cloud Practitioner (2023)"],
    )


# --- load_resume -------------------------------------------------------------


def test_load_resume_extracts_text() -> None:
    """Text extracted from the sample PDF should contain known resume content."""
    text = load_resume(SAMPLE_RESUME_PDF)

    assert "Aditi Sharma" in text
    assert "CustomerIQ" in text
    assert "XGBoost" in text


def test_load_resume_raises_on_empty_pdf(monkeypatch) -> None:
    """A scanned/empty PDF (no text layer) should raise a clear error."""
    blank_page = MagicMock()
    blank_page.extract_text.return_value = ""
    fake_reader = MagicMock()
    fake_reader.pages = [blank_page]

    monkeypatch.setattr("src.resume.PdfReader", lambda _path: fake_reader)

    with pytest.raises(ResumeParseError):
        load_resume(Path("fake_scanned_resume.pdf"))


# --- index_resume --------------------------------------------------------------


def test_index_resume_chunks_and_stores_text(vectorstore: Chroma) -> None:
    """Indexing should add at least one chunk and preserve resume content."""
    text = load_resume(SAMPLE_RESUME_PDF)

    added = index_resume(text, session_id="session-1", vectorstore=vectorstore)

    assert added > 0
    stored = vectorstore.get(include=["documents"])
    assert len(stored["ids"]) == added
    assert any("CustomerIQ" in doc for doc in stored["documents"])


def test_index_resume_sessions_do_not_mix(tmp_path) -> None:
    """Two sessions must use distinct collections so resumes never mix."""
    store_a = Chroma(
        collection_name="test_resume_session_a",
        embedding_function=_get_embeddings(),
        persist_directory=str(tmp_path),
    )
    store_b = Chroma(
        collection_name="test_resume_session_b",
        embedding_function=_get_embeddings(),
        persist_directory=str(tmp_path),
    )

    index_resume("Resume A: knows Python and Java.", session_id="a", vectorstore=store_a)
    index_resume("Resume B: knows Rust and Go.", session_id="b", vectorstore=store_b)

    docs_a = store_a.get(include=["documents"])["documents"]
    docs_b = store_b.get(include=["documents"])["documents"]

    assert any("Python" in d for d in docs_a)
    assert not any("Rust" in d for d in docs_a)
    assert any("Rust" in d for d in docs_b)
    assert not any("Python" in d for d in docs_b)


# --- extract_profile -----------------------------------------------------------


def test_extract_profile_uses_structured_output(profile: ResumeProfile) -> None:
    """extract_profile should invoke structured output and return the profile."""
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = profile
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    result = extract_profile("some resume text", llm=mock_llm)

    mock_llm.with_structured_output.assert_called_once_with(ResumeProfile)
    assert result == profile


# --- generate_resume_questions -------------------------------------------------


def test_generate_resume_questions_cites_real_sections(profile: ResumeProfile) -> None:
    """Generated questions should be returned as-is when all sections are valid."""
    mock_question_set = _ResumeQuestionSet(
        questions=[
            ResumeQuestion(
                question="In your CustomerIQ project, why did you choose XGBoost?",
                resume_section="Projects: CustomerIQ",
            ),
            ResumeQuestion(
                question="How did you use PostgreSQL in PriceWise?",
                resume_section="Projects: PriceWise",
            ),
            ResumeQuestion(
                question="Tell me about your experience with Docker.",
                resume_section="Skills",
            ),
        ]
    )
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = mock_question_set
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    questions = generate_resume_questions(profile, n=3, llm=mock_llm)

    assert len(questions) == 3
    assert all(isinstance(q, ResumeQuestion) for q in questions)
    assert any("CustomerIQ" in q.question for q in questions)


def test_generate_resume_questions_drops_hallucinated_projects(
    profile: ResumeProfile,
) -> None:
    """A question citing a project not in the profile must never be invented."""
    mock_question_set = _ResumeQuestionSet(
        questions=[
            ResumeQuestion(
                question="In your CustomerIQ project, why did you choose XGBoost?",
                resume_section="Projects: CustomerIQ",
            ),
            ResumeQuestion(
                question="Tell me about your imaginary project, RoboChef.",
                resume_section="Projects: RoboChef",
            ),
        ]
    )
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = mock_question_set
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    questions = generate_resume_questions(profile, n=2, llm=mock_llm)

    assert len(questions) == 1
    assert questions[0].resume_section == "Projects: CustomerIQ"
