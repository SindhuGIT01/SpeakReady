"""Tests for the interview question bank and its Chroma-backed RAG retrieval."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_chroma import Chroma
from pydantic import ValidationError

from src.question_bank import (
    HR_CATEGORIES,
    TECHNICAL_CATEGORIES,
    GeneratedQuestion,
    QuestionGenerationError,
    _GeneratedQuestionSet,
    _get_embeddings,
    generate_custom_role_questions,
    get_questions,
    ingest_questions,
    load_question_bank,
)

VALID_CATEGORIES = set(HR_CATEGORIES) | set(TECHNICAL_CATEGORIES)
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


@pytest.fixture
def vectorstore(tmp_path) -> Chroma:
    """A throwaway Chroma collection so tests never touch the real chroma_db/."""
    return Chroma(
        collection_name="test_question_bank",
        embedding_function=_get_embeddings(),
        persist_directory=str(tmp_path),
    )


def test_question_bank_has_at_least_150_well_formed_questions() -> None:
    """The committed question bank must satisfy the schema the app relies on."""
    questions = load_question_bank()
    assert len(questions) >= 150

    seen_ids = set()
    for q in questions:
        assert q["id"] not in seen_ids, f"duplicate id in question bank: {q['id']}"
        seen_ids.add(q["id"])
        assert q["question"].strip()
        assert q["category"] in VALID_CATEGORIES
        assert q["difficulty"] in VALID_DIFFICULTIES
        assert 2 <= len(q["what_good_answer_covers"]) <= 4


def test_ingest_questions_adds_every_question(vectorstore: Chroma) -> None:
    """A fresh collection should gain exactly as many entries as the bank has."""
    total = len(load_question_bank())

    added = ingest_questions(vectorstore=vectorstore)

    assert added == total
    assert len(vectorstore.get(include=[])["ids"]) == total


def test_ingest_questions_is_idempotent(vectorstore: Chroma) -> None:
    """Re-running ingestion must not create duplicate entries."""
    total = len(load_question_bank())

    first_added = ingest_questions(vectorstore=vectorstore)
    second_added = ingest_questions(vectorstore=vectorstore)

    assert first_added == total
    assert second_added == 0
    assert len(vectorstore.get(include=[])["ids"]) == total


def test_get_questions_filters_by_difficulty(vectorstore: Chroma) -> None:
    """Every returned question must match the requested difficulty."""
    ingest_questions(vectorstore=vectorstore)

    results = get_questions("Java Developer", "easy", 5, vectorstore=vectorstore)

    assert len(results) == 5
    assert all(q["difficulty"] == "easy" for q in results)


def test_get_questions_includes_hr_or_behavioral(vectorstore: Chroma) -> None:
    """A set of 5+ questions should always include 1-2 HR/Behavioral questions."""
    ingest_questions(vectorstore=vectorstore)

    results = get_questions("Python Developer", None, 5, vectorstore=vectorstore)
    categories = [q["category"] for q in results]
    hr_count = sum(1 for c in categories if c in HR_CATEGORIES)

    assert 1 <= hr_count <= 2


def test_get_questions_matches_role_semantically(vectorstore: Chroma) -> None:
    """A role search should surface questions from that role's own category."""
    ingest_questions(vectorstore=vectorstore)

    results = get_questions("Java Developer", "easy", 5, vectorstore=vectorstore)
    categories = [q["category"] for q in results]

    assert "Java" in categories


def test_get_questions_respects_requested_count(vectorstore: Chroma) -> None:
    """The number of returned questions should match n for small and large n."""
    ingest_questions(vectorstore=vectorstore)

    assert len(get_questions("SQL Developer", None, 3, vectorstore=vectorstore)) == 3
    assert len(get_questions("SQL Developer", None, 8, vectorstore=vectorstore)) == 8


# --- generate_custom_role_questions ---------------------------------------------


def test_generate_custom_role_questions_matches_bank_schema() -> None:
    """Generated questions must carry every key get_questions() results have."""
    mock_question_set = _GeneratedQuestionSet(
        questions=[
            GeneratedQuestion(
                question="How would you prioritize conflicting stakeholder requests?",
                category="Stakeholder Management",
                difficulty="medium",
                what_good_answer_covers=[
                    "A concrete prioritization framework",
                    "An example of a real trade-off made",
                ],
            ),
            GeneratedQuestion(
                question="Tell me about a time you had to push back on a deadline.",
                category="Behavioral",
                difficulty="medium",
                what_good_answer_covers=["Situation, action, outcome"],
            ),
        ]
    )
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = mock_question_set
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    questions = generate_custom_role_questions(
        "Business Analyst", difficulty="medium", n=2, llm=mock_llm
    )

    mock_llm.with_structured_output.assert_called_once_with(_GeneratedQuestionSet)
    assert len(questions) == 2
    for q in questions:
        assert set(q.keys()) == {
            "id",
            "question",
            "category",
            "difficulty",
            "what_good_answer_covers",
        }
        assert q["id"]
        assert q["difficulty"] == "medium"
    assert any("stakeholder" in q["question"].lower() for q in questions)


def test_generate_custom_role_questions_respects_requested_count() -> None:
    """Only the first n generated questions should be returned."""
    mock_question_set = _GeneratedQuestionSet(
        questions=[
            GeneratedQuestion(
                question=f"Question {i}",
                category="DevOps",
                difficulty="easy",
                what_good_answer_covers=["A point"],
            )
            for i in range(5)
        ]
    )
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = mock_question_set
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    questions = generate_custom_role_questions("DevOps Engineer", n=3, llm=mock_llm)

    assert len(questions) == 3


def test_generate_custom_role_questions_falls_back_on_invalid_difficulty() -> None:
    """An out-of-range difficulty from the LLM should fall back, never crash."""
    mock_question_set = _GeneratedQuestionSet(
        questions=[
            GeneratedQuestion(
                question="Describe your CI/CD pipeline experience.",
                category="DevOps",
                difficulty="expert",  # not one of easy/medium/hard
                what_good_answer_covers=["Concrete pipeline example"],
            )
        ]
    )
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = mock_question_set
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    questions = generate_custom_role_questions(
        "DevOps Engineer", difficulty="hard", n=1, llm=mock_llm
    )

    assert questions[0]["difficulty"] == "hard"


def test_generate_custom_role_questions_retries_on_malformed_output() -> None:
    """A single bad/unparseable LLM response should be retried, not raised."""
    mock_question_set = _GeneratedQuestionSet(
        questions=[
            GeneratedQuestion(
                question="Describe your CI/CD pipeline experience.",
                category="DevOps",
                difficulty="medium",
                what_good_answer_covers=["Concrete pipeline example"],
            )
        ]
    )
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = [ValueError("malformed JSON"), mock_question_set]
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    questions = generate_custom_role_questions("DevOps Engineer", n=1, llm=mock_llm)

    assert mock_structured.invoke.call_count == 2
    assert len(questions) == 1


def test_generate_custom_role_questions_raises_after_exhausting_retries() -> None:
    """Persistent invalid output should raise a clear error, not an unhandled one."""
    mock_structured = MagicMock()
    mock_structured.invoke.side_effect = ValidationError.from_exception_data(
        "_GeneratedQuestionSet", []
    )
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured

    with pytest.raises(QuestionGenerationError):
        generate_custom_role_questions("DevOps Engineer", n=1, llm=mock_llm)
