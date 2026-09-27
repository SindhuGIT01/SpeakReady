"""Tests for the interview question bank and its Chroma-backed RAG retrieval."""

from __future__ import annotations

import pytest
from langchain_chroma import Chroma

from src.question_bank import (
    HR_CATEGORIES,
    TECHNICAL_CATEGORIES,
    _get_embeddings,
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
