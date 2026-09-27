"""Interview question bank with retrieval-augmented (semantic) search.

Questions live in ``data/questions/question_bank.json`` and are embedded into
a persistent Chroma collection so :func:`get_questions` can retrieve the
questions most relevant to a given role, optionally narrowed by difficulty.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src import config

QUESTION_BANK_PATH = config.QUESTIONS_DIR / "question_bank.json"

HR_CATEGORIES: list[str] = ["HR", "Behavioral"]
TECHNICAL_CATEGORIES: list[str] = [
    "Java",
    "Python",
    "SQL",
    "Machine Learning",
    "AWS/Cloud",
    "Web Development",
]


def load_question_bank(path: Path = QUESTION_BANK_PATH) -> list[dict[str, Any]]:
    """Load the raw question bank from disk.

    Args:
        path: Path to the question bank JSON file.

    Returns:
        A list of question records as dictionaries.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached embeddings model instance."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


def _get_vectorstore(
    collection_name: str = config.QUESTIONS_COLLECTION,
    persist_directory: str | None = None,
) -> Chroma:
    """Build (or reopen) the Chroma vector store for the question bank.

    Args:
        collection_name: Name of the Chroma collection to use.
        persist_directory: Directory to persist the collection to. Defaults
            to ``config.CHROMA_DIR``.

    Returns:
        A ``Chroma`` vector store instance.
    """
    return Chroma(
        collection_name=collection_name,
        embedding_function=_get_embeddings(),
        persist_directory=str(persist_directory or config.CHROMA_DIR),
    )


def _to_document_text(question: dict[str, Any]) -> str:
    """Build the embedded text for a question record.

    Includes the category so semantic search on a job role (e.g. "Java
    Developer") matches questions from the corresponding category even when
    the role wording doesn't appear verbatim in the question text.

    Args:
        question: A question record from the question bank.

    Returns:
        The text to embed for this question.
    """
    return f"{question['category']}: {question['question']}"


def _to_metadata(question: dict[str, Any]) -> dict[str, Any]:
    """Build the Chroma metadata for a question record.

    Args:
        question: A question record from the question bank.

    Returns:
        A flat metadata dict (Chroma metadata values must be primitives).
    """
    return {
        "id": question["id"],
        "question": question["question"],
        "category": question["category"],
        "difficulty": question["difficulty"],
        "what_good_answer_covers": " | ".join(question["what_good_answer_covers"]),
    }


def _from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct a question record from stored Chroma metadata.

    Args:
        metadata: Metadata dict previously produced by :func:`_to_metadata`.

    Returns:
        A question record dict matching the question bank's schema.
    """
    return {
        "id": metadata["id"],
        "question": metadata["question"],
        "category": metadata["category"],
        "difficulty": metadata["difficulty"],
        "what_good_answer_covers": metadata["what_good_answer_covers"].split(" | "),
    }


def ingest_questions(vectorstore: Chroma | None = None) -> int:
    """Embed the question bank into the Chroma ``question_bank`` collection.

    Safe to re-run: questions whose ``id`` already exists in the collection
    are skipped, so re-ingesting never creates duplicates.

    Args:
        vectorstore: Optional pre-built vector store (mainly for testing).
            Defaults to the shared persistent collection.

    Returns:
        The number of new questions that were added.
    """
    store = vectorstore if vectorstore is not None else _get_vectorstore()
    questions = load_question_bank()

    existing_ids = set(store.get(include=[])["ids"])
    new_questions = [q for q in questions if q["id"] not in existing_ids]

    if new_questions:
        store.add_texts(
            texts=[_to_document_text(q) for q in new_questions],
            metadatas=[_to_metadata(q) for q in new_questions],
            ids=[q["id"] for q in new_questions],
        )

    return len(new_questions)


def _build_filter(
    difficulty: str | None,
    categories: list[str],
) -> dict[str, Any]:
    """Build a Chroma ``where`` filter for difficulty and category.

    Args:
        difficulty: Optional exact difficulty to filter on.
        categories: Categories the result must belong to.

    Returns:
        A Chroma-compatible metadata filter.
    """
    clauses: list[dict[str, Any]] = [{"category": {"$in": categories}}]
    if difficulty:
        clauses.append({"difficulty": difficulty})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def get_questions(
    role: str,
    difficulty: str | None = None,
    n: int = 5,
    vectorstore: Chroma | None = None,
) -> list[dict[str, Any]]:
    """Retrieve a set of interview questions relevant to a role.

    Combines semantic search against the role name with metadata filters on
    difficulty, and always mixes in one or two HR/Behavioral questions
    alongside the role-relevant technical questions.

    Args:
        role: Target job role, e.g. "Java Developer", "Data Analyst",
            "ML Engineer".
        difficulty: Optional difficulty filter ("easy", "medium", "hard").
        n: Total number of questions to return.
        vectorstore: Optional pre-built vector store (mainly for testing).

    Returns:
        A list of up to ``n`` question record dicts.
    """
    store = vectorstore if vectorstore is not None else _get_vectorstore()

    hr_count = min(2 if n >= 4 else 1, n)
    technical_count = n - hr_count

    results: list[dict[str, Any]] = []

    if technical_count > 0:
        technical_filter = _build_filter(difficulty, TECHNICAL_CATEGORIES)
        technical_docs = store.similarity_search(
            role, k=technical_count, filter=technical_filter
        )
        results.extend(_from_metadata(doc.metadata) for doc in technical_docs)

    if hr_count > 0:
        hr_filter = _build_filter(difficulty, HR_CATEGORIES)
        hr_docs = store.similarity_search(role, k=hr_count, filter=hr_filter)
        results.extend(_from_metadata(doc.metadata) for doc in hr_docs)

    return results


def _main() -> None:
    """CLI entry point: ``python -m src.question_bank --ingest``."""
    parser = argparse.ArgumentParser(description="Manage the interview question bank.")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Embed data/questions/question_bank.json into the Chroma vector store.",
    )
    args = parser.parse_args()

    if args.ingest:
        added = ingest_questions()
        total = len(load_question_bank())
        print(f"Ingested {added} new question(s); {total} total in the question bank.")
    else:
        parser.print_help()


if __name__ == "__main__":
    _main()
