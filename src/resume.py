"""Resume parsing, per-session resume RAG, and personalized question generation.

Loads a resume PDF, splits and embeds it into a per-session Chroma collection
(so different users' resumes never mix), extracts a structured profile with
the LLM, and generates interview questions grounded in that profile — each
citing the resume section it came from.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from pypdf import PdfReader

from src import config


class ResumeParseError(ValueError):
    """Raised when a resume PDF has no extractable text (e.g. it is scanned)."""


class Project(BaseModel):
    """A single project extracted from a resume."""

    name: str = Field(description="The project's name as it appears on the resume.")
    tech: list[str] = Field(
        default_factory=list, description="Technologies/tools used in the project."
    )
    description: str = Field(description="What the project does or accomplished.")


class ResumeProfile(BaseModel):
    """Structured profile extracted from a candidate's resume."""

    name: str = Field(description="The candidate's full name.")
    skills: list[str] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    education: list[str] = Field(
        default_factory=list, description="Degrees/institutions, one entry each."
    )
    certifications: list[str] = Field(default_factory=list)


class ResumeQuestion(BaseModel):
    """A single personalized interview question grounded in the resume."""

    question: str = Field(description="The interview question text.")
    resume_section: str = Field(
        description=(
            "Which resume section this question is based on, e.g. 'Skills', "
            "'Education', 'Certifications', or 'Projects: <exact project name>'."
        )
    )


class _ResumeQuestionSet(BaseModel):
    """Wrapper so the LLM's structured output is a single top-level object."""

    questions: list[ResumeQuestion]


def load_resume(pdf_path: Path) -> str:
    """Extract text from a resume PDF.

    Args:
        pdf_path: Path to the resume PDF file.

    Returns:
        The concatenated text of every page.

    Raises:
        ResumeParseError: If no extractable text is found (e.g. the PDF is a
            scanned image with no text layer).
    """
    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    if not text.strip():
        raise ResumeParseError(
            f"No extractable text found in '{pdf_path}'. It may be a scanned "
            "image PDF with no text layer — OCR it first, or upload a "
            "text-based PDF."
        )
    return text


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached embeddings model instance."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


def _get_resume_vectorstore(
    session_id: str,
    persist_directory: str | None = None,
) -> Chroma:
    """Build (or reopen) the per-session Chroma collection for a resume.

    Args:
        session_id: Unique identifier for the user's session; each session
            gets its own collection so resumes never mix between users.
        persist_directory: Directory to persist the collection to. Defaults
            to ``config.CHROMA_DIR``.

    Returns:
        A ``Chroma`` vector store instance scoped to this session.
    """
    return Chroma(
        collection_name=f"{config.RESUME_COLLECTION}_{session_id}",
        embedding_function=_get_embeddings(),
        persist_directory=str(persist_directory or config.CHROMA_DIR),
    )


def index_resume(
    text: str,
    session_id: str,
    vectorstore: Chroma | None = None,
) -> int:
    """Chunk a resume's text and embed it into a per-session Chroma collection.

    Args:
        text: Full resume text, as returned by :func:`load_resume`.
        session_id: Unique identifier for the user's session.
        vectorstore: Optional pre-built vector store (mainly for testing).
            Defaults to the persistent collection for ``session_id``.

    Returns:
        The number of chunks indexed.
    """
    store = vectorstore if vectorstore is not None else _get_resume_vectorstore(session_id)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_text(text)
    if not chunks:
        return 0

    store.add_texts(
        texts=chunks,
        ids=[f"{session_id}_{i}" for i in range(len(chunks))],
    )
    return len(chunks)


_PROFILE_SYSTEM_PROMPT = """You extract structured profile data from resumes.
Only include information that is explicitly present in the resume text. Never
invent, infer, or embellish skills, projects, education, or certifications
that are not stated. If a field has no information in the resume, leave it
empty."""


def extract_profile(text: str, llm: BaseChatModel | None = None) -> ResumeProfile:
    """Extract a structured profile from resume text using the LLM.

    Args:
        text: Full resume text, as returned by :func:`load_resume`.
        llm: Optional chat model to use (mainly for testing). Defaults to
            :func:`src.llm.get_llm`.

    Returns:
        A :class:`ResumeProfile` populated from the resume.

    Raises:
        config.ConfigError: If no ``llm`` is given and GROQ_API_KEY is missing.
    """
    from src.llm import get_llm

    model = llm if llm is not None else get_llm()
    structured_model = model.with_structured_output(ResumeProfile)

    result = structured_model.invoke(
        [
            SystemMessage(content=_PROFILE_SYSTEM_PROMPT),
            HumanMessage(content=f"Resume text:\n\n{text}"),
        ]
    )
    return result if isinstance(result, ResumeProfile) else ResumeProfile.model_validate(result)


_QUESTIONS_SYSTEM_PROMPT = """You write personalized interview questions based
strictly on a candidate's resume profile.

Rules:
- Only ask about projects, skills, education, or certifications that are
  listed in the profile below. Never invent a project, skill, or detail that
  is not present.
- Each question must reference something specific from the profile (e.g. a
  named project, a specific technology, a specific degree) rather than being
  generic.
- For every question, set resume_section to exactly one of: "Skills",
  "Education", "Certifications", or "Projects: <exact project name from the
  profile>"."""


def _format_profile_for_prompt(profile: ResumeProfile) -> str:
    """Render a profile as plain text for the question-generation prompt.

    Args:
        profile: The extracted resume profile.

    Returns:
        A human-readable listing of the profile's fields.
    """
    lines = [f"Name: {profile.name}"]
    if profile.skills:
        lines.append("Skills: " + ", ".join(profile.skills))
    if profile.education:
        lines.append("Education: " + "; ".join(profile.education))
    if profile.certifications:
        lines.append("Certifications: " + ", ".join(profile.certifications))
    if profile.projects:
        lines.append("Projects:")
        for project in profile.projects:
            tech = ", ".join(project.tech)
            lines.append(f"  - {project.name} (tech: {tech}): {project.description}")
    return "\n".join(lines)


def _valid_resume_sections(profile: ResumeProfile) -> set[str]:
    """Build the set of resume_section values that are grounded in the profile.

    Args:
        profile: The extracted resume profile.

    Returns:
        The set of valid section labels a generated question may cite.
    """
    sections = {"Skills", "Education", "Certifications"}
    sections.update(f"Projects: {project.name}" for project in profile.projects)
    return sections


def generate_resume_questions(
    profile: ResumeProfile,
    n: int = 5,
    llm: BaseChatModel | None = None,
) -> list[ResumeQuestion]:
    """Generate personalized interview questions grounded in a resume profile.

    Args:
        profile: The extracted resume profile to base questions on.
        n: Number of questions to generate.
        llm: Optional chat model to use (mainly for testing). Defaults to
            :func:`src.llm.get_llm`.

    Returns:
        Up to ``n`` :class:`ResumeQuestion` objects, each citing the resume
        section it came from. Questions citing a section not actually present
        in the profile (a hallucinated project, for instance) are dropped.

    Raises:
        config.ConfigError: If no ``llm`` is given and GROQ_API_KEY is missing.
    """
    from src.llm import get_llm

    model = llm if llm is not None else get_llm()
    structured_model = model.with_structured_output(_ResumeQuestionSet)

    prompt = (
        f"{_format_profile_for_prompt(profile)}\n\n"
        f"Generate exactly {n} personalized interview questions about this "
        "candidate's resume."
    )
    result = structured_model.invoke(
        [
            SystemMessage(content=_QUESTIONS_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    question_set = (
        result
        if isinstance(result, _ResumeQuestionSet)
        else _ResumeQuestionSet.model_validate(result)
    )

    valid_sections = _valid_resume_sections(profile)
    grounded = [q for q in question_set.questions if q.resume_section in valid_sections]
    return grounded[:n]
