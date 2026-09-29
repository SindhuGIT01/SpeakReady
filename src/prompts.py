"""Prompt templates for SpeakReady's LLM calls.

Prompts are kept here (rather than inline in the modules that use them) so
they can be reviewed, tuned, and diffed independently of the calling code.
Each prompt is versioned with a ``_V<n>`` suffix; when a prompt's wording
changes in a way that could affect existing behavior, bump the suffix and
keep the old version around until nothing references it, so past feedback
runs stay reproducible against the version they were generated with.
"""

from __future__ import annotations

# --- Feedback engine (Task 7) -------------------------------------------------

FEEDBACK_SYSTEM_PROMPT_V1 = """You are an expert interview coach giving structured \
feedback on one spoken answer to an interview question.

Score and critique only the CONTENT of the answer (what was said and how well \
it addresses the question) — not fluency or delivery. A separate system \
already measures speaking pace, pauses, and filler words, so ignore those \
entirely.

Rules:
- content_score.score is 0-10: 0-3 is off-topic, wrong, or effectively \
missing; 4-6 partially answers the question; 7-8 is a solid, complete \
answer; 9-10 is exceptional, specific, and well-structured. content_score.reason \
is a single sentence explaining that score.
- key_points_covered and key_points_missed: if a list of points a strong \
answer should cover is given below, judge the answer against exactly that \
list. If no list is given, use your own judgment of what a strong answer to \
this question would include.
- grammar_corrections: list ONLY real grammatical or phrasing mistakes that \
actually appear in the transcript, each as the exact original phrase, its \
correction, and a short explanation. If the transcript has no real \
mistakes, return an empty list — never invent a correction to fill the list.
- improved_answer: rewrite the answer so it is clearer and better \
structured, in the candidate's own voice and first person. Use ONLY facts, \
experience, and details the candidate actually stated in their transcript, \
or that appear in the resume context if one is given. Never invent \
experience, projects, skills, employers, or achievements the candidate did \
not state.
- one_tip: the single most important, actionable thing the candidate should \
improve — one sentence.
"""


def build_feedback_user_prompt(
    question: str,
    transcript: str,
    what_good_answer_covers: list[str] | None = None,
    resume_context: str | None = None,
) -> str:
    """Build the human-turn prompt for :data:`FEEDBACK_SYSTEM_PROMPT_V1`.

    Args:
        question: The interview question that was asked.
        transcript: The candidate's transcribed spoken answer.
        what_good_answer_covers: Optional list of points a strong answer to
            this question should cover (from the question bank), used to
            ground ``key_points_covered``/``key_points_missed``.
        resume_context: Optional resume text/snippets relevant to this
            answer, used only to verify facts for ``improved_answer`` —
            never to invent new ones.

    Returns:
        The formatted prompt text to send as the human message.
    """
    sections = [
        f"Interview question:\n{question}",
        f"Candidate's spoken answer (transcribed):\n{transcript}",
    ]

    if what_good_answer_covers:
        points = "\n".join(f"- {point}" for point in what_good_answer_covers)
        sections.append(f"A strong answer to this question should cover:\n{points}")

    if resume_context:
        sections.append(
            "Relevant resume context (use only to verify facts the candidate "
            f"could truthfully draw on; never invent beyond it):\n{resume_context}"
        )

    return "\n\n".join(sections)
