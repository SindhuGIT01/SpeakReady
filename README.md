# SpeakReady

An AI voice interview coach that interviews you by voice based on your resume and target role, scores your fluency with a trained ML model, and gives LLM-powered feedback.

> 🚧 **Work in progress** — this project is being built step by step.

## Progress

| # | Task | Status |
|---|------|--------|
| 1 | Setup: structure, config, Groq LLM client | ✅ Done |
| 2 | Question bank RAG | ✅ Done |
| 3 | Resume RAG + resume-based questions | ✅ Done |
| 4 | Speech-to-text (Groq Whisper) | ✅ Done |
| 5 | Speech feature extraction | ⬜ |
| 6 | ML fluency model | ⬜ |
| 7 | LLM feedback engine | ⬜ |
| 8 | Interview agent with follow-ups + voice (gTTS) | ⬜ |
| 9 | Streamlit app + progress tracker (SQLite) | ⬜ |
| 10 | Tests, README, deployment | ⬜ |

The question bank (`data/questions/question_bank.json`) has 159 fresher interview
questions across HR, Behavioral, Java, Python, SQL, Machine Learning, AWS/Cloud, and
Web Development. `src/question_bank.py` embeds them into a persistent Chroma
collection and retrieves a relevant, role-aware set with `get_questions(role,
difficulty, n)`. Re-ingest anytime with:

```bash
python -m src.question_bank --ingest
```

`src/resume.py` handles per-user resume RAG: `load_resume(pdf_path)` extracts text
with pypdf (raising a clear error on scanned/empty PDFs), `index_resume(text,
session_id)` chunks and embeds it into a Chroma collection scoped to that session
so resumes never mix between users, `extract_profile(text)` uses the LLM with
structured output (Pydantic) to pull out name/skills/projects/education/
certifications, and `generate_resume_questions(profile, n)` generates personalized
questions that each cite the resume section they came from and are filtered to
drop anything not grounded in the actual profile.

`src/speech.py` transcribes spoken answers with Groq's hosted Whisper API:
`transcribe(audio)` accepts raw bytes or a file path (wav/mp3/m4a/webm),
validates format, size, and minimum duration up front with friendly error
messages, retries automatically with backoff on rate limits (HTTP 429), and
returns a `Transcript` (text, word-level timestamps, duration, language).
`src/tts.py` provides `speak(text)`, which synthesizes speech with gTTS and
caches clips on disk (keyed by text + language) so repeated interview
questions are never regenerated.
