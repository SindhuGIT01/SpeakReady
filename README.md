# SpeakReady

An AI voice interview coach that interviews you by voice based on your resume and target role, scores your fluency with a trained ML model, and gives LLM-powered feedback.

> 🚧 **Work in progress** — this project is being built step by step.

## Progress

| # | Task | Status |
|---|------|--------|
| 1 | Setup: structure, config, Groq LLM client | ✅ Done |
| 2 | Question bank RAG | ✅ Done |
| 3 | Resume RAG + resume-based questions | ⬜ |
| 4 | Speech-to-text (Groq Whisper) | ⬜ |
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
