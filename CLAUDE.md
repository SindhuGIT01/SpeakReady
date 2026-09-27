# SpeakReady — AI Voice Interview Coach

Portfolio project. A user uploads their resume and picks a role; the app interviews
them by voice, transcribes spoken answers, scores fluency with a trained ML model,
gives LLM-based feedback, asks follow-up questions, and tracks progress over time.

## Tech stack (all free)
- **Python 3.10+**, **Streamlit** UI
- **LLM:** Groq API via `langchain-groq` — model `openai/gpt-oss-120b` (set in `src/config.py`)
- **Speech-to-text:** Groq Whisper API — `whisper-large-v3-turbo`
- **RAG:** LangChain + Chroma + HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- **ML:** scikit-learn / XGBoost fluency score model (features via librosa)
- **Text-to-speech:** gTTS
- **Storage:** SQLite for session history
- **Tests:** pytest

## Folder structure
```
speakready/
├── app.py              # Streamlit entry point
├── src/
│   ├── __init__.py
│   ├── config.py       # ALL settings: model names, paths, chunk sizes, env loading
│   └── llm.py          # get_llm() -> ChatGroq
├── data/
│   ├── questions/      # interview question bank (committed)
│   └── resumes/        # user uploads (git-ignored)
├── models/             # trained ML models
├── notebooks/          # exploration / model training
├── tests/              # pytest tests
├── requirements.txt
├── .env.example        # template; real secrets go in .env (git-ignored)
└── README.md
```

## Coding rules
- Type hints on all function signatures.
- Google-style docstrings on every module, class, and public function.
- Keep functions small and single-purpose; prefer pure functions where possible.
- **No hardcoded API keys or secrets.** Keys come from `.env` via `src/config.py`.
- **All settings live in `src/config.py`** (model names, paths, chunk sizes, top-k, etc.).
  Other modules import from `src.config`; never duplicate constants.
- Call `config.get_groq_api_key()` (raises `ConfigError` with a clear message) rather
  than reading `GROQ_API_KEY` directly.
- Tests that hit external APIs must skip automatically when the key is missing.
- Import project code as `from src import ...`; run commands from the project root.

## Git rules
- Do **not** add `Co-Authored-By: Claude` (or any Claude/AI attribution trailer) to commit messages.

## Commands
- Run app: `streamlit run app.py`
- Run tests: `python -m pytest -v`

## Roadmap
| # | Task | Status |
|---|------|--------|
| 1 | Setup: structure, config, Groq LLM client | ✅ Done |
| 2 | Question bank RAG | ⬜ |
| 3 | Resume RAG + resume-based questions | ⬜ |
| 4 | Speech-to-text (Groq Whisper) | ⬜ |
| 5 | Speech feature extraction | ⬜ |
| 6 | ML fluency model | ⬜ |
| 7 | LLM feedback engine | ⬜ |
| 8 | Interview agent with follow-ups + voice (gTTS) | ⬜ |
| 9 | Streamlit app + progress tracker (SQLite) | ⬜ |
| 10 | Tests, README, deployment | ⬜ |
