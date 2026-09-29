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
| 5 | Speech feature extraction | ✅ Done |
| 6 | ML fluency model | ✅ Done |
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

`src/features.py` turns a `Transcript` into fluency features with
`extract_features(transcript)` — speaking pace (WPM), pause count/duration,
filler word rate (configurable list, with a heuristic to skip "like" when
used as an ordinary verb), word repetition, vocabulary variety (type-token
ratio), and mean sentence length. The same function is used at training time
for the fluency model (Task 6) and in the app, so features never drift out of
sync. `explain_features(features)` turns the numbers into plain-language
coaching tips.

`notebooks/fluency_model.ipynb` trains the fluency classifier (Task 6) on
[speechocean762](https://huggingface.co/datasets/mispeech/speechocean762), a
public dataset of non-native English speakers with expert fluency scores.
`src/scorer.py` loads the saved model and exposes `predict_fluency(features)`,
which returns a `FluencyPrediction(label, confidence, score)` — a Beginner/
Intermediate/Fluent label, the model's confidence in it, and a continuous
0-100 score. See **Model results** below for how it was trained and how well
it performs.

## Model results

Trained in `notebooks/fluency_model.ipynb` on
[speechocean762](https://huggingface.co/datasets/mispeech/speechocean762)
(2,500 train / 2,500 test utterances, its built-in split). The 0-10 expert
fluency score was bucketed into 3 classes — Beginner (0-5), Intermediate
(6-7), Fluent (8-10) — and four candidates were tuned with 5-fold
stratified cross-validation on the train split, then evaluated once on the
held-out test split. All numbers below are copied directly from
`models/metrics.json`.

| Model | Test accuracy | Test macro F1 | CV macro F1 (train) |
|---|---|---|---|
| Baseline (majority class) | 68.0% | 0.270 | — |
| Logistic Regression | 70.0% | 0.598 | 0.638 |
| **Random Forest (winner)** | **72.7%** | **0.633** | 0.649 |
| XGBoost | 72.4% | 0.619 | 0.664 |

**Winner: Random Forest** (`max_depth=10, min_samples_leaf=3,
n_estimators=400`), selected by test macro F1. It comfortably beats the
majority-class baseline — accuracy alone barely moves (68.0% → 72.7%)
because ~66% of the dataset is already "Fluent," but macro F1 more than
doubles (0.270 → 0.633), showing it's actually learning the minority
Beginner/Intermediate classes rather than just guessing the majority one.
Per-class performance (test set): Beginner F1 0.570, Intermediate F1 0.488,
Fluent F1 0.841 — the model is most confident distinguishing clearly fluent
speech and struggles most on the Beginner/Intermediate boundary, which is
also where human raters tend to disagree most.

**Limitations:** speechocean762 is *read-aloud* speech — non-native
speakers reading a fixed sentence aloud — not spontaneous interview answers.
Fluency signals in free speech (self-correction, thinking pauses, filler
words while formulating an answer) differ from, and are often more varied
than, fluency signals in reading a known sentence. Treat this model as a
reasonable starting point, not a validated measure of spontaneous-speech
fluency for SpeakReady's actual interview use case. Transcripts were also
produced by `faster-whisper`'s local `small` model rather than the
Groq-hosted `whisper-large-v3-turbo` used in production, so any features
downstream of transcription errors carry that mismatch too.
